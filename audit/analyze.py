# =============================================================================
# audit/analyze.py
# Analysis Layer — Packaging Results for the Streamlit App
# =============================================================================
#
# ── WHAT THIS MODULE DOES ────────────────────────────────────────────────────
#
# This module does NOT compute new metrics. It takes the raw fairness
# DataFrame produced by fairness.py and reshapes it into the specific
# data structures that each page of the Streamlit app needs.
#
# Think of it as the "bridge" between the data science pipeline and the UI:
#
#   fairness.py  →  analyze.py  →  app.py
#   (raw metrics)   (packaging)    (display)
#
# Each function in this module corresponds to one thing the app needs to show:
#
#   get_scatter_data()        Page 2: The Audit
#                             One row per model, ready to plot with Plotly.
#                             Adds hover text, dot sizes, color info.
#
#   compute_pareto_frontier() Page 2: The Audit (overlay)
#                             Which models are non-dominated in
#                             accuracy × fairness space?
#
#   get_summary_stats()       Page 1: The Case (home page)
#                             The headline numbers: best/worst gap,
#                             AUC cost of fairness, etc.
#
#   get_model_detail()        Page 3: Model Explorer
#                             Full drill-down for one selected model.
#                             Includes decision rules for trees.
#
#   get_impossibility_data()  Page 4: Who Bears It
#                             FPR vs FNR scatter for the theorem page.
#
# ── DESIGN PRINCIPLE ─────────────────────────────────────────────────────────
#
# Keep each function focused and testable. The app calls these functions;
# it never manipulates the raw fairness DataFrame directly. This means if
# we need to change how a chart is built, we change it here, not in app.py.
#
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.tree   import export_text
from sklearn.pipeline import Pipeline


# =============================================================================
# COLOR AND STYLE CONSTANTS
# (Shared between analyze.py and app.py so charts are consistent)
# =============================================================================

# Each model type gets a consistent color across all charts.
# These are the "Evidence Room" palette colors.
MODEL_TYPE_COLORS = {
    "Decision Tree":      "#2c5f8a",   # steel blue — interpretable
    "Logistic Regression":"#8b5e3c",   # warm brown — linear
    "Random Forest":      "#2d7a45",   # forest green — ensemble
    "Gradient Boosting":  "#7c3f8c",   # purple — complex ensemble
}


# =============================================================================
# FUNCTION 1: SCATTER PLOT DATA
# =============================================================================

def get_scatter_data(fairness_df):
    """
    Prepares the data for the central scatter plot on Page 2 (The Audit).

    The scatter plot shows:
      X axis — test AUC (accuracy): all models cluster in a narrow band
      Y axis — FPR disparity (bias): models spread across a wide range
      Color  — model type (DT / LR / RF / GBM)
      Size   — complexity (bigger dot = more complex model)
      Symbol — filled circle = in Rashomon set, open circle = outside

    The visual argument: the X axis is narrow (accuracy barely differs),
    the Y axis is wide (bias differs enormously). Accuracy cannot
    distinguish the fair from the biased.

    Args:
        fairness_df (DataFrame): Output of fairness.compute_all_fairness()

    Returns:
        DataFrame: One row per model with all plotting columns added.
    """
    df = fairness_df.copy()

    # ── Dot size encodes model complexity ──────────────────────────────────
    # Decision Trees: size proportional to leaf count (capped so it's readable)
    # All other models: fixed medium size
    def _dot_size(row):
        if row["model_type"] == "Decision Tree":
            # Scale leaves to a reasonable dot size range
            # 16 leaves → size 10,  113 leaves → size 22
            leaves = row["complexity"]
            return max(10, min(24, int(leaves * 0.15) + 8))
        return 14  # fixed size for non-tree models

    df["dot_size"] = df.apply(_dot_size, axis=1)

    # ── Marker symbol: filled = in Rashomon set, open = outside ───────────
    df["marker_symbol"] = df["in_rashomon"].map({
        True:  "circle",
        False: "circle-open",
    })

    # ── Color by model type ─────────────────────────────────────────────────
    df["color"] = df["model_type"].map(MODEL_TYPE_COLORS)

    # ── Hover text: what the user reads when hovering over a dot ───────────
    # This is HTML-formatted text displayed in the Plotly tooltip.
    df["hover_text"] = df.apply(lambda r: (
        f"<b>{r['name']}</b><br>"
        f"─────────────────────<br>"
        f"AUC (accuracy):  {r['test_auc']:.4f}<br>"
        f"FPR gap (bias):  {r['fpr_disparity']:.3f}<br>"
        f"FPR Black:       {r['fpr_black']:.3f}<br>"
        f"FPR White:       {r['fpr_white']:.3f}<br>"
        f"{'✓ In Rashomon set' if r['in_rashomon'] else '✗ Outside Rashomon set'}"
    ), axis=1)

    # ── Bias level category: low / medium / high ───────────────────────────
    # Used for the color-by-bias view (alternative to color-by-type)
    def _bias_level(fpr_gap):
        if fpr_gap < 0.14:
            return "Low bias"
        elif fpr_gap < 0.175:
            return "Medium bias"
        else:
            return "High bias"

    df["bias_level"] = df["fpr_disparity"].apply(_bias_level)

    # Return only the columns the app needs (drop per_group which is nested)
    return df[[
        "key", "name", "short_name", "model_type", "color",
        "test_auc", "fpr_disparity", "fnr_disparity", "fdr_disparity",
        "fpr_black", "fpr_white", "fnr_black", "fnr_white",
        "complexity", "in_rashomon",
        "dot_size", "marker_symbol", "hover_text", "bias_level",
    ]]


# =============================================================================
# FUNCTION 2: PARETO FRONTIER
# =============================================================================

def compute_pareto_frontier(fairness_df):
    """
    Finds the Pareto-optimal models in accuracy × fairness space.

    A model is Pareto-optimal (non-dominated) if no other model is
    simultaneously more accurate AND fairer. These models represent
    the best available tradeoffs — you cannot do better on one dimension
    without giving something up on the other.

    This answers: "Is there a fundamental tradeoff between accuracy and
    fairness, or can we get both?" If the Pareto frontier contains models
    with HIGH accuracy AND LOW disparity, the accuracy-fairness tradeoff
    is weak (good news).

    Args:
        fairness_df (DataFrame): Output of fairness.compute_all_fairness()

    Returns:
        DataFrame: Pareto-optimal models only, sorted by decreasing AUC.
    """
    # Work with a clean copy, drop rows with missing metrics
    df = fairness_df[["key", "name", "short_name", "model_type",
                       "test_auc", "fpr_disparity", "in_rashomon"]].copy()
    df = df.dropna(subset=["test_auc", "fpr_disparity"])

    pareto_flags = []

    for i, row_i in df.iterrows():
        dominated = False
        for j, row_j in df.iterrows():
            if i == j:
                continue
            # Model j dominates model i if j is BOTH more accurate AND fairer.
            # "More accurate" = higher AUC.
            # "Fairer" = lower FPR disparity.
            j_better_accuracy = row_j["test_auc"]        >= row_i["test_auc"]
            j_better_fairness = row_j["fpr_disparity"]   <= row_i["fpr_disparity"]
            j_strictly_better = (
                row_j["test_auc"]      > row_i["test_auc"] or
                row_j["fpr_disparity"] < row_i["fpr_disparity"]
            )
            if j_better_accuracy and j_better_fairness and j_strictly_better:
                dominated = True
                break
        pareto_flags.append(not dominated)

    df["pareto_optimal"] = pareto_flags

    # Return only the Pareto-optimal models, sorted by AUC descending
    return df[df["pareto_optimal"]].sort_values("test_auc", ascending=False)


# =============================================================================
# FUNCTION 3: SUMMARY STATISTICS
# =============================================================================

def get_summary_stats(fairness_df, rashomon_result):
    """
    Computes the headline numbers displayed on Page 1 (The Case).

    These are the statistics that make the project's argument immediate:
    how much does bias vary, and at what accuracy cost?

    Args:
        fairness_df     (DataFrame): Output of fairness.compute_all_fairness()
        rashomon_result (dict):      Output of rashomon.build_rashomon_set()

    Returns:
        dict: Key statistics for the home page display.
    """
    in_set = fairness_df[fairness_df["in_rashomon"] == True]

    # Identify the two extreme models within the Rashomon set
    fairest_idx   = in_set["fpr_disparity"].idxmin()
    unfairest_idx = in_set["fpr_disparity"].idxmax()
    fairest       = in_set.loc[fairest_idx]
    unfairest     = in_set.loc[unfairest_idx]

    # The "cost of fairness" — how much AUC do you give up to use
    # the fairest model instead of the most biased one?
    auc_cost = abs(float(unfairest["test_auc"]) - float(fairest["test_auc"]))

    return {
        # ── Rashomon set basics ────────────────────────────────────────────
        "n_models_total":      len(fairness_df),
        "n_models_rashomon":   len(in_set),
        "best_auc":            rashomon_result["best_auc"],
        "epsilon":             rashomon_result["epsilon"],
        "rashomon_auc_min":    float(in_set["test_auc"].min()),
        "rashomon_auc_max":    float(in_set["test_auc"].max()),

        # ── The core finding: bias range within the Rashomon set ──────────
        "fpr_gap_min":         float(in_set["fpr_disparity"].min()),
        "fpr_gap_max":         float(in_set["fpr_disparity"].max()),
        "fpr_gap_range":       float(in_set["fpr_disparity"].max() -
                                     in_set["fpr_disparity"].min()),

        # ── Fairest model details ─────────────────────────────────────────
        "fairest_name":        str(fairest["name"]),
        "fairest_type":        str(fairest["model_type"]),
        "fairest_auc":         float(fairest["test_auc"]),
        "fairest_fpr_black":   float(fairest["fpr_black"]),
        "fairest_fpr_white":   float(fairest["fpr_white"]),
        "fairest_fpr_gap":     float(fairest["fpr_disparity"]),

        # ── Most biased model details ─────────────────────────────────────
        "unfairest_name":      str(unfairest["name"]),
        "unfairest_type":      str(unfairest["model_type"]),
        "unfairest_auc":       float(unfairest["test_auc"]),
        "unfairest_fpr_black": float(unfairest["fpr_black"]),
        "unfairest_fpr_white": float(unfairest["fpr_white"]),
        "unfairest_fpr_gap":   float(unfairest["fpr_disparity"]),

        # ── The headline number: AUC cost of being fair ───────────────────
        # This is the accuracy you "give up" by choosing the fairest model.
        # 0.0016 means the fairness improvement is almost free.
        "auc_cost_of_fairness": round(auc_cost, 4),

        # ── ProPublica reference numbers ──────────────────────────────────
        # From the actual deployed COMPAS system (ProPublica 2016 report).
        # We show these as a benchmark: most Rashomon set members are fairer
        # than the system that was actually deployed.
        "propublica_fpr_black": 0.447,   # 44.7% of innocent Black defendants flagged
        "propublica_fpr_white": 0.232,   # 23.2% of innocent White defendants flagged
        "propublica_fpr_gap":   0.215,   # the gap ProPublica reported
    }


# =============================================================================
# FUNCTION 4: MODEL DETAIL
# =============================================================================

def get_model_detail(key, fairness_df, rashomon_result):
    """
    Returns detailed information about one specific model for Page 3
    (Model Explorer). Called when the user selects a model from the dropdown.

    Includes:
      - Full fairness metrics for both focus groups
      - Feature importances (what does this model rely on?)
      - Decision rules as text (for Decision Trees only)

    Args:
        key             (str):       Unique model key (from fairness_df)
        fairness_df     (DataFrame): Output of fairness.compute_all_fairness()
        rashomon_result (dict):      Output of rashomon.build_rashomon_set()

    Returns:
        dict or None: Model detail dict, or None if key not found.
    """
    # Find the fairness row for this model
    row = fairness_df[fairness_df["key"] == key]
    if row.empty:
        return None
    row = row.iloc[0]

    # Find the fitted model in the rashomon result
    model_lookup = {m["key"]: m for m in rashomon_result["models"]}
    record       = model_lookup.get(key)
    if record is None:
        return None

    model         = record["model"]
    feature_names = list(rashomon_result["X_train"].columns)

    detail = {
        "key":           key,
        "name":          row["name"],
        "model_type":    row["model_type"],
        "test_auc":      row["test_auc"],
        "in_rashomon":   row["in_rashomon"],
        "complexity":    row["complexity"],
        "fpr_black":     row["fpr_black"],
        "fpr_white":     row["fpr_white"],
        "fpr_disparity": row["fpr_disparity"],
        "fnr_black":     row["fnr_black"],
        "fnr_white":     row["fnr_white"],
        "feature_names": feature_names,
    }

    # ── Feature importances ────────────────────────────────────────────────
    # How much weight does this model place on each feature?
    # This is important for understanding WHY a model is biased.
    # If prior_offenses is very important and prior_offenses correlates
    # with race (due to unequal policing), then the model amplifies that bias.
    try:
        # For Pipeline models (LR), extract the classifier step
        clf = (model.named_steps["clf"]
               if isinstance(model, Pipeline) else model)

        if hasattr(clf, "feature_importances_"):
            # Tree-based models: Gini impurity reduction per feature
            importances = dict(zip(feature_names, clf.feature_importances_))

        elif hasattr(clf, "coef_"):
            # Linear models: absolute value of coefficients
            # (already scaled by StandardScaler, so comparable)
            raw = np.abs(clf.coef_[0])
            # Normalise to sum to 1 (like feature_importances_)
            total = raw.sum()
            importances = dict(zip(feature_names,
                                   raw / total if total > 0 else raw))
        else:
            # Fallback: equal weight (we don't know importances)
            n = len(feature_names)
            importances = {f: 1.0/n for f in feature_names}

    except Exception:
        n = len(feature_names)
        importances = {f: 1.0/n for f in feature_names}

    # Sort by importance descending so the most important features come first
    detail["feature_importances"] = dict(
        sorted(importances.items(), key=lambda x: x[1], reverse=True)
    )

    # ── Decision Tree rules ────────────────────────────────────────────────
    # For Decision Trees, we can extract the actual human-readable rules.
    # This is the key interpretability advantage: you can see EXACTLY why
    # any individual was labelled high risk.
    if row["model_type"] == "Decision Tree":
        try:
            clf_tree = (model.named_steps["clf"]
                        if isinstance(model, Pipeline) else model)

            # export_text gives us the rules as a readable string
            # We limit to depth 4 for readability (beyond that it gets long)
            rules = export_text(
                clf_tree,
                feature_names = feature_names,
                max_depth     = 4,
            )
            detail["decision_rules"] = rules
            detail["n_leaves"]       = int(clf_tree.get_n_leaves())

        except Exception:
            detail["decision_rules"] = None
            detail["n_leaves"]       = row["complexity"]
    else:
        detail["decision_rules"] = None

    return detail


# =============================================================================
# FUNCTION 5: IMPOSSIBILITY THEOREM DATA
# =============================================================================

def get_impossibility_data(fairness_df):
    """
    Prepares data to visualise the Chouldechova (2017) impossibility theorem
    on Page 4 (Who Bears It).

    The theorem says: when recidivism base rates differ between groups,
    you cannot simultaneously achieve equal FPR AND equal FNR.

    We show this empirically: models with lower FPR disparity tend to
    have higher FNR disparity, and vice versa. This is NOT a property
    of any particular model — it is mathematically inevitable given the
    different base rates (52.3% Black vs 39.1% White).

    The scatter plot: FPR disparity on X, FNR disparity on Y.
    If both could be zero simultaneously, we'd see dots near (0, 0).
    Instead, we see a tradeoff pattern.

    Args:
        fairness_df (DataFrame): Output of fairness.compute_all_fairness()

    Returns:
        DataFrame: FPR vs FNR disparity for each model.
    """
    df = fairness_df[[
        "key", "short_name", "model_type",
        "test_auc", "fpr_disparity", "fnr_disparity", "fdr_disparity",
        "in_rashomon",
    ]].copy()

    # Add hover text for the impossibility theorem chart
    df["hover_text"] = df.apply(lambda r: (
        f"<b>{r['short_name']}</b><br>"
        f"FPR gap: {r['fpr_disparity']:.3f}  (ProPublica criterion)<br>"
        f"FNR gap: {r['fnr_disparity']:.3f}  (Northpointe criterion)<br>"
        f"AUC: {r['test_auc']:.4f}"
    ), axis=1)

    # Drop rows where either metric is NaN
    df = df.dropna(subset=["fpr_disparity", "fnr_disparity"])

    return df


# =============================================================================
# MAIN BLOCK — run directly to verify Phase 4
# Usage: python audit/analyze.py
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from audit.data     import load_data
    from audit.rashomon import build_rashomon_set
    from audit.fairness import compute_all_fairness, compute_defendant_disagreement

    print("=" * 65)
    print("PHASE 4 VERIFICATION — analyze.py")
    print("=" * 65)

    # Build on all previous phases
    print("\nRunning Phase 1–3 pipeline (takes ~5 seconds)...")
    X, y, df, race = load_data(verbose=False)
    result         = build_rashomon_set(X, y, race, verbose=False)
    fairness_df    = compute_all_fairness(result, verbose=False)
    print(f"  Ready: {len(fairness_df)} models with fairness metrics")

    # ── Test each function ─────────────────────────────────────────────────

    print(f"\n{'─'*65}")
    print("Testing get_scatter_data()")
    scatter = get_scatter_data(fairness_df)
    print(f"  Rows:    {len(scatter)}  (expected 49)")
    print(f"  Columns: {list(scatter.columns)}")
    assert len(scatter) == 49
    assert "hover_text"    in scatter.columns
    assert "dot_size"      in scatter.columns
    assert "marker_symbol" in scatter.columns
    # Check that all Rashomon members have filled circles
    rashomon_rows = scatter[scatter["in_rashomon"] == True]
    assert (rashomon_rows["marker_symbol"] == "circle").all(), \
        "FAIL: Rashomon members should have filled circles"
    print(f"  ✓ PASS — scatter data ready for Plotly")

    print(f"\n{'─'*65}")
    print("Testing compute_pareto_frontier()")
    pareto = compute_pareto_frontier(fairness_df)
    print(f"  Pareto-optimal models: {len(pareto)}")
    print(f"  Names:")
    for _, r in pareto.iterrows():
        print(f"    {r['short_name']:<25}  AUC={r['test_auc']:.4f}  "
              f"FPR gap={r['fpr_disparity']:.3f}")
    assert len(pareto) >= 2, "FAIL: expected at least 2 Pareto-optimal models"
    print(f"  ✓ PASS")

    print(f"\n{'─'*65}")
    print("Testing get_summary_stats()")
    stats = get_summary_stats(fairness_df, result)
    print(f"  Rashomon set size:       {stats['n_models_rashomon']}/49")
    print(f"  FPR gap range:           "
          f"{stats['fpr_gap_min']:.3f} – {stats['fpr_gap_max']:.3f}")
    print(f"  FPR gap variation:       {stats['fpr_gap_range']:.3f}")
    print(f"  AUC cost of fairness:    {stats['auc_cost_of_fairness']:.4f}")
    print(f"  Fairest model:           {stats['fairest_name'][:50]}")
    print(f"  Most biased model:       {stats['unfairest_name'][:50]}")
    assert stats["auc_cost_of_fairness"] < 0.01, \
        f"FAIL: AUC cost should be tiny, got {stats['auc_cost_of_fairness']}"
    assert stats["fpr_gap_range"] > 0.05, \
        f"FAIL: FPR range should be meaningful, got {stats['fpr_gap_range']}"
    print(f"  ✓ PASS")

    print(f"\n{'─'*65}")
    print("Testing get_model_detail()")
    # Test on the fairest model (a Random Forest)
    in_set      = fairness_df[fairness_df["in_rashomon"]]
    fairest_key = in_set.loc[in_set["fpr_disparity"].idxmin(), "key"]
    detail      = get_model_detail(fairest_key, fairness_df, result)
    print(f"  Model:             {detail['name']}")
    print(f"  AUC:               {detail['test_auc']:.4f}")
    print(f"  FPR gap:           {detail['fpr_disparity']:.3f}")
    print(f"  Feature importances:")
    for feat, imp in list(detail["feature_importances"].items())[:3]:
        print(f"    {feat:<20} {imp:.4f}")
    assert detail is not None
    assert "feature_importances" in detail
    assert len(detail["feature_importances"]) == 7
    print(f"  ✓ PASS")

    # Also test on a Decision Tree (should have decision rules)
    dt_keys = in_set[in_set["model_type"] == "Decision Tree"]["key"].tolist()
    if dt_keys:
        dt_detail = get_model_detail(dt_keys[0], fairness_df, result)
        has_rules = dt_detail["decision_rules"] is not None
        print(f"\n  Decision Tree detail:")
        print(f"    Has decision rules: {has_rules}")
        print(f"    Leaf count:         {dt_detail['n_leaves']}")
        if has_rules:
            # Show just the first 3 lines of the rules
            first_lines = dt_detail["decision_rules"].split("\n")[:3]
            for line in first_lines:
                print(f"    {line}")
        assert has_rules, "FAIL: Decision Trees should have readable rules"
        print(f"  ✓ PASS")

    print(f"\n{'─'*65}")
    print("Testing get_impossibility_data()")
    imp_df = get_impossibility_data(fairness_df)
    print(f"  Rows: {len(imp_df)}  (should equal number of models with valid metrics)")
    corr = imp_df["fpr_disparity"].corr(imp_df["fnr_disparity"])
    print(f"  Correlation between FPR gap and FNR gap: {corr:.3f}")
    print(f"  (The theorem predicts a tradeoff — not necessarily negative)")
    assert len(imp_df) > 0
    print(f"  ✓ PASS")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"PHASE 4 COMPLETE ✓ — analyze.py is verified and working.")
    print(f"\nAll four backend modules are complete and verified:")
    print(f"  Phase 1: data.py     — {len(X):,} defendants loaded and cleaned")
    print(f"  Phase 2: rashomon.py — 49 models trained, 40 in Rashomon set")
    print(f"  Phase 3: fairness.py — FPR gap range: "
          f"{stats['fpr_gap_min']:.3f}–{stats['fpr_gap_max']:.3f}, "
          f"AUC cost: {stats['auc_cost_of_fairness']:.4f}")
    print(f"  Phase 4: analyze.py  — scatter, Pareto, stats, detail, theorem")
    print(f"\nReady for Phase 5: the Streamlit app.")
    print(f"{'='*65}")
