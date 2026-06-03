# =============================================================================
# audit/fairness.py
# Racial Fairness Metrics for Every Model
# =============================================================================
#
# ── WHAT THIS MODULE COMPUTES ────────────────────────────────────────────────
#
# For every model the Rashomon set builder trained, we compute three fairness
# metrics — measured separately for Black and White defendants on the
# held-out test set. We then compute the DISPARITY: the absolute difference
# between the two groups on each metric.
#
# The central result of the entire project comes from this module:
# equally-accurate models (the Rashomon set) span a WIDE RANGE of disparities.
#
# ── THE THREE METRICS ────────────────────────────────────────────────────────
#
# 1. FALSE POSITIVE RATE (FPR) — ProPublica's metric
#    ─────────────────────────────────────────────────
#    FPR = FP / (FP + TN)
#
#    "Among defendants who did NOT reoffend (y_true = 0),
#     what fraction did the model label as high risk (y_pred = 1)?"
#
#    A high FPR for Black defendants = more innocent Black people are being
#    labelled dangerous than innocent White people.
#
#    FPR gap = |FPR_Black − FPR_White|
#    Higher gap = more racially biased.
#
# 2. FALSE NEGATIVE RATE (FNR) — Northpointe's metric
#    ──────────────────────────────────────────────────
#    FNR = FN / (FN + TP)
#
#    "Among defendants who DID reoffend (y_true = 1),
#     what fraction did the model label as low risk (y_pred = 0)?"
#
#    Northpointe argued their model was fair because FNR was similar
#    across races. Both sides were measuring real things. Neither was lying.
#
# 3. FALSE DISCOVERY RATE (FDR) — predictive parity metric
#    ────────────────────────────────────────────────────────
#    FDR = FP / (FP + TP)  =  1 − Precision
#
#    "Among everyone labelled high risk (y_pred = 1),
#     what fraction was actually low risk (y_true = 0)?"
#
#    If this is equal across groups, the 'high risk' label means the same
#    probability of recidivism regardless of race. This is sometimes called
#    'calibration' or 'predictive parity.'
#
# ── THE IMPOSSIBILITY THEOREM ────────────────────────────────────────────────
#
# Chouldechova (2017) proved mathematically:
#
#   If two groups have DIFFERENT base rates (different true recidivism rates),
#   then no classifier can simultaneously satisfy all three of:
#     - Equal FPR across groups
#     - Equal FNR across groups
#     - Equal FDR across groups
#
# Black defendants have a 52.3% recidivism rate; White defendants 39.1%.
# Because these rates differ, ProPublica's metric and Northpointe's metric
# cannot BOTH be satisfied at once. This is not anyone's fault — it is math.
# The choice of which metric to optimise is a political and ethical decision.
#
# ── CLASSIFICATION THRESHOLD ────────────────────────────────────────────────
#
# All our models output a PROBABILITY: P(recidivism) ∈ [0, 1].
# To compute FPR/FNR/FDR we need a binary prediction: high risk or low risk.
# We use a threshold of 0.5: if P(recidivism) ≥ 0.5, predict high risk.
#
# This is a methodological choice. COMPAS used a proprietary 10-point scale
# with a different effective threshold. Using 0.5 is standard practice and
# ensures all our models are compared on equal footing.
#
# ── COMPUTATION SCOPE ────────────────────────────────────────────────────────
#
# CRITICAL: All metrics are computed on the HELD-OUT TEST SET ONLY.
# (1,235 defendants the models never saw during training.)
# Computing fairness on training data would be meaningless — models can
# memorise training defendants, producing artificially good metrics.
#
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score


# =============================================================================
# CONSTANTS
# =============================================================================

# The probability threshold for classifying a defendant as "high risk."
# P(recidivism) >= this threshold → predicted high risk (y_pred = 1)
CLASSIFICATION_THRESHOLD = 0.5

# The two racial groups we focus on for the primary fairness comparison.
# These are the groups ProPublica's investigation focused on.
# We compute metrics for ALL groups, but the disparity headline numbers
# are for this pair.
FOCUS_GROUP_1 = "African-American"
FOCUS_GROUP_2 = "Caucasian"


# =============================================================================
# CORE METRIC FUNCTIONS
# These are small, focused functions. Each computes exactly one thing.
# We test them individually so we know each metric is correct.
# =============================================================================

def false_positive_rate(y_true, y_pred):
    """
    Computes the False Positive Rate for a group of defendants.

    FPR = FP / (FP + TN)
        = (predicted high-risk AND actually safe)
          / (all actually safe defendants)

    In plain English: "Among the innocent people, what fraction did we
    wrongly label as dangerous?"

    Args:
        y_true (array-like): True labels (0 = no recidivism, 1 = recidivism)
        y_pred (array-like): Predicted labels (0 = low risk, 1 = high risk)

    Returns:
        float: FPR in [0, 1].
               Returns NaN if there are no negative (innocent) examples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Only look at defendants who are truly innocent (y_true = 0)
    innocent_mask    = (y_true == 0)
    n_innocent       = innocent_mask.sum()

    if n_innocent == 0:
        return float("nan")  # cannot compute without any innocent defendants

    # False positives: innocent (y_true=0) but labelled dangerous (y_pred=1)
    false_positives  = ((y_pred == 1) & innocent_mask).sum()

    return float(false_positives / n_innocent)


def false_negative_rate(y_true, y_pred):
    """
    Computes the False Negative Rate for a group of defendants.

    FNR = FN / (FN + TP)
        = (predicted low-risk AND actually reoffended)
          / (all defendants who actually reoffended)

    In plain English: "Among the people who really did reoffend, what
    fraction did we wrongly label as safe?" This was Northpointe's
    fairness metric.

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels

    Returns:
        float: FNR in [0, 1], or NaN if no positive examples.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    recidivist_mask  = (y_true == 1)
    n_recidivists    = recidivist_mask.sum()

    if n_recidivists == 0:
        return float("nan")

    # False negatives: actually reoffended (y_true=1) but labelled safe (y_pred=0)
    false_negatives  = ((y_pred == 0) & recidivist_mask).sum()

    return float(false_negatives / n_recidivists)


def false_discovery_rate(y_true, y_pred):
    """
    Computes the False Discovery Rate for a group of defendants.

    FDR = FP / (FP + TP)  =  1 − Precision
        = (innocent but labelled high-risk)
          / (all defendants labelled high-risk)

    In plain English: "Among everyone we called dangerous, what fraction
    was actually innocent?" Equal FDR across groups is called 'predictive
    parity' — the high-risk label means the same probability for everyone.

    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels

    Returns:
        float: FDR in [0, 1], or NaN if nobody was predicted high-risk.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    predicted_highrisk = (y_pred == 1)
    n_predicted_highrisk = predicted_highrisk.sum()

    if n_predicted_highrisk == 0:
        return float("nan")

    # False positives: predicted high-risk (y_pred=1) but innocent (y_true=0)
    false_positives = ((y_true == 0) & predicted_highrisk).sum()

    return float(false_positives / n_predicted_highrisk)


# =============================================================================
# PER-GROUP METRICS
# =============================================================================

def compute_group_metrics(y_true, y_prob, race_arr, group,
                           threshold=CLASSIFICATION_THRESHOLD):
    """
    Computes all fairness metrics for ONE racial group.

    This function isolates the defendants belonging to the specified group
    and computes FPR, FNR, FDR, accuracy, and AUC for that subgroup.

    Args:
        y_true    (array): True labels for ALL test defendants
        y_prob    (array): Predicted probabilities for ALL test defendants
        race_arr  (array): Race labels for ALL test defendants
        group     (str):   The racial group to compute metrics for
                           e.g. "African-American"
        threshold (float): Binary classification threshold (default 0.5)

    Returns:
        dict or None: Metrics for this group, or None if group has fewer
                      than 10 defendants (not enough for reliable estimates).
    """
    y_true   = np.asarray(y_true)
    y_prob   = np.asarray(y_prob)
    race_arr = np.asarray(race_arr)

    # Isolate this racial group
    group_mask = (race_arr == group)
    n          = group_mask.sum()

    # Require at least 10 defendants for a meaningful estimate
    # (fewer would give very high variance)
    if n < 10:
        return None

    # Extract this group's true labels and probabilities
    y_t    = y_true[group_mask]
    y_p    = y_prob[group_mask]

    # Apply threshold to get binary predictions
    y_pred = (y_p >= threshold).astype(int)

    # Compute group AUC (how well does this model separate recidivists
    # from non-recidivists WITHIN this racial group?)
    try:
        group_auc = float(roc_auc_score(y_t, y_p))
    except ValueError:
        # roc_auc_score raises ValueError if only one class is present
        group_auc = float("nan")

    return {
        "group":       group,
        "n":           int(n),
        "base_rate":   float(y_t.mean()),         # true recidivism rate in this group
        "pred_rate":   float(y_pred.mean()),       # fraction model labels as high risk
        "accuracy":    float(accuracy_score(y_t, y_pred)),
        "auc":         group_auc,
        "fpr":         false_positive_rate(y_t, y_pred),
        "fnr":         false_negative_rate(y_t, y_pred),
        "fdr":         false_discovery_rate(y_t, y_pred),
    }


# =============================================================================
# FULL FAIRNESS REPORT FOR ONE MODEL
# =============================================================================

def compute_model_fairness(model_record, X_test, y_test, race_test,
                            threshold=CLASSIFICATION_THRESHOLD):
    """
    Computes the complete fairness profile for a single model.

    Called for every model in the Rashomon set (and outside it)
    to produce the data for the scatter plot and fairness breakdown pages.

    Steps:
      1. Get the model's predicted probabilities on the test set
      2. Compute group metrics for every racial group
      3. Compute disparity = |metric_Black − metric_White| for FPR, FNR, FDR

    Args:
        model_record (dict): A single model record from rashomon.py
        X_test       (DataFrame): Held-out test features
        y_test       (Series):    Held-out test labels
        race_test    (Series):    Held-out race labels
        threshold    (float):     Binary classification threshold

    Returns:
        dict: Complete fairness report. The most important fields are:
              fpr_disparity — our headline bias measure (lower = fairer)
              fpr_black     — FPR for Black defendants
              fpr_white     — FPR for White defendants
    """
    model  = model_record["model"]

    # Get predicted probabilities for each test defendant
    # predict_proba returns [[P(class=0), P(class=1)], ...]
    # We want P(recidivism) = the probability of class 1
    y_prob = model.predict_proba(X_test)[:, 1]

    y_true   = np.asarray(y_test)
    race_arr = np.asarray(race_test)

    # ── Compute metrics for every racial group ─────────────────────────────
    per_group = {}
    for group in sorted(race_test.unique()):
        metrics = compute_group_metrics(y_true, y_prob, race_arr,
                                        group, threshold)
        if metrics is not None:
            per_group[group] = metrics

    # ── Compute pairwise disparities for our two focus groups ──────────────
    # Disparity = absolute difference between the two focus groups.
    # A disparity of 0 = perfectly equal treatment.
    # A disparity of 0.20 = the two groups experience this metric 20% apart.

    m1 = per_group.get(FOCUS_GROUP_1)  # African-American metrics
    m2 = per_group.get(FOCUS_GROUP_2)  # Caucasian metrics

    def _abs_disparity(metric_name):
        """
        |metric_Black − metric_White|
        Returns NaN if either group's metric is missing or NaN.
        """
        if m1 is None or m2 is None:
            return float("nan")
        v1 = m1.get(metric_name, float("nan"))
        v2 = m2.get(metric_name, float("nan"))
        if np.isnan(v1) or np.isnan(v2):
            return float("nan")
        return round(abs(v1 - v2), 4)

    def _signed_disparity(metric_name):
        """
        metric_Black − metric_White  (signed, positive = Black bears more)
        Returns NaN if either is missing.
        """
        if m1 is None or m2 is None:
            return float("nan")
        v1 = m1.get(metric_name, float("nan"))
        v2 = m2.get(metric_name, float("nan"))
        if np.isnan(v1) or np.isnan(v2):
            return float("nan")
        return round(v1 - v2, 4)

    # Build the full fairness record for this model
    return {
        # Model identification
        "key":              model_record["key"],
        "name":             model_record["name"],
        "short_name":       model_record["short_name"],
        "model_type":       model_record["model_type"],
        "test_auc":         model_record["test_auc"],
        "complexity":       model_record["complexity"],
        "in_rashomon":      model_record["in_rashomon"],

        # Per-group metrics (nested dict, accessed by group name)
        "per_group":        per_group,

        # PRIMARY HEADLINE: FPR disparity (ProPublica's measure)
        # This is the Y axis of the central scatter plot.
        "fpr_disparity":    _abs_disparity("fpr"),
        "fpr_black":        m1["fpr"] if m1 else float("nan"),
        "fpr_white":        m2["fpr"] if m2 else float("nan"),
        "fpr_signed":       _signed_disparity("fpr"),  # positive = Black bears more

        # SECONDARY: FNR disparity (Northpointe's measure)
        "fnr_disparity":    _abs_disparity("fnr"),
        "fnr_black":        m1["fnr"] if m1 else float("nan"),
        "fnr_white":        m2["fnr"] if m2 else float("nan"),

        # TERTIARY: FDR disparity (predictive parity measure)
        "fdr_disparity":    _abs_disparity("fdr"),
        "fdr_black":        m1["fdr"] if m1 else float("nan"),
        "fdr_white":        m2["fdr"] if m2 else float("nan"),

        # Accuracy by group (for context)
        "acc_black":        m1["accuracy"] if m1 else float("nan"),
        "acc_white":        m2["accuracy"] if m2 else float("nan"),

        # Base rates (for showing the Chouldechova context)
        "base_rate_black":  m1["base_rate"] if m1 else float("nan"),
        "base_rate_white":  m2["base_rate"] if m2 else float("nan"),
    }


# =============================================================================
# SWEEP ALL MODELS
# =============================================================================

def compute_all_fairness(rashomon_result, verbose=True):
    """
    Computes fairness metrics for EVERY trained model (all 49).

    This is the central computation of the project. It produces a
    DataFrame where each row is one model and each column is a metric.
    The scatter plot, tables, and comparison panels in the app all
    come from this DataFrame.

    Args:
        rashomon_result (dict): Output of rashomon.build_rashomon_set()
        verbose         (bool): Print progress

    Returns:
        pd.DataFrame: One row per model, ~25 columns of metrics.
    """
    models    = rashomon_result["models"]
    X_test    = rashomon_result["X_test"]
    y_test    = rashomon_result["y_test"]
    race_test = rashomon_result["race_test"]

    if verbose:
        print(f"Computing fairness metrics for {len(models)} models "
              f"on {len(X_test):,}-defendant test set...")
        print(f"  (Held-out test set — models never saw these defendants)")
        print()

    rows = []

    for i, model_record in enumerate(models):
        fairness = compute_model_fairness(
            model_record, X_test, y_test, race_test
        )
        rows.append(fairness)

        if verbose:
            tag  = "✓" if model_record["in_rashomon"] else " "
            print(f"  {tag} [{i+1:2d}/49] {model_record['short_name']:<25}  "
                  f"AUC={model_record['test_auc']:.4f}  "
                  f"FPR gap={fairness['fpr_disparity']:.3f}  "
                  f"(Black={fairness['fpr_black']:.3f}, "
                  f"White={fairness['fpr_white']:.3f})")

    df = pd.DataFrame(rows)

    if verbose:
        in_set = df[df["in_rashomon"] == True]

        print(f"\n{'='*65}")
        print(f"FAIRNESS SWEEP COMPLETE")
        print(f"{'='*65}")
        print(f"\nWithin the Rashomon set ({len(in_set)} models):")
        print(f"  FPR disparity range:  "
              f"{in_set['fpr_disparity'].min():.3f} – "
              f"{in_set['fpr_disparity'].max():.3f}")
        print(f"  FPR gap varies by:    "
              f"{in_set['fpr_disparity'].max() - in_set['fpr_disparity'].min():.3f}")

        # Identify the two extreme models
        fairest_idx   = in_set["fpr_disparity"].idxmin()
        unfairest_idx = in_set["fpr_disparity"].idxmax()
        fairest       = in_set.loc[fairest_idx]
        unfairest     = in_set.loc[unfairest_idx]

        print(f"\n  FAIREST model in the Rashomon set:")
        print(f"    {fairest['name']}")
        print(f"    AUC = {fairest['test_auc']:.4f}  |  "
              f"FPR Black = {fairest['fpr_black']:.3f}  |  "
              f"FPR White = {fairest['fpr_white']:.3f}  |  "
              f"Gap = {fairest['fpr_disparity']:.3f}")

        print(f"\n  MOST BIASED model in the Rashomon set:")
        print(f"    {unfairest['name']}")
        print(f"    AUC = {unfairest['test_auc']:.4f}  |  "
              f"FPR Black = {unfairest['fpr_black']:.3f}  |  "
              f"FPR White = {unfairest['fpr_white']:.3f}  |  "
              f"Gap = {unfairest['fpr_disparity']:.3f}")

        auc_cost = abs(unfairest["test_auc"] - fairest["test_auc"])
        print(f"\n  AUC cost of choosing the fairest model: {auc_cost:.4f}")
        print(f"  (This is the accuracy you 'give up' to get a much fairer model.)")

    return df


# =============================================================================
# DEFENDANT DISAGREEMENT
# =============================================================================

def compute_defendant_disagreement(fairness_df, rashomon_result):
    """
    Finds defendants who receive DIFFERENT predictions from the fairest
    and most biased equally-accurate models.

    This is the most visceral demonstration of the project's argument.
    These are real people whose bail decision depends entirely on which
    equally-accurate model happened to be deployed.

    Args:
        fairness_df     (DataFrame): Output of compute_all_fairness()
        rashomon_result (dict):      Output of build_rashomon_set()

    Returns:
        dict: Statistics about the disagreements, or None if not computable.
    """
    in_set = fairness_df[fairness_df["in_rashomon"] == True]
    if len(in_set) < 2:
        return None

    # Identify the two extreme models within the Rashomon set
    fairest_key   = in_set.loc[in_set["fpr_disparity"].idxmin(), "key"]
    unfairest_key = in_set.loc[in_set["fpr_disparity"].idxmax(), "key"]

    # Get the fitted models
    model_lookup = {m["key"]: m["model"] for m in rashomon_result["models"]}
    fair_model   = model_lookup[fairest_key]
    unfair_model = model_lookup[unfairest_key]

    X_test    = rashomon_result["X_test"]
    y_test    = rashomon_result["y_test"]
    race_test = rashomon_result["race_test"]

    # Get binary predictions from each model (using 0.5 threshold)
    pred_fair   = (fair_model.predict_proba(X_test)[:, 1]
                   >= CLASSIFICATION_THRESHOLD).astype(int)
    pred_unfair = (unfair_model.predict_proba(X_test)[:, 1]
                   >= CLASSIFICATION_THRESHOLD).astype(int)

    race_arr = np.asarray(race_test)

    # Defendants where the two models DISAGREE
    disagree_mask = (pred_fair != pred_unfair)
    n_disagree    = int(disagree_mask.sum())
    n_total       = len(pred_fair)

    # How many disagreements are in each racial group?
    black_mask = (race_arr == "African-American")
    white_mask = (race_arr == "Caucasian")

    disagree_black = int((disagree_mask & black_mask).sum())
    disagree_white = int((disagree_mask & white_mask).sum())

    # Retrieve fairness row details for both models
    fair_row   = in_set[in_set["key"] == fairest_key].iloc[0]
    unfair_row = in_set[in_set["key"] == unfairest_key].iloc[0]

    return {
        "fairest_key":        fairest_key,
        "unfairest_key":      unfairest_key,
        "fairest_name":       fair_row["name"],
        "unfairest_name":     unfair_row["name"],
        "fpr_gap_fairest":    float(fair_row["fpr_disparity"]),
        "fpr_gap_unfairest":  float(unfair_row["fpr_disparity"]),
        "auc_fairest":        float(fair_row["test_auc"]),
        "auc_unfairest":      float(unfair_row["test_auc"]),
        # Disagreement counts
        "n_total":            n_total,
        "n_disagreements":    n_disagree,
        "disagree_pct":       round(n_disagree / n_total * 100, 1),
        "disagree_black":     disagree_black,
        "disagree_white":     disagree_white,
        "disagree_black_pct": round(disagree_black / black_mask.sum() * 100, 1),
        "disagree_white_pct": round(disagree_white / white_mask.sum() * 100, 1),
    }


# =============================================================================
# MAIN BLOCK — run directly to verify Phase 3
# Usage: python audit/fairness.py
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from audit.data     import load_data
    from audit.rashomon import build_rashomon_set

    print("=" * 65)
    print("PHASE 3 VERIFICATION — fairness.py")
    print("=" * 65)

    # Load data and build Rashomon set (reusing Phase 1 and 2)
    print("\nLoading data and building Rashomon set (this takes ~5 seconds)...")
    X, y, df, race = load_data(verbose=False)
    result         = build_rashomon_set(X, y, race, verbose=False)
    print(f"  Ready: {len(result['models'])} models, "
          f"{len(result['X_test'])} test defendants")

    # ── Compute fairness for all models ────────────────────────────────────
    print()
    fairness_df = compute_all_fairness(result, verbose=True)

    # ── Verification checks ────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("VERIFICATION CHECKS")
    print(f"{'─'*65}")

    in_set = fairness_df[fairness_df["in_rashomon"] == True]

    # Check 1: Every model has a valid FPR disparity
    print(f"\nCheck 1 — All models have valid FPR disparity")
    n_nan = fairness_df["fpr_disparity"].isna().sum()
    print(f"  Models with NaN FPR disparity: {n_nan}")
    assert n_nan == 0, f"FAIL: {n_nan} models have NaN FPR disparity"
    print(f"  ✓ PASS")

    # Check 2: FPR values are in [0, 1]
    print(f"\nCheck 2 — All FPR values are in [0, 1]")
    bad_fpr = ((fairness_df["fpr_black"] < 0) |
               (fairness_df["fpr_black"] > 1) |
               (fairness_df["fpr_white"] < 0) |
               (fairness_df["fpr_white"] > 1)).sum()
    print(f"  Out-of-range FPR values: {bad_fpr}")
    assert bad_fpr == 0, f"FAIL: {bad_fpr} out-of-range FPR values"
    print(f"  ✓ PASS")

    # Check 3: FPR disparity is always positive (it is an absolute value)
    print(f"\nCheck 3 — FPR disparity is always non-negative")
    negative = (fairness_df["fpr_disparity"] < 0).sum()
    print(f"  Negative disparities: {negative}")
    assert negative == 0, f"FAIL: {negative} negative disparities"
    print(f"  ✓ PASS")

    # Check 4: The Rashomon set shows meaningful variation in fairness
    # If there is no variation, the project has nothing interesting to say.
    print(f"\nCheck 4 — Rashomon set shows meaningful FPR disparity variation")
    fpr_range = in_set["fpr_disparity"].max() - in_set["fpr_disparity"].min()
    print(f"  FPR disparity range within Rashomon set: {fpr_range:.3f}")
    print(f"  (A range of 0.02 or more is meaningful)")
    assert fpr_range >= 0.02, f"FAIL: variation too small ({fpr_range:.3f})"
    print(f"  ✓ PASS — variation is {fpr_range:.3f}")

    # Check 5: Verify the metric functions on a known example
    # Create a toy example where we know the exact answer:
    # 4 innocent people: 2 labelled high-risk, 2 labelled low-risk → FPR = 0.5
    print(f"\nCheck 5 — Manual verification of FPR formula")
    toy_true = np.array([0, 0, 0, 0, 1, 1])  # 4 innocent, 2 guilty
    toy_pred = np.array([1, 1, 0, 0, 1, 0])  # 2 innocent flagged, 2 not
    expected_fpr = 2/4  # = 0.5
    computed_fpr = false_positive_rate(toy_true, toy_pred)
    print(f"  Toy example: 4 innocent, 2 labelled high-risk")
    print(f"  Expected FPR: {expected_fpr:.3f}")
    print(f"  Computed FPR: {computed_fpr:.3f}")
    assert abs(computed_fpr - expected_fpr) < 1e-9, \
        f"FAIL: expected {expected_fpr}, got {computed_fpr}"
    print(f"  ✓ PASS")

    # Check 6: FNR formula verification
    print(f"\nCheck 6 — Manual verification of FNR formula")
    expected_fnr = 1/2  # 1 guilty person labelled safe, out of 2 guilty total
    computed_fnr = false_negative_rate(toy_true, toy_pred)
    print(f"  Toy example: 2 guilty, 1 labelled low-risk")
    print(f"  Expected FNR: {expected_fnr:.3f}")
    print(f"  Computed FNR: {computed_fnr:.3f}")
    assert abs(computed_fnr - expected_fnr) < 1e-9, \
        f"FAIL: expected {expected_fnr}, got {computed_fnr}"
    print(f"  ✓ PASS")

    # ── Defendant disagreement ─────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("DEFENDANT DISAGREEMENT ANALYSIS")
    print(f"{'─'*65}")

    disagree = compute_defendant_disagreement(fairness_df, result)
    if disagree:
        print(f"\n  Comparing:")
        print(f"    Fairest model:   {disagree['fairest_name'][:55]}")
        print(f"    Most biased:     {disagree['unfairest_name'][:55]}")
        print(f"\n  These two models have identical accuracy to within ε=0.02.")
        print(f"  Yet {disagree['n_disagreements']} defendants "
              f"({disagree['disagree_pct']}% of the test set)")
        print(f"  receive a DIFFERENT prediction depending on which was deployed.")
        print(f"\n  Breakdown by race:")
        print(f"    Black defendants affected: {disagree['disagree_black']} "
              f"({disagree['disagree_black_pct']}% of Black test defendants)")
        print(f"    White defendants affected: {disagree['disagree_white']} "
              f"({disagree['disagree_white_pct']}% of White test defendants)")

    print(f"\n{'='*65}")
    print(f"PHASE 3 COMPLETE ✓ — fairness.py is verified and working.")
    print(f"{'='*65}")
