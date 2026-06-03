# =============================================================================
# audit/rashomon.py
# Rashomon Set Construction
# =============================================================================
#
# ── WHAT IS THE RASHOMON SET? ─────────────────────────────────────────────────
#
# Named after Akira Kurosawa's 1950 film "Rashomon," where four witnesses
# describe the same murder with conflicting but equally plausible accounts,
# the Rashomon set in machine learning is the collection of all models whose
# accuracy is within ε (epsilon) of the best model's accuracy.
#
# Formally:   R(ε) = { f  :  AUC(f) ≥ AUC* − ε }
#
# where AUC* is the best AUC achieved by any model we trained, and ε is
# our tolerance threshold (we use ε = 0.02, meaning "within 2%").
#
# Every model in R(ε) is a defensible deployment choice — you could not
# justify rejecting it on accuracy grounds. An institution choosing between
# two Rashomon set members and picking the more biased one is making a
# fairness choice, not a technical one.
#
# ── WHAT THIS MODULE DOES ────────────────────────────────────────────────────
#
# 1. Splits the data into 80% train / 20% test (stratified by race × label)
# 2. Defines a family of 49 model configurations across four algorithm types
# 3. Trains each model on the training set
# 4. Evaluates each model's AUC on the held-out test set
# 5. Marks every model within 2% of the best AUC as a Rashomon set member
# 6. Returns everything the fairness module needs to do its analysis
#
# ── THE MODEL FAMILY ─────────────────────────────────────────────────────────
#
# We train across four algorithm families to get a wide spread of model types:
#
#   Decision Trees (21 configs)
#     Depths 2, 3, 4, 5, 6, 7, 8  ×  min_leaf 5, 10, 20
#     These range from very simple (depth-2 = 4 decision rules) to complex.
#     Depth-2 trees are interpretable — a human can read every rule.
#     They are also the LEAST accurate in this family.
#
#   Logistic Regression (10 configs)
#     C values: 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0
#     C controls regularisation strength. Low C = heavily constrained = simple.
#     High C = lightly constrained = can fit training data closely.
#     Uses StandardScaler (required: LR is sensitive to feature scale).
#
#   Random Forest (12 configs)
#     max_depth: 3, 5, 7, None  ×  min_samples_leaf: 5, 10, 20
#     Ensemble of 50 trees. More accurate but not individually interpretable.
#
#   Gradient Boosting (6 configs)
#     max_depth: 2, 3, 5  ×  learning_rate: 0.05, 0.1
#     Sequential ensemble — typically the highest accuracy of the four families.
#
# ── TRAIN/TEST SPLIT DISCIPLINE ──────────────────────────────────────────────
#
# CRITICAL: All fairness metrics in Phase 3 are computed on the TEST SET ONLY.
# We must never evaluate fairness on the training set — models can memorise
# training data, which would distort the fairness measurements.
#
# We stratify by race × label: the test set contains proportional fractions
# of every (race, recidivism_outcome) combination. This ensures reliable
# fairness estimates even for smaller racial groups.
#
# =============================================================================

import warnings
import time
import numpy as np
import pandas as pd

from sklearn.tree             import DecisionTreeClassifier
from sklearn.linear_model     import LogisticRegression
from sklearn.ensemble         import (RandomForestClassifier,
                                       GradientBoostingClassifier)
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import roc_auc_score
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline

# Suppress harmless sklearn convergence warnings during the hyperparameter
# sweep. Some LR configs converge slowly at extreme C values — the fitted
# model is still valid and useful for our analysis.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*max_iter.*")
warnings.filterwarnings("ignore", message=".*solver.*")


# =============================================================================
# CONSTANTS
# =============================================================================

# ε = 0.02 means: a model is in the Rashomon set if its test AUC is within
# 2 percentage points of the best model's test AUC.
# This is a standard threshold in the literature (see Semenova et al. 2022).
EPSILON = 0.02

# Reproducibility: fix the random seed so every run gives identical results.
# Change this if you want to see how results vary across different splits.
RANDOM_STATE = 42

# Train/test split ratio
TEST_SIZE = 0.20  # 20% held out for evaluation


# =============================================================================
# MODEL FAMILY DEFINITIONS
# =============================================================================

def _make_model_configs():
    """
    Returns the full list of 49 model configurations to train.

    This is a function (not a module-level constant) because sklearn
    estimators are stateful objects — if we stored them as constants and
    trained them, the constant would hold the fitted model, which would
    cause bugs if we called the function twice. A fresh call gives fresh
    unfitted estimators every time.

    Each config is a dict with:
        name        — human-readable name shown in the app
        short_name  — compact name for chart labels
        key         — unique identifier for lookups
        model_type  — category label ('Decision Tree', etc.)
        model       — unfitted sklearn estimator (or Pipeline)
        complexity  — number of leaves for trees, 999 for black-box models.
                      Used to illustrate that simple models exist in the set.
    """
    configs = []

    # ── DECISION TREES ────────────────────────────────────────────────────
    # We vary two hyperparameters:
    #   max_depth:        controls how many splits the tree can make
    #   min_samples_leaf: minimum number of training examples in any leaf
    #                     (prevents the tree from making rules that apply
    #                     to only 1–2 people — overfitting)
    #
    # A depth-2 tree has at most 4 leaves — 4 possible decision outcomes.
    # That is extremely simple. A depth-8 tree has up to 256 leaves.
    for depth in [2, 3, 4, 5, 6, 7, 8]:
        for min_leaf in [5, 10, 20]:
            configs.append({
                "name":       f"Decision Tree (depth={depth}, min_leaf={min_leaf})",
                "short_name": f"DT d={depth}",
                "key":        f"dt_d{depth}_ml{min_leaf}",
                "model_type": "Decision Tree",
                "model":      DecisionTreeClassifier(
                                  max_depth        = depth,
                                  min_samples_leaf = min_leaf,
                                  random_state     = RANDOM_STATE,
                              ),
                # Complexity will be updated after fitting to the actual
                # number of leaves the trained tree has.
                "complexity": -1,
            })

    # ── LOGISTIC REGRESSION ───────────────────────────────────────────────
    # C is the inverse of regularisation strength.
    #   Small C (e.g. 0.001) = strong regularisation = model is forced to be
    #                           simple, large coefficients are penalised.
    #   Large C (e.g. 100)   = weak regularisation = model can fit training
    #                           data more closely.
    #
    # StandardScaler is required inside a Pipeline because Logistic Regression
    # is sensitive to feature scale. Without it, priors_count (range 0–38)
    # and age (range 18–96) dominate capital_loss in the coefficient math.
    # StandardScaler rescales every feature to mean=0, std=1 so they are
    # all on equal footing.
    for C in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
        configs.append({
            "name":       f"Logistic Regression (C={C})",
            "short_name": f"LR C={C}",
            "key":        f"lr_c{C}",
            "model_type": "Logistic Regression",
            "model":      Pipeline([
                              ("scaler", StandardScaler()),
                              ("clf",    LogisticRegression(
                                             C            = C,
                                             max_iter     = 2000,
                                             random_state = RANDOM_STATE,
                                         )),
                          ]),
            # LR does not have "leaves" — we assign 999 to mark it as a
            # complex/non-enumerable model for the complexity chart.
            "complexity": 999,
        })

    # ── RANDOM FOREST ─────────────────────────────────────────────────────
    # An ensemble of 50 Decision Trees, each trained on a random subset of
    # the data and a random subset of features. The final prediction is the
    # average probability across all 50 trees.
    #
    # Why is RF more accurate than a single tree?
    # Because the errors of individual trees tend to cancel out when averaged.
    # A single tree might overfit to a peculiarity in the data. 50 trees
    # trained on different subsets will average out those peculiarities.
    #
    # Why n_estimators=50 and not 100 or 500?
    # 50 gives good accuracy and trains fast (~2–3 seconds for our dataset).
    # Accuracy improvements beyond 50 trees are minimal for this data size.
    for max_depth in [3, 5, 7, None]:      # None = no depth limit
        for min_leaf in [5, 10, 20]:
            depth_str = str(max_depth) if max_depth else "unlimited"
            configs.append({
                "name":       f"Random Forest (depth={depth_str}, min_leaf={min_leaf})",
                "short_name": f"RF d={depth_str}",
                "key":        f"rf_d{depth_str}_ml{min_leaf}",
                "model_type": "Random Forest",
                "model":      RandomForestClassifier(
                                  n_estimators     = 50,
                                  max_depth        = max_depth,
                                  min_samples_leaf = min_leaf,
                                  random_state     = RANDOM_STATE,
                                  n_jobs           = -1,  # use all CPU cores
                              ),
                "complexity": 999,
            })

    # ── GRADIENT BOOSTING ─────────────────────────────────────────────────
    # A sequential ensemble: each tree tries to correct the mistakes of the
    # previous trees. This tends to be the most accurate family but also the
    # slowest to train and least interpretable.
    #
    # learning_rate controls how much each tree contributes to the ensemble.
    # Low lr = each tree contributes a little = slower learning, more trees
    #           needed to reach the same accuracy, but less overfitting.
    # High lr = each tree contributes more = faster learning, more overfitting.
    for max_depth in [2, 3, 5]:
        for lr in [0.05, 0.1]:
            configs.append({
                "name":       f"Gradient Boosting (depth={max_depth}, lr={lr})",
                "short_name": f"GBM d={max_depth}",
                "key":        f"gbm_d{max_depth}_lr{lr}",
                "model_type": "Gradient Boosting",
                "model":      GradientBoostingClassifier(
                                  n_estimators = 100,
                                  max_depth    = max_depth,
                                  learning_rate= lr,
                                  random_state = RANDOM_STATE,
                              ),
                "complexity": 999,
            })

    return configs


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

def make_train_test_split(X, y, race):
    """
    Creates a stratified 80/20 train/test split.

    WHY STRATIFY BY RACE × LABEL?
    ─────────────────────────────
    A simple random split might accidentally put almost all Black
    non-recidivists into the training set, leaving only a handful in the
    test set. With a small test group, our FPR estimate for that group
    would be unreliable (high variance).

    By stratifying on the combined (race, label) stratum, we ensure the
    test set has proportional representation of every combination, giving
    us reliable fairness estimates.

    For example: if 15% of the training set is "Black, did not reoffend",
    then approximately 15% of the test set will also be that group.

    Args:
        X    (DataFrame): Feature matrix
        y    (Series):    Binary labels
        race (Series):    Race labels (for stratification)

    Returns:
        X_train, X_test, y_train, y_test, race_train, race_test
    """
    # Create the combined stratum: "African-American_0", "Caucasian_1", etc.
    strata = race.astype(str) + "_" + y.astype(str)

    (X_train, X_test,
     y_train, y_test,
     race_train, race_test) = train_test_split(
        X, y, race,
        test_size    = TEST_SIZE,
        stratify     = strata,
        random_state = RANDOM_STATE,
    )

    return X_train, X_test, y_train, y_test, race_train, race_test


# =============================================================================
# MAIN FUNCTION — BUILD THE RASHOMON SET
# =============================================================================

def build_rashomon_set(X, y, race, epsilon=EPSILON, verbose=True):
    """
    Trains all 49 model configurations and identifies the Rashomon set.

    For each model:
      1. Trains on the training split (never sees the test set)
      2. Computes AUC on the held-out test set
      3. Records complexity (leaf count for trees)
      4. Checks if AUC ≥ best_AUC − ε  → marks as Rashomon set member

    Args:
        X       (DataFrame): Full feature matrix (from data.py)
        y       (Series):    Full binary labels
        race    (Series):    Full race labels
        epsilon (float):     Rashomon threshold (default 0.02 = 2%)
        verbose (bool):      Print progress

    Returns:
        dict containing:
            "models"      — list of model record dicts (see below)
            "best_auc"    — best test AUC achieved by any model
            "epsilon"     — the threshold used
            "X_train"     — training features
            "X_test"      — test features
            "y_train"     — training labels
            "y_test"      — test labels
            "race_train"  — training race labels
            "race_test"   — test race labels (used in Phase 3)

    Each model record dict contains:
        "name"        — full human-readable name
        "short_name"  — compact chart label
        "key"         — unique identifier
        "model_type"  — algorithm family
        "model"       — the FITTED sklearn estimator
        "test_auc"    — AUC on held-out test set (the Rashomon criterion)
        "train_auc"   — AUC on training set (to check for overfitting)
        "complexity"  — leaf count for trees, 999 for black-box models
        "in_rashomon" — True if test_auc >= best_auc - epsilon
        "params"      — hyperparameter summary string for display
    """

    # ── Step 1: Create the train/test split ───────────────────────────────
    if verbose:
        print("Creating train/test split (80/20, stratified by race × label)...")

    (X_train, X_test,
     y_train, y_test,
     race_train, race_test) = make_train_test_split(X, y, race)

    if verbose:
        print(f"  Train: {len(X_train):,} defendants")
        print(f"  Test:  {len(X_test):,} defendants")
        # Show racial composition of the test set
        for r in ["African-American", "Caucasian", "Hispanic"]:
            n = (race_test == r).sum()
            print(f"    Test set — {r}: {n}")

    # ── Step 2: Get fresh unfitted model configurations ────────────────────
    configs = _make_model_configs()

    if verbose:
        print(f"\nTraining {len(configs)} model configurations...")
        t_start = time.time()

    trained_models = []
    best_auc       = 0.0   # tracks the best AUC seen so far

    # ── Step 3: Train and evaluate each model ─────────────────────────────
    for i, config in enumerate(configs):

        if verbose and (i % 10 == 0 or i == len(configs) - 1):
            print(f"  [{i+1:2d}/{len(configs)}] Training {config['short_name']}...")

        model = config["model"]

        try:
            # Train on training set ONLY — the model never sees test data
            model.fit(X_train, y_train)

            # Evaluate on test set: predict_proba gives P(recidivism)
            # We use the probability (not the hard 0/1 prediction) for AUC
            # because AUC measures discrimination across all thresholds.
            test_probs  = model.predict_proba(X_test)[:, 1]
            test_auc    = float(roc_auc_score(y_test, test_probs))

            # Also record train AUC to check for overfitting.
            # If train_auc >> test_auc the model memorised training data.
            train_probs = model.predict_proba(X_train)[:, 1]
            train_auc   = float(roc_auc_score(y_train, train_probs))

            # For Decision Trees, record the actual number of leaves in
            # the trained tree (may be less than max_depth allows if the
            # tree stopped splitting early due to min_samples_leaf).
            complexity = config["complexity"]
            if config["model_type"] == "Decision Tree":
                complexity = int(model.get_n_leaves())

            # Update best AUC seen so far across all models
            if test_auc > best_auc:
                best_auc = test_auc

            trained_models.append({
                "name":        config["name"],
                "short_name":  config["short_name"],
                "key":         config["key"],
                "model_type":  config["model_type"],
                "model":       model,           # fitted estimator
                "test_auc":    round(test_auc,  4),
                "train_auc":   round(train_auc, 4),
                "complexity":  complexity,
                "in_rashomon": False,           # will be set below
                "params":      _params_summary(config["name"]),
            })

        except Exception as err:
            if verbose:
                print(f"    WARNING: {config['name']} failed — {err}")
            continue

    # ── Step 4: Mark Rashomon set members ─────────────────────────────────
    # Now that we know best_auc, mark every model within epsilon of it.
    rashomon_threshold = best_auc - epsilon
    n_in_set           = 0

    for record in trained_models:
        if record["test_auc"] >= rashomon_threshold:
            record["in_rashomon"] = True
            n_in_set += 1

    # ── Step 5: Report ────────────────────────────────────────────────────
    if verbose:
        elapsed = time.time() - t_start
        print(f"\nTraining complete in {elapsed:.1f} seconds.")
        print(f"\n{'='*55}")
        print(f"RASHOMON SET RESULTS  (ε = {epsilon})")
        print(f"{'='*55}")
        print(f"  Best test AUC:      {best_auc:.4f}")
        print(f"  Threshold:          {rashomon_threshold:.4f}  (= {best_auc:.4f} - {epsilon})")
        print(f"  Models in set:      {n_in_set} / {len(trained_models)}")

        # Break down by model type
        print(f"\n  Rashomon set composition:")
        from collections import Counter
        type_counts = Counter(
            m["model_type"] for m in trained_models if m["in_rashomon"]
        )
        for mtype, count in sorted(type_counts.items()):
            print(f"    {mtype:<25} {count} models")

        # Show the top 5 most accurate models
        print(f"\n  Top 5 models by test AUC:")
        top5 = sorted(trained_models, key=lambda x: x["test_auc"], reverse=True)[:5]
        for r in top5:
            tag = "✓" if r["in_rashomon"] else " "
            cmplx = (f"{r['complexity']} leaves"
                     if r["complexity"] < 999 else "complex")
            print(f"  {tag} {r['short_name']:<25}  "
                  f"AUC={r['test_auc']:.4f}  {cmplx}")

    return {
        "models":      trained_models,
        "best_auc":    round(best_auc, 4),
        "epsilon":     epsilon,
        "X_train":     X_train,
        "X_test":      X_test,
        "y_train":     y_train,
        "y_test":      y_test,
        "race_train":  race_train,
        "race_test":   race_test,
    }


def _params_summary(full_name):
    """
    Extracts the parenthetical parameter summary from a model name.
    e.g. "Decision Tree (depth=5, min_leaf=10)" → "depth=5, min_leaf=10"
    Used for compact display in the app.
    """
    start = full_name.find("(")
    end   = full_name.rfind(")")
    if start != -1 and end != -1:
        return full_name[start+1 : end]
    return ""


# =============================================================================
# CONVENIENCE ACCESSORS
# These are small helper functions used by Phase 3 and the app.
# =============================================================================

def get_rashomon_models(result):
    """Returns only the models that are in the Rashomon set."""
    return [m for m in result["models"] if m["in_rashomon"]]


def get_all_models(result):
    """Returns all 49 trained models (in set and outside set)."""
    return result["models"]


def summarise_rashomon_set(result):
    """
    Returns a summary dict of the Rashomon set composition.
    Used on the Home page of the app for the quick-stats section.
    """
    rashomon  = get_rashomon_models(result)
    all_m     = result["models"]

    # Count models per type within the set
    from collections import Counter
    type_counts = dict(Counter(m["model_type"] for m in rashomon))

    # Find the complexity range of Decision Trees in the set
    dt_in_set  = [m for m in rashomon if m["model_type"] == "Decision Tree"]
    min_leaves = min((m["complexity"] for m in dt_in_set), default=0)
    max_leaves = max((m["complexity"] for m in dt_in_set), default=0)

    return {
        "n_total":       len(all_m),
        "n_in_set":      len(rashomon),
        "best_auc":      result["best_auc"],
        "epsilon":       result["epsilon"],
        "threshold_auc": round(result["best_auc"] - result["epsilon"], 4),
        "type_counts":   type_counts,
        "n_dt_in_set":   len(dt_in_set),
        "min_dt_leaves": min_leaves,
        "max_dt_leaves": max_leaves,
        "auc_range": (
            round(min(m["test_auc"] for m in rashomon), 4),
            round(max(m["test_auc"] for m in rashomon), 4),
        ),
    }


# =============================================================================
# MAIN BLOCK — run this file directly to test Phase 2
# Usage: python audit/rashomon.py
# =============================================================================

if __name__ == "__main__":
    import sys
    import os

    # Allow running from the project root directory
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from audit.data import load_data

    print("=" * 60)
    print("PHASE 2 VERIFICATION — rashomon.py")
    print("=" * 60)

    # Load the data we verified in Phase 1
    print("\nLoading data from Phase 1...")
    X, y, df, race = load_data(verbose=False)
    print(f"  Loaded {len(X):,} defendants, {X.shape[1]} features")

    # Build the Rashomon set
    print()
    result = build_rashomon_set(X, y, race, epsilon=EPSILON, verbose=True)

    # ── Verification checks ────────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"VERIFICATION CHECKS")
    print(f"{'─'*55}")

    models   = result["models"]
    rashomon = get_rashomon_models(result)
    summary  = summarise_rashomon_set(result)

    # Check 1: We trained the expected number of models
    print(f"\nCheck 1 — Total models trained")
    print(f"  Expected: 49")
    print(f"  Got:      {len(models)}")
    assert len(models) == 49, f"FAIL: expected 49, got {len(models)}"
    print(f"  ✓ PASS")

    # Check 2: The Rashomon set is non-trivial (not empty, not everything)
    print(f"\nCheck 2 — Rashomon set is non-trivial")
    print(f"  Got {len(rashomon)} models in the set (out of 49)")
    assert 10 < len(rashomon) < 49, \
        f"FAIL: unexpected set size {len(rashomon)}"
    print(f"  ✓ PASS")

    # Check 3: Best AUC is in a reasonable range for COMPAS
    # Published COMPAS AUC estimates range from 0.65 to 0.75
    print(f"\nCheck 3 — Best AUC is in a reasonable range")
    print(f"  Expected: between 0.65 and 0.80")
    print(f"  Got:      {result['best_auc']:.4f}")
    assert 0.65 <= result["best_auc"] <= 0.80, \
        f"FAIL: unexpected AUC {result['best_auc']}"
    print(f"  ✓ PASS")

    # Check 4: All Rashomon set members satisfy the epsilon criterion
    print(f"\nCheck 4 — All Rashomon members satisfy AUC ≥ best - ε")
    threshold = result["best_auc"] - EPSILON
    violations = [m for m in rashomon if m["test_auc"] < threshold]
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Violations found: {len(violations)}")
    assert len(violations) == 0, f"FAIL: {len(violations)} models in set below threshold"
    print(f"  ✓ PASS")

    # Check 5: Train/test split sizes are correct
    print(f"\nCheck 5 — Train/test split sizes")
    total = len(result["X_train"]) + len(result["X_test"])
    test_frac = len(result["X_test"]) / total
    print(f"  Train: {len(result['X_train']):,}  Test: {len(result['X_test']):,}")
    print(f"  Test fraction: {test_frac:.2f}  (expected ~0.20)")
    assert abs(test_frac - 0.20) < 0.01, f"FAIL: test fraction {test_frac}"
    print(f"  ✓ PASS")

    # Check 6: No data leakage — test indices not in train indices
    print(f"\nCheck 6 — No data leakage (train and test are disjoint)")
    train_idx = set(result["X_train"].index)
    test_idx  = set(result["X_test"].index)
    overlap   = train_idx & test_idx
    print(f"  Overlapping indices: {len(overlap)}")
    assert len(overlap) == 0, f"FAIL: {len(overlap)} rows appear in both sets!"
    print(f"  ✓ PASS")

    # ── Print a helpful summary for understanding ──────────────────────────
    print(f"\n{'─'*55}")
    print(f"RASHOMON SET SUMMARY")
    print(f"{'─'*55}")
    print(f"\n  {summary['n_in_set']} out of {summary['n_total']} models "
          f"are in the Rashomon set (ε={summary['epsilon']})")
    print(f"\n  Composition:")
    for mtype, count in sorted(summary["type_counts"].items()):
        print(f"    {mtype:<25}  {count} models")
    print(f"\n  Decision Trees in set: {summary['n_dt_in_set']}")
    if summary['n_dt_in_set'] > 0:
        print(f"  Smallest tree in set:  {summary['min_dt_leaves']} leaves")
        print(f"  Largest tree in set:   {summary['max_dt_leaves']} leaves")
    print(f"\n  AUC range within set:  "
          f"{summary['auc_range'][0]:.4f} – {summary['auc_range'][1]:.4f}")
    print(f"\n  KEY INSIGHT: These {summary['n_in_set']} models are all")
    print(f"  equally defensible as 'the best model' by accuracy.")
    print(f"  In Phase 3 we will measure how much their racial bias differs.")

    print(f"\n{'='*60}")
    print(f"PHASE 2 COMPLETE ✓ — rashomon.py is verified and working.")
    print(f"{'='*60}")
