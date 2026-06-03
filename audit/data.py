# =============================================================================
# audit/data.py
# COMPAS Data Loading and Preprocessing
# =============================================================================
#
# ── BACKGROUND ───────────────────────────────────────────────────────────────
#
# In 2016, ProPublica filed a public records request with Broward County,
# Florida and obtained the COMPAS recidivism scores for 7,214 defendants
# screened between 2013 and 2014. They then matched those records against
# the county courthouse database to determine who actually reoffended within
# two years. The result is the dataset we use here.
#
# The file we download: compas-scores-two-years.csv
#   — Published by ProPublica on GitHub (fully public, no login required)
#   — Contains COMPAS scores + actual 2-year recidivism outcomes
#   — The column 'two_year_recid' is our prediction target (0 or 1)
#
# ── WHY WE FOLLOW PROBUBLICA'S EXACT FILTER ──────────────────────────────────
#
# ProPublica applied four specific filtering steps before their analysis.
# We apply the same four steps so our results are directly comparable to
# their published findings. If we used different filters, our numbers would
# not match theirs, and we could not say "we found X compared to ProPublica's Y."
#
# ── WHY RACE IS NEVER A MODEL INPUT ──────────────────────────────────────────
#
# Race is the protected attribute we measure fairness with respect to.
# We keep it separate — it goes into the fairness analysis but never
# into the model's feature matrix. This makes the finding more powerful:
# models that have NEVER SEEN race still produce racially disparate outcomes.
# Why? Because prior offenses, age, and charge type all correlate with race
# due to decades of unequal policing. The bias enters through proxies.
#
# ── FEATURES WE USE ──────────────────────────────────────────────────────────
#
# From the ProPublica CSV, we use only the columns that reflect a defendant's
# criminal history and basic demographics — not race:
#
#   age             — age at time of COMPAS screening (continuous, 18–96)
#   priors_count    — number of prior criminal offenses (continuous, 0–38)
#   c_charge_degree — felony (F) or misdemeanor (M) → encoded as 1 or 0
#   sex             — Male or Female → encoded as 1 or 0
#   juv_fel_count   — juvenile felony offense count
#   juv_misd_count  — juvenile misdemeanor count
#   juv_other_count — juvenile other offense count
#
# TARGET:    two_year_recid  (0 = did not reoffend, 1 = reoffended within 2 yrs)
# PROTECTED: race  (kept separate, NEVER a model input)
#
# =============================================================================

import os
import urllib.request
import pandas as pd
import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

# The URL where ProPublica published their dataset on GitHub.
# This is fully public — no API key, no login, no download limit.
COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)

# Where to save the cleaned CSV so we do not re-download on every run.
# The data/ folder is created by the project setup step.
CACHE_PATH = os.path.join("data", "compas_clean.csv")

# The features we will pass to our ML models.
# IMPORTANT: 'race' is deliberately absent from this list.
FEATURE_COLUMNS = [
    "age",             # continuous
    "priors_count",    # continuous
    "charge_degree",   # encoded: F=1, M=0  (we create this from c_charge_degree)
    "sex_male",        # encoded: Male=1, Female=0  (we create this from sex)
    "juv_fel_count",   # continuous (may be 0 for all)
    "juv_misd_count",  # continuous
    "juv_other_count", # continuous
]

# The target column: did this person reoffend within 2 years?
TARGET_COLUMN = "two_year_recid"

# The protected attribute: used only for fairness analysis, never as a feature.
RACE_COLUMN = "race"

# Human-readable names for feature display in charts
FEATURE_LABELS = {
    "age":             "Age",
    "priors_count":    "Prior Offenses",
    "charge_degree":   "Charge Degree (F/M)",
    "sex_male":        "Sex (Male)",
    "juv_fel_count":   "Juvenile Felonies",
    "juv_misd_count":  "Juvenile Misdemeanors",
    "juv_other_count": "Juvenile Other",
}


# =============================================================================
# STEP 1 — DOWNLOAD AND CLEAN
# =============================================================================

def download_and_clean(verbose=True):
    """
    Downloads the ProPublica COMPAS dataset, applies their exact filtering
    methodology, encodes categorical features, and saves a clean CSV locally.

    On subsequent calls, loads from the cached file rather than downloading
    again (saves time and bandwidth).

    What "cleaning" means here:
      1. Apply ProPublica's four filter conditions (explained below)
      2. Encode c_charge_degree: 'F' → 1, 'M' → 0
      3. Encode sex: 'Male' → 1, 'Female' → 0
      4. Fill any missing juvenile counts with 0
      5. Drop rows where essential columns are missing

    Returns:
        str: Path to the saved clean CSV file.
    """

    # ── Use cache if it already exists ────────────────────────────────────
    os.makedirs("data", exist_ok=True)  # create data/ folder if needed

    if os.path.exists(CACHE_PATH):
        if verbose:
            print(f"  Using cached dataset at: {CACHE_PATH}")
        return CACHE_PATH

    # ── Download raw CSV from ProPublica's GitHub ─────────────────────────
    if verbose:
        print("  Downloading COMPAS dataset from ProPublica GitHub...")

    temp_path = CACHE_PATH + ".tmp"
    try:
        urllib.request.urlretrieve(COMPAS_URL, temp_path)
        raw_df = pd.read_csv(temp_path, low_memory=False)
        os.remove(temp_path)
    except Exception as err:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"  Download failed: {err}")

    if verbose:
        print(f"  Raw rows downloaded: {len(raw_df)}")

    # ── Apply ProPublica's four filtering criteria ─────────────────────────
    #
    # These are the exact conditions from ProPublica's published R script:
    # github.com/propublica/compas-analysis/blob/master/compas-analysis.Rmd
    #
    # Filter 1: Screening must happen close to the arrest date.
    #   days_b_screening_arrest is how many days before/after arrest
    #   the COMPAS screen occurred. Large gaps indicate edge cases that
    #   ProPublica excluded for consistency.
    df = raw_df[
        (raw_df["days_b_screening_arrest"] >= -30) &
        (raw_df["days_b_screening_arrest"] <=  30)
    ]

    # Filter 2: Exclude cases where recidivism data is missing.
    #   is_recid = -1 means the courthouse records did not contain
    #   a usable recidivism outcome for this defendant.
    df = df[df["is_recid"] != -1]

    # Filter 3: Exclude ordinance violations (traffic, parking, etc.).
    #   c_charge_degree = 'O' marks these minor non-criminal charges.
    #   ProPublica wanted to focus on genuine criminal recidivism.
    df = df[df["c_charge_degree"] != "O"]

    # Filter 4: Keep only defendants who were actually scored by COMPAS.
    #   score_text = 'N/A' means COMPAS did not produce a score for them.
    df = df[df["score_text"] != "N/A"]

    if verbose:
        print(f"  Rows after ProPublica filtering: {len(df)}")

    # ── Select and clean columns we need ──────────────────────────────────
    # Keep only the columns relevant to our analysis.
    keep = [
        "age", "race", "sex",
        "priors_count", "c_charge_degree", "c_charge_desc",
        "juv_fel_count", "juv_misd_count", "juv_other_count",
        "two_year_recid", "decile_score", "score_text",
    ]
    # Only keep columns that actually exist in this CSV version
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy().reset_index(drop=True)

    # ── Encode categorical columns as numbers ──────────────────────────────
    #
    # Why not one-hot encoding?
    # Because our recourse algorithms work better with ordinal integers,
    # and for trees and logistic regression these binary encodings work fine.
    #
    # Charge degree: Felony is more serious than Misdemeanor.
    # Encoding F=1, M=0 preserves this ordering.
    df["charge_degree"] = (df["c_charge_degree"] == "F").astype(int)

    # Sex: Male=1, Female=0. Simple binary encoding.
    df["sex_male"] = (df["sex"] == "Male").astype(int)

    # ── Handle juvenile offense counts ────────────────────────────────────
    # Some records have missing values here. A missing count means 0 offenses
    # (the defendant has no juvenile record — it is truly absent, not unknown).
    for col in ["juv_fel_count", "juv_misd_count", "juv_other_count"]:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
        else:
            # If the column is not in the CSV at all, create it as all zeros
            df[col] = 0

    # ── Enforce numeric types on key columns ──────────────────────────────
    # Ensure no string values slipped through on critical columns
    df["age"]           = pd.to_numeric(df["age"],           errors="coerce")
    df["priors_count"]  = pd.to_numeric(df["priors_count"],  errors="coerce")
    df["two_year_recid"]= pd.to_numeric(df["two_year_recid"],errors="coerce")

    # Drop the small number of rows where essential values are truly missing
    df = df.dropna(subset=["age", "priors_count", "two_year_recid"])
    df = df.reset_index(drop=True)

    # ── Save clean CSV to disk ─────────────────────────────────────────────
    df.to_csv(CACHE_PATH, index=False)

    if verbose:
        recid_rate = df["two_year_recid"].mean() * 100
        print(f"  Final clean rows:   {len(df)}")
        print(f"  Recidivism rate:    {recid_rate:.1f}%")
        print(f"  Saved to:           {CACHE_PATH}")

    return CACHE_PATH


# =============================================================================
# STEP 2 — LOAD DATA
# =============================================================================

def load_data(verbose=True):
    """
    Loads the cleaned COMPAS dataset and returns it in the form needed
    by the Rashomon set builder and fairness analysis.

    This function calls download_and_clean() if needed, then returns
    four things:

      X    — the feature matrix (DataFrame, shape ~6172 × 7)
             This is what our ML models will be trained on.
             Does NOT contain race.

      y    — the target labels (Series, values are 0 or 1)
             0 = did not reoffend within 2 years
             1 = did reoffend within 2 years

      df   — the full cleaned DataFrame with all columns
             Used for the dataset summary page in the app.

      race — the race label for each defendant (Series of strings)
             Values: 'African-American', 'Caucasian', 'Hispanic', etc.
             This is ONLY used for computing fairness metrics —
             never passed to the model.

    Args:
        verbose (bool): Print progress messages (default True)

    Returns:
        tuple: (X, y, df, race)
    """
    path = download_and_clean(verbose=verbose)
    df   = pd.read_csv(path)

    # Build the feature matrix from only the approved feature columns
    # (race is excluded — it is the protected attribute)
    X = df[FEATURE_COLUMNS].astype(float)

    # Target: binary label
    y = df[TARGET_COLUMN].astype(int)

    # Protected attribute: race strings (e.g. 'African-American')
    race = df[RACE_COLUMN]

    return X, y, df, race


# =============================================================================
# STEP 3 — SUMMARY STATISTICS (used by the app)
# =============================================================================

def get_dataset_summary(df, X, y, race):
    """
    Returns a dictionary of headline numbers shown on the Home page
    and the About page.

    This is purely descriptive — it just counts and averages things
    from the data we already loaded.

    Args:
        df   (DataFrame): Full cleaned DataFrame
        X    (DataFrame): Feature matrix
        y    (Series):    Binary labels
        race (Series):    Race labels

    Returns:
        dict: Summary statistics
    """
    black_mask = (race == "African-American")
    white_mask = (race == "Caucasian")

    return {
        "total_defendants":  len(df),
        "n_features":        X.shape[1],
        "recidivism_rate":   round(float(y.mean()) * 100, 1),
        "n_black":           int(black_mask.sum()),
        "n_white":           int(white_mask.sum()),
        "recid_rate_black":  round(float(y[black_mask].mean()) * 100, 1),
        "recid_rate_white":  round(float(y[white_mask].mean()) * 100, 1),
        "pct_male":          round(float((df["sex"] == "Male").mean()) * 100, 1),
        "age_mean":          round(float(df["age"].mean()), 1),
        "priors_mean":       round(float(df["priors_count"].mean()), 1),
    }


def get_race_stats(df, race):
    """
    Returns a breakdown of the dataset by racial group:
    how many defendants, what fraction of the dataset, and
    what the recidivism rate is for each group.

    Used on the Home page to give context about who is in the data.

    Returns:
        dict: { race_label: { 'count', 'fraction', 'recid_rate' } }
    """
    stats = {}
    total = len(df)

    for r in sorted(race.unique()):
        mask         = (race == r)
        count        = int(mask.sum())
        recid_rate   = float(df.loc[mask, "two_year_recid"].mean()) * 100
        stats[r] = {
            "count":      count,
            "fraction":   round(count / total * 100, 1),
            "recid_rate": round(recid_rate, 1),
        }

    return stats


# =============================================================================
# MAIN BLOCK — run this file directly to test it
# =============================================================================
# Usage:  python audit/data.py
# This is how we verify Phase 1 is working correctly.

if __name__ == "__main__":

    print("=" * 60)
    print("PHASE 1 VERIFICATION — data.py")
    print("=" * 60)

    # Load the data
    X, y, df, race = load_data(verbose=True)

    # ── Check 1: Row count matches ProPublica's published number ──────────
    print(f"\nCheck 1 — Row count")
    print(f"  Expected: 6,172  (ProPublica's published count)")
    print(f"  Got:      {len(df):,}")
    assert len(df) == 6172, f"FAIL: expected 6172, got {len(df)}"
    print(f"  ✓ PASS")

    # ── Check 2: Feature matrix has the right shape ────────────────────────
    print(f"\nCheck 2 — Feature matrix shape")
    print(f"  Expected: (6172, 7)")
    print(f"  Got:      {X.shape}")
    assert X.shape == (6172, 7), f"FAIL: unexpected shape {X.shape}"
    print(f"  ✓ PASS")

    # ── Check 3: Target is binary with no missing values ──────────────────
    print(f"\nCheck 3 — Target column")
    unique_vals = sorted(y.unique())
    print(f"  Expected values: [0, 1]")
    print(f"  Got values:      {unique_vals}")
    assert unique_vals == [0, 1], f"FAIL: unexpected values {unique_vals}"
    assert y.isna().sum() == 0, "FAIL: target has missing values"
    print(f"  ✓ PASS")

    # ── Check 4: Race is never in the feature matrix ───────────────────────
    print(f"\nCheck 4 — Race excluded from features")
    print(f"  Feature columns: {list(X.columns)}")
    assert "race" not in X.columns, "FAIL: race is in the feature matrix!"
    print(f"  ✓ PASS — 'race' is not in X")

    # ── Print descriptive stats ────────────────────────────────────────────
    print(f"\n{'─'*55}")
    print(f"DESCRIPTIVE STATISTICS")
    print(f"{'─'*55}")

    summary    = get_dataset_summary(df, X, y, race)
    race_stats = get_race_stats(df, race)

    print(f"\nOverall:")
    print(f"  Total defendants:   {summary['total_defendants']:,}")
    print(f"  Recidivism rate:    {summary['recidivism_rate']}%")
    print(f"  Male defendants:    {summary['pct_male']}%")
    print(f"  Avg age:            {summary['age_mean']} years")
    print(f"  Avg prior offenses: {summary['priors_mean']}")

    print(f"\nClass balance:")
    print(f"  No recidivism (y=0): {(y==0).sum():,}  ({(y==0).mean()*100:.1f}%)")
    print(f"  Recidivated   (y=1): {(y==1).sum():,}  ({(y==1).mean()*100:.1f}%)")

    print(f"\nRace breakdown (sorted by count):")
    for r, s in sorted(race_stats.items(), key=lambda x: -x[1]["count"]):
        print(f"  {r:<22}  n={s['count']:5,}  "
              f"({s['fraction']:5.1f}% of dataset)  "
              f"recid rate: {s['recid_rate']:.1f}%")

    print(f"\nProPublica comparison group:")
    bw = race_stats.get("African-American", {})
    wh = race_stats.get("Caucasian", {})
    print(f"  Black defendants: {bw.get('count','?'):,}  "
          f"(recidivism rate: {bw.get('recid_rate','?')}%)")
    print(f"  White defendants: {wh.get('count','?'):,}  "
          f"(recidivism rate: {wh.get('recid_rate','?')}%)")
    print(f"  Base rate gap:    "
          f"{bw.get('recid_rate',0) - wh.get('recid_rate',0):.1f} percentage points")
    print(f"\n  NOTE: This base rate gap (52.3% vs 39.1%) is why the")
    print(f"  ProPublica vs Northpointe debate is mathematically complex.")
    print(f"  The Chouldechova theorem says: if base rates differ,")
    print(f"  you cannot simultaneously equalize all fairness criteria.")

    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE ✓ — data.py is verified and working.")
    print(f"{'='*60}")
