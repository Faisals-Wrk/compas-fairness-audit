# audit/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
# This empty file tells Python: "treat the audit/ folder as a package."
# Without it, we would get ImportError when app.py tries to do:
#   from audit.data import load_data
#
# All four modules that make up the analysis pipeline live here:
#   data.py      — download + clean the COMPAS dataset
#   rashomon.py  — train 49 models, find the Rashomon set
#   fairness.py  — measure racial bias for every model
#   analyze.py   — package results for the Streamlit app
# ─────────────────────────────────────────────────────────────────────────────
