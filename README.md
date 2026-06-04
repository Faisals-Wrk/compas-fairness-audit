# Equally Accurate, Unequally Fair
### A COMPAS Rashomon Fairness Audit

---

## The Question

In 2016, ProPublica found that COMPAS — an algorithm used to decide bail in Florida — falsely labelled Black defendants as high-risk at nearly **twice the rate** of white defendants. Northpointe responded that the algorithm was fair by a different measure. Both were right. The debate became one of the most famous in AI ethics.

But both sides were arguing about a **single deployed model**.

We asked a deeper question:

> *Within all the equally-accurate models that could have been deployed, how much did racial bias vary?*

---

## The Finding

We trained **49 models** across four algorithm families on the ProPublica COMPAS dataset. The **40 models within 2% of peak accuracy** form the ε-Rashomon set — all are defensibly "the best model" by standard accuracy metrics.

Within that set:

| Metric | Value |
|--------|-------|
| FPR gap — fairest model | **0.116** |
| FPR gap — most biased model | **0.200** |
| Range within equally-accurate models | **0.084** |
| AUC cost of choosing the fairest model | **0.0016** |
| Defendants with different outcomes | **205 / 1,235 (16.6%)** |

The bias was not technically inevitable. Fairer models existed in the same accuracy range. **Choosing the most biased model was a choice made by omission.**

---

## Research Connection

This project is an interactive empirical demonstration of:

> **Semenova, Hsu, Chen, Zhong** — *"The Double-Edged Nature of the Rashomon Set for Trustworthy Machine Learning"* (arXiv 2025)

The paper proves that the Rashomon set creates both:
- **Fairness risk** — the set almost always contains models with very different fairness profiles
- **Fairness opportunity** — fairer models are almost always available within the set at negligible accuracy cost

This app shows both sides empirically on real criminal justice data.

---

## Five Interactive Pages

| Page | What it shows |
|------|--------------|
| **The Case** | The ProPublica story, dataset context, and our key finding |
| **The Audit** | Scatter plot: 49 models, accuracy vs racial bias |
| **Model Explorer** | Drill into any model — feature weights, decision rules |
| **Who Bears It** | 205 defendants with different outcomes + Chouldechova's impossibility theorem |
| **About** | Full methodology, research citations, limitations |

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/Faisals-Wrk/compas-fairness-audit.git
cd compas-fairness-audit

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**First run:** ~8 seconds (downloads COMPAS data + trains 49 models).
**Every subsequent interaction:** instant (all results cached in memory).

---

## Project Structure

```
compas-fairness-audit/
├── app.py                    # Streamlit application — 5 pages
├── requirements.txt          # Python dependencies
└── audit/
    ├── __init__.py
    ├── data.py               # COMPAS download, ProPublica filtering, encoding
    ├── rashomon.py           # 49-model family, ε-Rashomon set construction
    ├── fairness.py           # FPR / FNR / FDR metrics by race for all models
    └── analyze.py            # Pareto frontier, summary stats, app data prep
```

---

## Methodology

### Data
- **Source:** ProPublica COMPAS dataset (compas-scores-two-years.csv), Broward County FL 2013–2014
- **Filter:** ProPublica's exact 4-step criteria → 7,214 raw rows → **6,172 defendants**
- **Features:** Age, prior offenses, charge degree, sex, juvenile counts (7 features)
- **Target:** `two_year_recid` — did the defendant reoffend within 2 years?
- **Protected attribute:** Race — excluded from model inputs, used only for fairness evaluation

### Model Family (49 configurations)
| Algorithm | Configs | Hyperparameters |
|-----------|---------|-----------------|
| Decision Tree | 21 | depth 2–8 × min_leaf 5/10/20 |
| Logistic Regression | 10 | C ∈ {0.001, 0.01, …, 100} |
| Random Forest | 12 | depth {3,5,7,∞} × min_leaf {5,10,20} |
| Gradient Boosting | 6 | depth {2,3,5} × lr {0.05, 0.1} |

### Rashomon Set
- **ε = 0.02**: models with test AUC ≥ best\_AUC − 0.02 qualify
- **40 of 49 models** meet this criterion
- **80/20 train/test split** stratified by race × label
- All fairness metrics computed on **held-out test data only**

### Fairness Metrics
- **FPR disparity** = |FPR\_Black − FPR\_White| — ProPublica's metric (primary)
- **FNR disparity** = |FNR\_Black − FNR\_White| — Northpointe's metric
- **FDR disparity** = |FDR\_Black − FDR\_White| — predictive parity
- **Classification threshold:** P(recidivism) ≥ 0.5

---

## Reproducibility

| Number | Description |
|--------|-------------|
| 6,172 | Defendants (after ProPublica filter) |
| 49 | Model configurations trained |
| 40 | In Rashomon set (ε = 0.02) |
| 0.7317 | Best test AUC |
| 0.116 | Minimum FPR gap (fairest model) |
| 0.200 | Maximum FPR gap (most biased model) |
| 0.0016 | AUC cost of choosing the fairest model |
| 205 | Defendants with different outcomes between fairest and most biased |

---

---

## Deployment

The app is deployed on **Streamlit Community Cloud** (free):

🔗 **[compas-fairness-audit.streamlit.app](https://compas-fairness-audit-rnramsesh82ybuccgztqf3.streamlit.app/)**

> First load takes some time.

---


## Limitations

**Threshold:** FPR/FNR values depend on the 0.5 classification threshold. COMPAS used a 10-point proprietary scale. The qualitative finding — bias varies enormously within the Rashomon set — is threshold-independent.

**Not the actual COMPAS algorithm:** We train our own models on the features available in the ProPublica CSV. The finding applies to this data space, not specifically to Northpointe's proprietary code.

**Historical data:** 2013–2014 Broward County. Policing patterns and recidivism rates may differ in other jurisdictions or time periods.

---

## Citations

```bibtex
@article{semenova2025doubledged,
  title   = {The Double-Edged Nature of the Rashomon Set for Trustworthy
             Machine Learning},
  author  = {Semenova, Lesia and Hsu, Harry and Chen, Jiachang and Zhong, Cynthia},
  journal = {arXiv preprint},
  year    = {2025}
}

@article{semenova2022existence,
  title   = {On the Existence of Simpler Machine Learning Models},
  author  = {Semenova, Lesia and Rudin, Cynthia and Parr, Ronald},
  journal = {Proceedings of FAccT},
  year    = {2022}
}

@article{angwin2016machine,
  title   = {Machine Bias},
  author  = {Angwin, Julia and Larson, Jeff and Mattu, Surya and Kirchner, Lauren},
  journal = {ProPublica},
  year    = {2016}
}

@article{chouldechova2017fair,
  title   = {A Study of the Impossibility of Fairness},
  author  = {Chouldechova, Alexandra},
  journal = {Criminal Justice and Behavior},
  year    = {2017}
}
```

*Built as part of a machine learning interpretability portfolio.*
