# Intelligent Credit Card Fraud Detection & Explainable AI Fraud Analyst

> Flag the rare transactions that deserve investigation while making the precision–recall and review-cost trade-off visible to analysts.

![Python](https://img.shields.io/badge/Python-3.11%2B-blue) ![Status](https://img.shields.io/badge/status-capstone%20prototype-orange) ![Tests](https://img.shields.io/badge/tests-pytest-informational) ![Licence](https://img.shields.io/badge/code%20licence-MIT-green)

**Program / track:** Fill in your registered upGrad program and track.
**Author:** Fill in your name.

## Business problem

Fraud cases are rare, so accuracy can look excellent while a system misses fraud. This project builds a reproducible detection-and-triage pipeline over the ULB/Kaggle benchmark data, compares a logistic baseline with XGBoost, uses chronological evaluation, tests data hypotheses, surfaces local model attribution, and provides an interactive Streamlit dashboard plus an optional FastAPI endpoint. It is a decision-support prototype, not an online authorization system.

## Headline results (measured on the supplied dataset)

| Metric | Result |
|---|---:|
| Raw transactions | 284,807 |
| Exact duplicate rows identified | 1,081 (19 duplicated rows had `Class=1`) |
| Rows after exact deduplication | 283,726 |
| Fraud cases after deduplication | 473 (0.1667%; Wilson 95% CI 0.1524–0.1824%) |
| Chronological held-out test size | 56,746 transactions, 74 frauds |
| XGBoost average precision on test | 0.786 |
| XGBoost ROC-AUC on test | 0.980 |
| XGBoost precision / recall / F1 at score threshold 0.50 | 0.414 / 0.811 / 0.548 |

The train/validation/test split is chronological (60/20/20). Results are specific to this 2013 two-day benchmark and are not estimates of current issuer performance. Model scores are **not calibrated probabilities**.

## Architecture

```text
creditcard.csv
  -> DuckDB SQL profile + exact-dedup Parquet
  -> feature engineering (PCA features, Amount/log1p, elapsed-time phase)
  -> chronological train / validation / test
  -> Logistic Regression baseline + XGBoost
  -> PR-focused evaluation + validation cost/threshold analysis
  -> TreeSHAP local attributions + risk tiers
  -> Streamlit monitoring / investigation UI + optional FastAPI
  -> Power BI-ready CSV exports + technical report / executive deck
```

## Dataset provenance

Source: [Kaggle — Credit Card Fraud Detection (ULB Machine Learning Group)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The original CSV contains 284,807 rows, 31 columns, and 492 fraud cases. `V1`–`V28` are PCA-transformed/anonymized features, `Time` is elapsed seconds and `Amount` is transaction amount. The Kaggle source page identifies database contents as Open Database; review ODbL 1.0 and the current source-page terms before redistribution. The MIT licence in this repo applies to code only; it does not relicense the dataset. See [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

**Raw dataset is intentionally not committed** (about 150 MB and governed by separate data terms). Copy the already supplied file to `data/raw/creditcard.csv`. The bundled small sample exists only for offline UI demonstration.

## Local setup

Python 3.11+ recommended. From the repository root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `creditcard.csv` to `data/raw/creditcard.csv`; then run:

```bash
PYTHONPATH=src python -m src.run_pipeline
pytest -q
streamlit run app/streamlit_app.py
```

Optional API server:

```bash
uvicorn fraud_capstone.api:app --app-dir src --reload
# Open http://127.0.0.1:8000/docs
```

No API keys or secrets are required. Do not add external credentials to the repository.

## Use the demo artifacts immediately

The ZIP contains a compact, fraud-enriched demo sample, a trained XGBoost model, model metadata, aggregate metrics and figures so reviewers can inspect the project without the full raw CSV. The sample is not suitable for prevalence estimates. `streamlit run app/streamlit_app.py` provides monitoring, investigation and model-performance views. For a full rerun, add `data/raw/creditcard.csv` and execute the pipeline.

## Repository map

```text
intelligent-credit-card-fraud-detection/
├── app/                 # Interactive Streamlit decision interface
├── assets/              # Report and README figure copies
├── data/raw/            # Local-only original CSV (not committed)
├── data/sample/         # Small labelled sample for offline demo
├── data/processed/      # Generated Parquet and processed data (not committed)
├── docs/                # Data provenance, decision policy, GitHub steps
├── models/              # XGBoost JSON artifact, metadata and baseline artifacts
├── notebooks/           # Supporting walkthrough, not production entry point
├── outputs/             # Metrics, test results, figures and Power BI-ready exports
├── reports/             # Technical report and executive deck copies
├── src/fraud_capstone/  # Feature, DuckDB, statistics, model and scoring modules
├── tests/               # Lightweight unit / artefact tests
├── requirements.txt     # Pinned/restricted dependencies
├── LICENSE              # MIT licence for project code only
└── README.md            # Onboarding document
```

## Analytical findings

- The source data have no missing cells in the supplied file. This does not justify blind imputation; the pipeline records missingness before transformation.
- Exact duplicate rows were documented and excluded from primary modelling to avoid identical-record leakage. Since no transaction ID exists, this is a modelling assumption, not proof of duplicated real-world events.
- Fraud amount median (dataset amount units) is 9.82 vs 22.00 for legitimate rows; Mann–Whitney U p=2.69e-05. Fraud mean amount (123.87) exceeds legitimate mean (88.41) because fraud amounts are heavy-tailed.
- Fraud prevalence differs between the first and second chronological halves (two-proportion z test p=0.0189); this is evidence of time variation, not a causal explanation.
- Threshold tuning uses a declared **illustrative** cost ratio (false negative 20 units; false positive 1 unit). The threshold minimizing validation cost was 0.831; on the held-out test, its cost was 384 units compared with 365 at threshold 0.50. That failure to generalize is deliberately reported and argues for more representative calibration data before operational decisions.

## How to navigate the evidence

- [`outputs/model_comparison.csv`](outputs/model_comparison.csv): primary held-out metrics.
- [`outputs/hypothesis_tests.csv`](outputs/hypothesis_tests.csv): formal statistical tests.
- [`outputs/threshold_analysis_validation.csv`](outputs/threshold_analysis_validation.csv): validation threshold/cost sweep.
- [`outputs/threshold_analysis.csv`](outputs/threshold_analysis.csv): held-out test comparisons at default and validation-selected thresholds.
- [`outputs/feature_importance.csv`](outputs/feature_importance.csv): model-level gain; not causal importance.
- `outputs/figures/`: imbalance, amount distribution, time variation, PR curve, confusion matrix, feature importance and threshold cost.
- [`docs/GITHUB_SETUP.md`](docs/GITHUB_SETUP.md): create the GitHub repo and push commands.

## Run a Power BI report

In Power BI Desktop, use **Get Data → Text/CSV** to import `outputs/powerbi_kpis.csv`, `outputs/powerbi_time_buckets.csv`, and (after a full run) `outputs/powerbi_high_risk_transactions.csv`. Recommended pages: executive KPI cards, fraud by elapsed-time bucket, risk tier distribution, model performance, and top flagged transactions. The local Streamlit app supplies the required interactive decision interface even if Power BI Desktop is not available on the current OS.

## Limitations and next steps

- Benchmark data are anonymized, from September 2013, and span only two days; no current customer/device/merchant signals or identity features are available.
- `V1`–`V28` are opaque PCA components; attribution can identify influential components but cannot assign them human business semantics.
- Exact-duplicate handling could discard repeated identical events; run sensitivity analyses with/without de-duplication.
- Model scores are not calibrated probabilities; validate calibration and thresholds on recent data before any real decision.
- False-positive/false-negative cost units are scenario assumptions, not observed money.
- Productionisation would require current labels, delayed-label handling, drift monitoring, audit logging, security testing, privacy/fairness assessment, human-in-the-loop workflows and shadow deployment.
- A future GenAI investigation analyst could summarize structured model attributions against an approved policy knowledge base, but only after privacy/security checks; an LLM should not independently authorize transactions.

## References

- Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- DuckDB CSV ingestion: https://duckdb.org/docs/current/clients/python/data_ingestion
- Scikit-learn model evaluation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Streamlit documentation: https://docs.streamlit.io/
- FastAPI tutorial: https://fastapi.tiangolo.com/tutorial/
- NIST AI Risk Management Framework: https://www.nist.gov/itl/ai-risk-management-framework
