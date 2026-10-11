# Deliverables included

- `Technical_Report.docx`: 12-page corporate report organized under the six required top-level headings.
- `Executive_Presentation.pptx`: 11-slide executive narrative with methodology, results, trade-offs, next steps and appendix.
- `../README.md`: stand-alone repository onboarding.
- `../docs/GITHUB_SETUP.md`: step-by-step repository creation and push procedure.
- `../outputs/`: measured metrics, hypothesis tests, threshold sweeps, feature importance and figures.
- `../app/streamlit_app.py`: interactive decision interface.
- `../src/`: modular source code and a DuckDB SQL ingestion layer.

The raw 150 MB CSV is not included in the repository ZIP. Copy the supplied dataset into `data/raw/creditcard.csv` for a full local rerun.

Measured holdout summary: XGBoost AP 0.786, ROC-AUC 0.980; threshold 0.50 recall 0.811, precision 0.414.
