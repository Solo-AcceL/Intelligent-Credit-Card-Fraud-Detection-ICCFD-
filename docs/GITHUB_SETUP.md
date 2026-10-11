# Create the GitHub repository and push this capstone

## 1. Create an empty remote repository

1. Sign in to https://github.com/.
2. Select **+ → New repository**.
3. Name it `intelligent-credit-card-fraud-detection`.
4. Add the description: `Explainable credit-card fraud triage with DuckDB, XGBoost, statistical validation and Streamlit`.
5. Select Public for a portfolio repository or Private until approval.
6. Leave **Add a README**, **.gitignore**, and **licence** unchecked because this project already includes them.
7. Select **Create repository** and copy the HTTPS remote URL shown by GitHub.

## 2. Extract and prepare the local project

Extract the ZIP. Open a terminal in the `intelligent-credit-card-fraud-detection` directory, then copy the supplied `creditcard.csv` into `data/raw/creditcard.csv`. The raw data are intentionally excluded from Git because of file size and separate data licence terms.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
PYTHONPATH=src python -m src.run_pipeline
pytest -q
streamlit run app/streamlit_app.py
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -m src.run_pipeline
pytest -q
streamlit run app/streamlit_app.py
```

The full pipeline writes DuckDB profiling and a de-duplicated Parquet dataset, hypothesis-test results, model comparison, model metadata, threshold analysis, diagnostic figures, and Power BI-ready exports. The DOCX report and PPTX executive deck are included as review artifacts; refresh their metrics if you change the dataset, split or model.

## 3. Initialize Git and push the project

Run these commands from the project root after confirming the raw CSV is not staged. Replace `<YOUR-USERNAME>` with your GitHub username.

```bash
git status
git init
git add .
git status
# Verify data/raw/creditcard.csv is not listed. If it appears, unstage it with: git rm --cached data/raw/creditcard.csv
git commit -m "feat: build intelligent credit card fraud capstone"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/intelligent-credit-card-fraud-detection.git
git push -u origin main
```

If Git asks you to authenticate, use Git Credential Manager/browser sign-in or `gh auth login`. Never paste a token into the remote URL or commit credentials.

## 4. Verify and maintain repository hygiene

- Open the GitHub repository page and confirm the README and folders render properly.
- Use meaningful follow-up commits such as `feat: add threshold cost analysis` and `test: validate feature engineering`.
- Keep `.venv`, the raw CSV, Parquet files, secrets, caches and bulky predictions out of Git. The `.gitignore` already excludes them.
- Check `git status` before every commit. Never publish raw data unless the source licence and platform terms permit it.
- `outputs/figures/` contains small visuals. Keep notebooks supporting rather than replacing modular code.
