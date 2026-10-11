"""Reproduce data quality, statistical, model, and dashboard export artifacts.

Expected input: data/raw/creditcard.csv (download separately; never committed).
Run from the repository root with PYTHONPATH=src.
"""
from pathlib import Path
import json
import pandas as pd
from fraud_capstone.duckdb_pipeline import run_duckdb_pipeline
from fraud_capstone.statistics import run_hypothesis_tests
from fraud_capstone.train import main as train_main
from fraud_capstone.reporting import generate_reporting


def main():
    raw = Path('data/raw/creditcard.csv')
    if not raw.exists():
        raise FileNotFoundError('Place the supplied creditcard.csv at data/raw/creditcard.csv. See docs/DATA_PROVENANCE.md.')
    Path('outputs').mkdir(parents=True, exist_ok=True)
    meta = run_duckdb_pipeline(raw, 'data/processed')
    clean_path = Path('data/processed/clean_transactions.parquet')
    clean = pd.read_parquet(clean_path).sort_values('Time', kind='stable').reset_index(drop=True)
    run_hypothesis_tests(clean).to_csv('outputs/hypothesis_tests.csv', index=False)
    train_meta = train_main(clean_path, 'models', 'outputs')
    generate_reporting(clean, 'outputs', 'models')
    print('Pipeline completed. Data profile, hypothesis tests, model comparison, threshold analysis, figures and Power BI exports regenerated.')
    print(json.dumps({'ingestion': meta, 'model': train_meta}, indent=2))


if __name__ == '__main__':
    main()
