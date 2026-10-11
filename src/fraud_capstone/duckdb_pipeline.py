"""DuckDB ingestion, profiling, deduplication and time-bucket aggregation.

The raw CSV is read-only. Outputs are created under data/processed and outputs/.
"""
from __future__ import annotations
from pathlib import Path
import json
import duckdb
import pandas as pd


def _sql_literal(value: str) -> str:
    """Quote a local path for use as a SQL string literal."""
    return "'" + value.replace("'", "''") + "'"


def run_duckdb_pipeline(raw_csv: str | Path, output_dir: str | Path) -> dict:
    raw_path = Path(raw_csv).resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {raw_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = output_dir.resolve().parent.parent
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    raw_sql = _sql_literal(str(raw_path))
    con = duckdb.connect()
    try:
        con.execute("SET threads TO 4")
        source = f"read_csv_auto({raw_sql}, header=true)"
        schema = con.execute(f"DESCRIBE SELECT * FROM {source}").df()
        # One SQL aggregation profiles all columns in a single scan.
        expressions = ["COUNT(*) AS row_count"]
        for col in schema["column_name"].tolist():
            qcol = '"' + col.replace('"', '""') + '"'
            expressions.append(f'COUNT(*) FILTER (WHERE {qcol} IS NULL) AS "missing__{col}"')
            expressions.append(f'COUNT(DISTINCT {qcol}) AS "distinct__{col}"')
        summary = con.execute(f"SELECT {', '.join(expressions)} FROM {source}").df().iloc[0]
        row_count = int(summary["row_count"])
        profile_rows = []
        for _, row in schema.iterrows():
            col = row["column_name"]
            missing = int(summary[f"missing__{col}"])
            profile_rows.append({
                "column": col,
                "data_type": row["column_type"],
                "row_count": row_count,
                "missing_count": missing,
                "missing_pct": (missing / row_count * 100.0) if row_count else 0.0,
                "distinct_count": int(summary[f"distinct__{col}"]),
            })
        pd.DataFrame(profile_rows).to_csv(outputs_dir / "data_quality_profile.csv", index=False)
        if "Class" not in schema["column_name"].tolist():
            raise ValueError("Required target column 'Class' is missing from the CSV")
        target = con.execute(
            f"SELECT SUM(CASE WHEN Class=1 THEN 1 ELSE 0 END) AS fraud_count FROM {source}"
        ).fetchone()[0]
        distinct_rows = int(con.execute(f"SELECT COUNT(*) FROM (SELECT DISTINCT * FROM {source})").fetchone()[0])
        duplicate_rows = row_count - distinct_rows
        parquet_path = (output_dir / "clean_transactions.parquet").resolve()
        parquet_sql = _sql_literal(str(parquet_path))
        con.execute(
            f"COPY (SELECT DISTINCT * FROM {source} ORDER BY Time) "
            f"TO {parquet_sql} (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        clean_source = f"read_parquet({_sql_literal(str(parquet_path))})"
        time_summary = con.execute(
            f"SELECT CAST(FLOOR(Time/21600) AS INTEGER) AS elapsed_6h_bucket, "
            f"COUNT(*) AS transactions, SUM(Class) AS fraud, AVG(Class)*100 AS fraud_rate_pct, "
            f"SUM(Amount) AS amount_total FROM {clean_source} "
            "GROUP BY 1 ORDER BY 1"
        ).df()
        time_summary.to_csv(outputs_dir / "powerbi_time_buckets.csv", index=False)
        metadata = {
            "source_csv": str(raw_path),
            "rows_raw": row_count,
            "rows_after_exact_dedup": distinct_rows,
            "exact_duplicate_rows": duplicate_rows,
            "fraud_raw": int(target or 0),
            "parquet_path": str(parquet_path),
            "profile_path": str(outputs_dir / "data_quality_profile.csv"),
        }
        (output_dir / "duckdb_run_metadata.json").write_text(json.dumps(metadata, indent=2))
        return metadata
    finally:
        con.close()
