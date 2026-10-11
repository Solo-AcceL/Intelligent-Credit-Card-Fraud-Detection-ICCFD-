"""Feature transformation shared by training, batch scoring, and the app."""
import numpy as np
import pandas as pd

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
INPUT_COLUMNS = ["Time", *PCA_FEATURES, "Amount"]
FEATURE_COLUMNS = [*PCA_FEATURES, "Amount", "log_amount", "time_phase_sin", "time_phase_cos"]

def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required input columns: {sorted(missing)}")
    out = frame[PCA_FEATURES].copy()
    amount = pd.to_numeric(frame["Amount"], errors="raise").astype(float).to_numpy()
    elapsed = pd.to_numeric(frame["Time"], errors="raise").astype(float).to_numpy()
    if np.any(amount < 0):
        raise ValueError("Amount must be non-negative for log1p transformation")
    out["Amount"] = amount
    out["log_amount"] = np.log1p(amount)
    phase = 2 * np.pi * (elapsed % 86400.0) / 86400.0
    out["time_phase_sin"] = np.sin(phase)
    out["time_phase_cos"] = np.cos(phase)
    return out[FEATURE_COLUMNS]
