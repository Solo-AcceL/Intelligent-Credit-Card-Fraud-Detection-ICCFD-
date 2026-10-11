"""Batch scoring helper and native XGBoost TreeSHAP contribution extraction."""
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from .features import engineer_features


def load_model(model_path='models/xgb_fraud_model.json'):
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    return model


def score_frame(frame: pd.DataFrame, model_path='models/xgb_fraud_model.json', threshold=.5) -> pd.DataFrame:
    if not 0 <= threshold <= 1:
        raise ValueError('threshold must be between 0 and 1')
    model = load_model(model_path)
    features = engineer_features(frame)
    scores = model.predict_proba(features)[:, 1]
    out = frame.copy()
    out['fraud_score'] = scores
    out['predicted_fraud'] = (scores >= threshold).astype(int)
    out['risk_tier'] = np.select([scores < .1, scores < .5, scores < .8], ['Low', 'Moderate', 'High'], default='Critical')
    # XGBoost pred_contribs returns one contribution per feature plus a bias term.
    dm = xgb.DMatrix(features, feature_names=list(features.columns))
    contrib = model.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    names = np.asarray(list(features.columns))
    top = np.abs(contrib).argmax(axis=1)
    out['top_contributing_feature'] = names[top]
    out['top_feature_contribution'] = contrib[np.arange(len(out)), top]
    return out
