"""Train baseline and boosted models with chronological validation and test splits."""
from __future__ import annotations
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, accuracy_score, balanced_accuracy_score, confusion_matrix,
)
from .features import engineer_features


def _metrics(y, score, threshold: float, model_name: str) -> dict:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "model": model_name,
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "flagged": int(tp + fp),
        "cost_units": float(20 * fn + fp),
    }


def main(data_path='data/processed/clean_transactions.parquet', model_dir='models', outputs_dir='outputs'):
    data_path = Path(data_path)
    model_dir = Path(model_dir)
    outputs_dir = Path(outputs_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(data_path).sort_values('Time', kind='stable').reset_index(drop=True)
    if not {'Time', 'Amount', 'Class'}.issubset(df.columns):
        raise ValueError('Parquet dataset must contain Time, Amount and Class')
    if df['Class'].nunique() != 2:
        raise ValueError('Target Class must contain both 0 and 1 labels')
    n = len(df)
    i1, i2 = int(.60 * n), int(.80 * n)
    tr, va, te = df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]
    Xtr, Xva, Xte = engineer_features(tr), engineer_features(va), engineer_features(te)
    ytr, yva, yte = tr.Class.to_numpy(), va.Class.to_numpy(), te.Class.to_numpy()
    if ytr.sum() == 0 or yva.sum() == 0 or yte.sum() == 0:
        raise ValueError('A chronological partition has no positive class; revise the split before training')

    # Scaled, class-weighted linear baseline.
    scaler = StandardScaler()
    Xtr_scaled = scaler.fit_transform(Xtr)
    Xva_scaled = scaler.transform(Xva)
    Xte_scaled = scaler.transform(Xte)
    lr = LogisticRegression(C=.1, class_weight='balanced', max_iter=2000, solver='lbfgs', random_state=42)
    lr.fit(Xtr_scaled, ytr)
    p_lr_test = lr.predict_proba(Xte_scaled)[:, 1]
    joblib.dump(lr, model_dir / 'logistic_baseline.joblib')
    joblib.dump(scaler, model_dir / 'logistic_scaler.joblib')

    neg, pos = int((ytr == 0).sum()), int(ytr.sum())
    model = xgb.XGBClassifier(
        n_estimators=220, max_depth=4, learning_rate=.05, subsample=.9,
        colsample_bytree=.9, objective='binary:logistic', eval_metric='aucpr',
        scale_pos_weight=neg / max(pos, 1), reg_lambda=2, min_child_weight=3,
        n_jobs=4, random_state=42, tree_method='hist'
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]

    costs = []
    for threshold in np.linspace(.001, .999, 250):
        pred = (pva >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(yva, pred, labels=[0, 1]).ravel()
        costs.append({
            'threshold': float(threshold), 'cost_units': float(20 * fn + fp),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        })
    best_val = min(costs, key=lambda r: r['cost_units'])
    selected_threshold = float(best_val['threshold'])
    model.save_model(model_dir / 'xgb_fraud_model.json')
    validation_metrics = _metrics(yva, pva, selected_threshold, 'XGBoost-validation-selected')
    xgb_default = _metrics(yte, pte, .5, 'XGBoost')
    xgb_selected = _metrics(yte, pte, selected_threshold, 'XGBoost-validation-selected')
    lr_metrics = _metrics(yte, p_lr_test, .5, 'Logistic Regression')
    comparison = pd.DataFrame([lr_metrics, xgb_default])
    comparison.to_csv(outputs_dir / 'model_comparison.csv', index=False)

    # Test prediction export stays small and can be consumed by the dashboard.
    pred = te[['Time', 'Amount', 'Class']].copy()
    pred['fraud_score'] = pte
    pred['prediction_0_5'] = (pte >= .5).astype(int)
    pred['prediction_validation_threshold'] = (pte >= selected_threshold).astype(int)
    pred['risk_tier'] = np.select([pte < .1, pte < .5, pte < .8], ['Low', 'Moderate', 'High'], default='Critical')
    pred['top_contributing_feature'] = 'not_computed'
    pred['top_feature_contribution'] = np.nan
    # Native XGBoost prediction contributions (TreeSHAP) for top explanation field.
    dtest = xgb.DMatrix(Xte, feature_names=list(Xte.columns))
    contributions = model.get_booster().predict(dtest, pred_contribs=True)[:, :-1]
    feature_names = np.asarray(list(Xte.columns))
    top = np.abs(contributions).argmax(axis=1)
    pred['top_contributing_feature'] = feature_names[top]
    pred['top_feature_contribution'] = contributions[np.arange(len(pred)), top]
    pred.to_csv(outputs_dir / 'test_predictions.csv', index=False)
    pd.DataFrame(costs).to_csv(outputs_dir / 'threshold_analysis_validation.csv', index=False)
    pd.DataFrame([
        {'operating_point': 'test_threshold_0.50', **xgb_default},
        {'operating_point': f'test_validation_selected_{selected_threshold:.6f}', **xgb_selected},
    ]).to_csv(outputs_dir / 'threshold_analysis.csv', index=False)
    importance = pd.DataFrame({'feature': list(Xtr.columns), 'importance': model.feature_importances_})
    importance.sort_values('importance', ascending=False).to_csv(outputs_dir / 'feature_importance.csv', index=False)

    split_rows = []
    for name, part in [('train', tr), ('validation', va), ('test', te)]:
        split_rows.append({
            'split': name, 'rows': int(len(part)), 'fraud': int(part.Class.sum()),
            'legitimate': int((part.Class == 0).sum()), 'fraud_rate_pct': float(part.Class.mean() * 100),
            'time_min': float(part.Time.min()), 'time_max': float(part.Time.max()),
        })
    pd.DataFrame(split_rows).to_csv(outputs_dir / 'temporal_split_summary.csv', index=False)
    meta = {
        'feature_columns': list(Xtr.columns), 'data_rows': int(n),
        'split_rows': {'train': len(tr), 'validation': len(va), 'test': len(te)},
        'split_fraud_counts': {'train': int(ytr.sum()), 'validation': int(yva.sum()), 'test': int(yte.sum())},
        'default_threshold': .5, 'validation_cost_threshold': selected_threshold,
        'cost_assumptions': {'false_negative': 20, 'false_positive': 1, 'units': 'illustrative, not currency'},
        'validation_selected_cost_units': float(best_val['cost_units']),
        'validation_selected_metrics': validation_metrics,
        'test_average_precision': float(average_precision_score(yte, pte)),
        'test_roc_auc': float(roc_auc_score(yte, pte)),
        'test_metrics_default_threshold': xgb_default,
        'test_metrics_validation_selected_threshold': xgb_selected,
        'logistic_test_average_precision': float(average_precision_score(yte, p_lr_test)),
        'logistic_test_roc_auc': float(roc_auc_score(yte, p_lr_test)),
        'note': 'Scores are not calibrated probabilities; threshold costs are illustrative.'
    }
    (model_dir / 'model_metadata.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


if __name__ == '__main__':
    main()
