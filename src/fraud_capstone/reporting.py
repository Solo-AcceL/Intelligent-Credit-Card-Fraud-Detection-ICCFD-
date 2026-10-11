"""Create reproducible KPI exports and diagnostic figures from pipeline outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches='tight')
    plt.close(fig)


def generate_reporting(clean: pd.DataFrame, outputs_dir='outputs', model_dir='models') -> None:
    out = Path(outputs_dir); figdir = out / 'figures'; figdir.mkdir(parents=True, exist_ok=True)
    clean = clean.sort_values('Time', kind='stable').reset_index(drop=True)
    pred_path = out / 'test_predictions.csv'
    if not pred_path.exists():
        raise FileNotFoundError('test_predictions.csv missing: run training before reporting')
    pred = pd.read_csv(pred_path)
    y = pred['Class'].astype(int).to_numpy(); score = pred['fraud_score'].astype(float).to_numpy()
    flagged = pred['prediction_0_5'].astype(int).to_numpy() if 'prediction_0_5' in pred else (score >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, flagged, labels=[0, 1]).ravel()

    # Class imbalance after exact-row deduplication.
    counts = clean['Class'].value_counts().reindex([0, 1], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    bars = ax.bar(['Legitimate', 'Fraud'], counts.values, color=['#26748a', '#d95757'])
    ax.set_yscale('log'); ax.set_ylabel('Transactions (log scale)'); ax.set_title('Extreme class imbalance after duplicate removal')
    for bar, val in zip(bars, counts.values): ax.text(bar.get_x()+bar.get_width()/2, val*1.15, f'{val:,}\n({val/len(clean)*100:.3f}%)', ha='center', va='bottom', fontsize=9)
    _save(fig, figdir / 'class_imbalance.png')

    # Long-tailed Amount view; cap samples per class for readable, deterministic plotting.
    rng = np.random.default_rng(42)
    samples = []
    for label in [0, 1]:
        group = clean.loc[clean.Class == label, 'Amount'].clip(lower=0).to_numpy()
        if len(group) > 15000: group = rng.choice(group, 15000, replace=False)
        samples.append(np.log1p(group))
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    violin = ax.violinplot(samples, showmedians=True, showextrema=False)
    for body, color in zip(violin['bodies'], ['#26748a', '#d95757']): body.set_facecolor(color); body.set_alpha(.8)
    ax.set_xticks([1, 2], ['Legitimate', 'Fraud']); ax.set_ylabel('log1p(Amount)'); ax.set_title('Transaction amount distribution by label')
    _save(fig, figdir / 'amount_distribution.png')

    # Six-hour buckets based on elapsed seconds in the source, not clock time.
    time = clean.copy(); time['elapsed_6h_bucket'] = np.floor(time.Time / 21600).astype(int)
    tb = time.groupby('elapsed_6h_bucket', as_index=False).agg(transactions=('Class', 'size'), fraud=('Class', 'sum'), amount_total=('Amount', 'sum'))
    tb['fraud_rate_pct'] = tb['fraud'] / tb['transactions'] * 100
    tb.to_csv(out / 'powerbi_time_buckets.csv', index=False)
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.plot(tb.elapsed_6h_bucket, tb.fraud_rate_pct, marker='o', color='#26748a')
    ax.set_xlabel('Six-hour elapsed-time bucket'); ax.set_ylabel('Fraud rate (%)'); ax.set_title('Fraud prevalence across elapsed-time buckets')
    _save(fig, figdir / 'fraud_rate_time.png')

    precision, recall, _ = precision_recall_curve(y, score)
    prevalence = float(y.mean())
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(recall, precision, color='#26748a', label='XGBoost')
    ax.axhline(prevalence, linestyle='--', color='#d95757', label=f'Prevalence baseline ({prevalence:.3%})')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('Precision–recall curve on chronological test set'); ax.legend(loc='best')
    _save(fig, figdir / 'precision_recall_curve.png')

    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(5.8, 4.5)); im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1], ['Pred. legitimate', 'Pred. fraud']); ax.set_yticks([0, 1], ['Actual legitimate', 'Actual fraud'])
    ax.set_title('XGBoost confusion matrix at score 0.50')
    for (i, j), val in np.ndenumerate(cm): ax.text(j, i, f'{val:,}', ha='center', va='center', color='white' if val > cm.max()*.55 else 'black', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=.045); _save(fig, figdir / 'confusion_matrix.png')

    importance_path = out / 'feature_importance.csv'
    if importance_path.exists():
        imp = pd.read_csv(importance_path).sort_values('importance', ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(7.3, 4.4)); ax.barh(imp.feature, imp.importance, color='#26748a')
        ax.set_xlabel('Relative model importance'); ax.set_title('XGBoost feature importance (not causal)')
        _save(fig, figdir / 'feature_importance.png')

    thr_path = out / 'threshold_analysis_validation.csv'
    if thr_path.exists():
        thr = pd.read_csv(thr_path)
        fig, ax = plt.subplots(figsize=(7.2, 4.0)); ax.plot(thr.threshold, thr.cost_units, label='Validation cost units', color='#26748a')
        if len(thr):
            best = thr.loc[thr.cost_units.idxmin()]
            ax.axvline(best.threshold, linestyle='--', color='#d95757', label=f'Validation minimum ({best.threshold:.3f})')
        ax.set_xlabel('Model score threshold'); ax.set_ylabel('Illustrative cost units'); ax.set_title('Threshold selection on validation only'); ax.legend()
        _save(fig, figdir / 'threshold_cost_curve.png')

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.hist(score[y == 0], bins=50, density=True, alpha=.65, label='Legitimate', color='#26748a')
    if np.any(y == 1): ax.hist(score[y == 1], bins=25, density=True, alpha=.65, label='Fraud', color='#d95757')
    ax.set_xlabel('Model score (not calibrated probability)'); ax.set_ylabel('Density'); ax.set_title('Score distribution by observed label'); ax.legend()
    _save(fig, figdir / 'score_distribution.png')

    # Power BI-ready transaction-level alert extract and concise KPI table.
    selected = pred.loc[score >= .5].copy()
    selected['risk_tier'] = np.select([selected.fraud_score < .1, selected.fraud_score < .5, selected.fraud_score < .8], ['Low', 'Moderate', 'High'], default='Critical')
    keep = [c for c in ['Time', 'Amount', 'Class', 'fraud_score', 'prediction_0_5', 'risk_tier', 'top_contributing_feature', 'top_feature_contribution'] if c in selected.columns]
    selected[keep].sort_values('fraud_score', ascending=False).to_csv(out / 'powerbi_high_risk_transactions.csv', index=False)
    metadata_path = Path(model_dir) / 'model_metadata.json'
    meta = {}
    if metadata_path.exists(): meta = json.loads(metadata_path.read_text())
    test_default = meta.get('test_metrics_default_threshold', {})
    kpis = {
        'test_transactions': len(pred), 'test_fraud_labels': int(y.sum()), 'test_fraud_rate_pct': prevalence * 100,
        'flagged_at_0_50': int(flagged.sum()), 'review_rate_pct': float(flagged.mean() * 100),
        'true_positives': int(tp), 'false_positives': int(fp), 'false_negatives': int(fn), 'true_negatives': int(tn),
        'average_precision': float(meta.get('test_average_precision', np.nan)), 'roc_auc': float(meta.get('test_roc_auc', np.nan)),
        'test_cost_units_fn20_fp1': float(20 * fn + fp),
    }
    pd.DataFrame([kpis]).to_csv(out / 'powerbi_kpis.csv', index=False)
