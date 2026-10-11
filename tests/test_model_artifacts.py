import json
from pathlib import Path

def test_metadata_exists_and_has_metrics():
    root=Path(__file__).resolve().parents[1]
    meta=json.loads((root/'models/model_metadata.json').read_text())
    assert 0 <= meta['test_average_precision'] <= 1
    assert 0 <= meta['test_roc_auc'] <= 1
    assert meta['test_metrics_default_threshold']['tp'] >= 0
