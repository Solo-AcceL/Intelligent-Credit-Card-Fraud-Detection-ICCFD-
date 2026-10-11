import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pandas as pd
import numpy as np
import pytest
from fraud_capstone.features import engineer_features, FEATURE_COLUMNS

def test_feature_engineering_shape_and_columns():
    row={'Time':3600.0,'Amount':12.5,**{f'V{i}':0.1*i for i in range(1,29)}}
    X=engineer_features(pd.DataFrame([row]))
    assert list(X.columns)==FEATURE_COLUMNS
    assert X.shape==(1,32)
    assert np.isclose(X.loc[0,'log_amount'],np.log1p(12.5))

def test_negative_amount_rejected():
    row={'Time':1.0,'Amount':-1.0,**{f'V{i}':0.0 for i in range(1,29)}}
    with pytest.raises(ValueError): engineer_features(pd.DataFrame([row]))

def test_missing_columns_rejected():
    with pytest.raises(ValueError): engineer_features(pd.DataFrame([{'Time':1,'Amount':2}]))
