"""Optional FastAPI inference endpoint. Scores are decision-support only."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import pandas as pd
from .features import INPUT_COLUMNS
from .score import score_frame

app=FastAPI(title='Credit Card Fraud Risk API',version='1.0.0',description='Decision-support scoring demo; not production payment authorization.')
class Transaction(BaseModel):
    Time: float = Field(ge=0)
    Amount: float = Field(ge=0)
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float; V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float; V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float; V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
@app.get('/health')
def health(): return {'status':'ok','mode':'demo'}
@app.post('/predict')
def predict(txn: Transaction, threshold: float=.5):
    try:
        frame=pd.DataFrame([txn.model_dump()])[INPUT_COLUMNS]
        out=score_frame(frame,threshold=threshold).iloc[0]
        score=float(out['fraud_score'])
        tier='Low' if score<.1 else 'Moderate' if score<.5 else 'High' if score<.8 else 'Critical'
        return {'fraud_score':score,'score_is_calibrated_probability':False,'threshold':threshold,'flag_for_review':bool(score>=threshold),'risk_tier':tier,'top_contributing_feature':str(out['top_contributing_feature']),'top_feature_contribution':float(out['top_feature_contribution']),'decision_note':'Decision support only. Do not auto-block real transactions without validation, calibration, controls and human oversight.'}
    except Exception as exc:
        raise HTTPException(status_code=400,detail=str(exc))
