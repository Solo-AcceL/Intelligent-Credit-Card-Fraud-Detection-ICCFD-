from pathlib import Path
import sys, json
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fraud_capstone.score import score_frame

st.set_page_config(page_title='Fraud Intelligence',page_icon='🛡️',layout='wide')
st.title('Intelligent Credit Card Fraud Detection')
st.caption('Decision-support demo using anonymized ULB/Kaggle benchmark data. The bundled sample is fraud-enriched for demonstration and does not represent natural prevalence. Model scores are not calibrated probabilities and must not be used for live transaction blocking.')
meta_path=ROOT/'models/model_metadata.json'
meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
pred_path=ROOT/'outputs/test_predictions.csv'
sample_path=ROOT/'data/sample/sample_transactions.csv'
if pred_path.exists():
    preds=pd.read_csv(pred_path)
elif sample_path.exists():
    preds=pd.read_csv(sample_path)
else:
    st.error('No demo predictions found. Run the pipeline or restore data/sample/sample_transactions.csv.'); st.stop()

page=st.sidebar.radio('Navigate',['Monitoring overview','Transaction investigation','Model performance','Power BI exports'])
if page=='Monitoring overview':
    total=len(preds); actual_fraud=int(preds['Class'].sum()) if 'Class' in preds else None
    scorecol='fraud_score' if 'fraud_score' in preds else None
    flagged=int((preds[scorecol]>=.5).sum()) if scorecol else 0
    c1,c2,c3,c4=st.columns(4); c1.metric('Transactions in view',f'{total:,}'); c2.metric('Actual fraud labels (evaluation only)',f'{actual_fraud:,}' if actual_fraud is not None else 'N/A'); c3.metric('Flagged at 0.50',f'{flagged:,}'); c4.metric('Flag rate',f'{flagged/total*100:.2f}%' if total else 'N/A')
    if 'risk_tier' in preds:
        st.subheader('Risk-tier distribution')
        st.bar_chart(preds['risk_tier'].value_counts().reindex(['Low','Moderate','High','Critical']).fillna(0))
    st.subheader('Filter transactions')
    threshold=st.slider('Review threshold',0.01,0.99,0.50,0.01)
    if 'fraud_score' in preds:
        filtered=preds[preds.fraud_score>=threshold].copy()
        st.write(f'{len(filtered):,} rows exceed the selected model-score threshold.')
        show=[c for c in ['Time','Amount','Class','fraud_score','risk_tier','predicted_fraud_0_5','top_contributing_feature'] if c in filtered.columns]
        st.dataframe(filtered[show].sort_values('fraud_score',ascending=False).head(1000),use_container_width=True)
        st.download_button('Download filtered results CSV',filtered.to_csv(index=False).encode(),file_name='filtered_fraud_results.csv',mime='text/csv')
elif page=='Transaction investigation':
    st.subheader('Score one transaction')
    st.write('Choose a row with the complete feature vector. The bundled demonstration sample is fraud-enriched for illustrating the interface; it is not representative of natural prevalence.')
    investigation_data = pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame()
    parquet_path = ROOT/'data/processed/clean_transactions.parquet'
    if investigation_data.empty and parquet_path.exists():
        full = pd.read_parquet(parquet_path)
        pos = full[full.Class == 1].head(25)
        neg = full[full.Class == 0].head(975)
        investigation_data = pd.concat([pos, neg], ignore_index=True)
    required = {'Time', 'Amount', *[f'V{i}' for i in range(1,29)]}
    if investigation_data.empty or not required.issubset(investigation_data.columns):
        st.warning('No full-feature transaction sample is available. Restore data/sample/sample_transactions.csv or run the data pipeline locally.')
    else:
        idx=st.number_input('Sample row index',min_value=0,max_value=max(0,len(investigation_data)-1),value=0,step=1)
        row=investigation_data.iloc[int(idx)]
        if st.button('Analyze selected transaction'):
            threshold=st.slider('Scoring threshold',0.01,0.99,0.50,0.01,key='score_threshold')
            result=score_frame(pd.DataFrame([row]),model_path=ROOT/'models/xgb_fraud_model.json',threshold=threshold).iloc[0]
            cols=st.columns(3); cols[0].metric('Model score',f"{result['fraud_score']:.3f}"); cols[1].metric('Risk tier',result['risk_tier']); cols[2].metric('Flag for review','YES' if result['predicted_fraud'] else 'NO')
            st.info(f"Largest absolute local TreeSHAP contribution: {result['top_contributing_feature']} ({result['top_feature_contribution']:+.4f}). The feature names V1-V28 are anonymized PCA components and do not have direct business labels.")
            if 'Class' in row: st.caption(f"Demo/evaluation label available for this row: {'Fraud' if int(row['Class'])==1 else 'Legitimate'}")
            st.warning('The explanation is a model attribution, not causal proof. Human review is required.')
elif page=='Model performance':
    st.subheader('Held-out chronological evaluation')
    st.json({k:meta[k] for k in ['test_average_precision','test_roc_auc','test_metrics_default_threshold','test_metrics_validation_selected_threshold'] if k in meta})
    fig=ROOT/'outputs/figures/precision_recall_curve.png'
    if fig.exists(): st.image(str(fig),use_container_width=True)
    fig=ROOT/'outputs/figures/confusion_matrix.png'
    if fig.exists(): st.image(str(fig),width=600)
    st.markdown('**Interpretation:** evaluate precision and recall together. A high accuracy score can conceal missed fraud because the positive class is rare.')
elif page=='Power BI exports':
    st.subheader('Power BI-ready CSV exports')
    for file in ['powerbi_kpis.csv','powerbi_time_buckets.csv','powerbi_high_risk_transactions.csv']:
        path=ROOT/'outputs'/file
        if path.exists():
            st.write(file, f'({path.stat().st_size/1024:.1f} KB)')
            st.download_button(f'Download {file}',path.read_bytes(),file_name=file,mime='text/csv',key=file)
    st.caption('In Power BI Desktop, use Get Data → Text/CSV, select the exports, then build slicers for elapsed time buckets and risk tiers.')
st.sidebar.caption('Scope: offline benchmark and human-in-the-loop triage. Not a production authorization system.')
