import streamlit as st
from components.header import show_header
from components.animations import add_page_transition
from utils.model_predictor import ModelPredictor
from .customer_dashboard import show_customer_churn_analysis

def show():
    # 애니메이션 적용
    add_page_transition()
    show_header()
    
    # 고객 대시보드 표시
    show_customer_churn_analysis()
    # 데이터 생성 및 예측
    try:
        df = ModelPredictor.predict_churn()
        ModelPredictor.display_customer_analysis(df)
    except Exception as e:
        st.error(str(e))