import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import uuid
import sys

# 경로 설정
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.churn_model import load_xgboost_model2, ChurnPredictor2

# 함수: 위험군 분류
def classify_risk(prob):
    if prob >= 0.9:
        return "초고위험군"
    elif prob >= 0.7:
        return "고위험군"
    elif prob >= 0.5:
        return "주의단계"
    else:
        return "관찰단계"

st.set_page_config(page_title="고객 이탈 예측 시스템", layout="wide")
st.title("고객 이탈 예측 시스템")

st.subheader("📁 데이터 입력")
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ {df.shape[0]}명의 고객 데이터가 로드되었습니다.")

    df["고객ID"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]

    model = load_xgboost_model2()
    predictor = ChurnPredictor2(external_model=model)

    required_features = [
        'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
        'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
        'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
        'DaySinceLastOrder', 'CashbackAmount',
        'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
        'PreferedOrderCat', 'MaritalStatus'
    ]

    missing_cols = [col for col in required_features if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ 다음 컬럼이 누락되어 분석이 불가합니다: {missing_cols}")
        st.stop()

    # 예측 수행
    df_encoded = pd.get_dummies(df[required_features])
    model_features = predictor.model.get_booster().feature_names
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_features]

    _, y_proba = predictor.predict(df_encoded)
    df["이탈확률"] = y_proba
    df["위험군"] = df["이탈확률"].apply(classify_risk)

    # 위험군별 고객 ID 표시
    st.subheader("📌 위험군별 고객 ID (상위 10개)")
    for group in ["초고위험군", "고위험군", "주의단계", "관찰단계"]:
        st.markdown(f"**{group}**")
        top_ids = df[df["위험군"] == group].nlargest(10, "이탈확률")["고객ID"].tolist()
        st.write(top_ids)

    # 다음 단계: ID 선택 → 게이지 표시 → 데이터 튜닝 → 변동 예측 등
