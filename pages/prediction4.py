import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import uuid
from pathlib import Path
import sys

# 경로 설정
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.churn_model import load_xgboost_model2, ChurnPredictor2

st.set_page_config(page_title="고객 이탈 예측 - 데이터 분석", layout="wide")
st.title("📥 고객 데이터 분석 및 튜닝 시스템")

# --------------------------
# 1️⃣ CSV 업로드 및 예측 실행
# --------------------------
st.subheader("📁 CSV 고객 데이터 업로드")
file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success(f"✅ {df.shape[0]}명의 고객 데이터가 로드되었습니다.")

    # ID 생성
    df["CustomerID"] = [f"CUST-{str(uuid.uuid4())[:8]}" for _ in range(len(df))]

    # 모델 로드 및 예측
    model = load_xgboost_model2()
    predictor = ChurnPredictor2(external_model=model)

    # 누락 피처 보완
    required_features = [
        'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
        'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
        'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
        'DaySinceLastOrder', 'CashbackAmount',
        'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone',
        'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery',
        'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
        'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI',
        'Gender_Male', 'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
        'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone',
        'MaritalStatus_Married', 'MaritalStatus_Single'
    ]

    # 원-핫 인코딩
    cat_cols = ["PreferredLoginDevice", "PreferredPaymentMode", "Gender", "PreferedOrderCat", "MaritalStatus"]
    df_encoded = pd.get_dummies(df, columns=cat_cols)
    for col in required_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[required_features]

    # 예측
    _, probs = predictor.predict(df_encoded)
    df["ChurnProbability"] = probs * 100

    # 위험군 분류
    def classify_risk(prob):
        if prob >= 90:
            return "초고위험군"
        elif prob >= 70:
            return "고위험군"
        elif prob >= 50:
            return "주의단계"
        else:
            return "관찰단계"

    df["RiskLevel"] = df["ChurnProbability"].apply(classify_risk)

    # --------------------------
    # 2️⃣ 군별 고객 ID 시각화
    # --------------------------
    st.subheader("📊 고객 위험도 군 분포")
    risk_groups = ["초고위험군", "고위험군", "주의단계", "관찰단계"]
    for level in risk_groups:
        st.markdown(f"#### 🔸 {level}")
        ids = df[df["RiskLevel"] == level]["CustomerID"].tolist()
        id_cols = st.columns(min(len(ids), 5))
        for i, cid in enumerate(ids):
            id_cols[i % 5].button(cid, key=f"{level}_{cid}")

    # 이후 단계: ID 클릭 시 값 추출 → 18개 칼럼 채우기 → 수정 및 재예측 → SHAP 표시 등
else:
    st.info("👆 왼쪽 상단에서 CSV 파일을 먼저 업로드하세요.")

    