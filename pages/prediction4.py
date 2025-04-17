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

# 위험군 분류 함수
def classify_risk(prob):
    if prob >= 0.9:
        return "초고위험군"
    elif prob >= 0.7:
        return "고위험군"
    elif prob >= 0.5:
        return "주의단계"
    elif prob >= 0.3:
        return "관찰단계"
    else:
        return "분류제외"

# 인코딩된 범주형을 복원하고 한글 컬럼 적용
def reverse_one_hot_columns(df_encoded):
    reverse_map = {
        "PreferredLoginDevice": "선호 로그인 기기",
        "PreferredPaymentMode": "선호 결제 방식",
        "Gender": "성별",
        "PreferedOrderCat": "선호 주문 카테고리",
        "MaritalStatus": "결혼 여부"
    }

    recovered = pd.DataFrame()

    for prefix_en, label_kr in reverse_map.items():
        matched = df_encoded.filter(like=prefix_en + "_")
        recovered[label_kr] = matched.idxmax(axis=1).str.replace(prefix_en + "_", "")

    numeric_features = [
        'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
        'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
        'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
        'DaySinceLastOrder', 'CashbackAmount'
    ]
    numeric_labels = [
        "이용 기간", "거주 도시 등급", "창고-집 거리", "앱 사용 시간",
        "등록된 기기 수", "만족도 점수", "배송지 등록 수",
        "불만 제기 여부", "주문금액 상승률", "쿠폰 사용 횟수", "주문 횟수",
        "마지막 주문 후 경과일", "캐시백 금액"
    ]
    for en, kr in zip(numeric_features, numeric_labels):
        if en in df_encoded.columns:
            recovered[kr] = df_encoded[en]

    return recovered[numeric_labels + list(reverse_map.values())]

# Streamlit 앱 시작
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
    model_features = predictor.model.get_booster().feature_names

    df_encoded = pd.get_dummies(df)
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_features]

    _, y_proba = predictor.predict(df_encoded)
    df["이탈확률"] = y_proba
    df["위험군"] = df["이탈확률"].apply(classify_risk)

    # 위험군별 고객 ID (상위 10개씩)
    st.subheader("📌 위험군별 고객 ID (상위 10개)")
    for group in ["초고위험군", "고위험군", "주의단계", "관찰단계"]:
        st.markdown(f"**{group}**")
        top_ids = df[df["위험군"] == group].nlargest(10, "이탈확률")["고객ID"].tolist()
        st.write(top_ids)

    st.success("✅ 고객 ID 부여 및 군별 분류까지 완료되었습니다.")