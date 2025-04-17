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

st.set_page_config(page_title="고객 이탈 예측 시스템", layout="wide")
st.title("고객 이탈 예측 시스템")

st.subheader("📁 데이터 입력")
uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ {df.shape[0]}명의 고객 데이터가 로드되었습니다.")

    model = load_xgboost_model2()
    predictor = ChurnPredictor2(external_model=model)

    # 모델 입력 피처 기준
    model_features = predictor.model.get_booster().feature_names

    # One-hot 인코딩되어 있는 데이터를 복원하는 함수
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

    # 모델 입력에 필요한 컬럼만 추출 후 인코딩
    df_input = df.copy()
    df_encoded = pd.get_dummies(df_input)

    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_features]

    # 복원 및 한글 컬럼 적용
    df_recovered = reverse_one_hot_columns(df_encoded)
    st.subheader("🔍 복원된 고객 데이터 (한글 컬럼)")
    st.dataframe(df_recovered.head())