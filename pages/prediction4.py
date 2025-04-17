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

    # 임의 ID 생성
    df["RandomID"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]

    # 모델 로드 및 예측
    model = load_xgboost_model2()
    predictor = ChurnPredictor2(external_model=model)

    # 필요한 칼럼이 모두 있는지 확인 후 처리
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
    pred_df = df[required_features + ["RandomID"]].copy()
    df_encoded = pd.get_dummies(pred_df.drop(columns="RandomID"))

    # 모델이 요구하는 모든 컬럼 맞춤
    model_features = predictor.model.get_booster().feature_names
    for col in model_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[model_features]

    _, y_proba = predictor.predict(df_encoded)
    df["ChurnProbability"] = y_proba

    # 위험군 분류
    def risk_group(p):
        if p >= 0.9:
            return "초고위험군"
        elif p >= 0.7:
            return "고위험군"
        elif p >= 0.5:
            return "주의단계"
        else:
            return "관찰단계"

    df["RiskGroup"] = df["ChurnProbability"].apply(risk_group)

    # 📌 군별 ID 나열
    st.subheader("📌 군별 고객 ID")
    for group in ["초고위험군", "고위험군", "주의단계", "관찰단계"]:
        st.markdown(f"**{group}**")
        group_ids = df[df["RiskGroup"] == group]["RandomID"].tolist()
        st.write(group_ids)

    # 🔍 특정 고객 선택
    st.subheader("👤 고객 ID 선택")
    selected_id = st.selectbox("분석할 고객 ID 선택", df["RandomID"].unique())
    selected_row = df[df["RandomID"] == selected_id].iloc[0]

    st.markdown("---")
    st.subheader("📈 이탈 확률 게이지")
    prob_pct = float(selected_row["ChurnProbability"] * 100)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={'suffix': '%'},
        title={"text": "이탈 가능성 (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': 'darkblue'},
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 50], 'color': 'yellowgreen'},
                {'range': [50, 70], 'color': 'yellow'},
                {'range': [70, 90], 'color': 'orange'},
                {'range': [90, 100], 'color': 'red'}
            ]
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    # 📊 입력값 수정 인터페이스
    st.subheader("⚙ 고객 데이터 튜닝")
    modified_inputs = {}
    for col in required_features:
        val = selected_row[col]
        if isinstance(val, (int, float)):
            modified_inputs[col] = st.number_input(col, value=float(val))
        else:
            modified_inputs[col] = st.text_input(col, value=str(val))

    # 🔁 변동 예측
    if st.button("변동 예측하기"):
        df_mod = pd.DataFrame([modified_inputs])
        df_encoded2 = pd.get_dummies(df_mod)
        for col in model_features:
            if col not in df_encoded2.columns:
                df_encoded2[col] = 0
        df_encoded2 = df_encoded2[model_features]

        _, new_proba = predictor.predict(df_encoded2)
        new_pct = float(new_proba[0]) * 100

        st.success(f"새로운 예측 이탈 확률: {new_pct:.2f}%")

        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=new_pct,
            number={'suffix': '%'},
            title={"text": "이탈 가능성 (변동 후)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': 'darkblue'},
                'steps': [
                    {'range': [0, 30], 'color': 'green'},
                    {'range': [30, 50], 'color': 'yellowgreen'},
                    {'range': [50, 70], 'color': 'yellow'},
                    {'range': [70, 90], 'color': 'orange'},
                    {'range': [90, 100], 'color': 'red'}
                ]
            }
        ))
        st.plotly_chart(fig2, use_container_width=True)

        # 🎯 중요도
        st.subheader("🎯 예측에 영향을 준 주요 요인")
        processed = predictor._preprocess_data(df_mod)
        _ = predictor._compute_feature_importance(processed)
        importance_dict = predictor.get_feature_importance()
        fi_df = pd.DataFrame(importance_dict.items(), columns=["Feature", "Importance"])

        top5 = fi_df.sort_values("Importance", ascending=False).head(5)
        bottom5 = fi_df.sort_values("Importance", ascending=True).head(5)

        st.markdown("**상위 5개 변수**")
        st.dataframe(top5)
        st.markdown("**하위 5개 변수**")
        st.dataframe(bottom5)
