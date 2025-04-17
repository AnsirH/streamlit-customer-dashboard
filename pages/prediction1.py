import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# 루트 경로 설정
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.churn_model import load_xgboost_model2, ChurnPredictor2

st.set_page_config(page_title="고객 이탈 예측", layout="wide")
st.title("📊 고객 이탈 예측 시스템")

st.subheader("1️⃣ 필수 입력 필드")
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("거래 기간 (개월)", min_value=0, value=12)
    satisfaction = st.slider("만족도 점수 (1~5)", 1, 5, 3)

with col2:
    hour = st.number_input("앱 사용 시간 (시간)", min_value=0.0, value=3.0)
    orders = st.number_input("주문 횟수", min_value=0, value=10)

with col3:
    last_order_days = st.number_input("마지막 주문 후 경과일", min_value=0, value=15)
    complain = st.selectbox("불만 제기 여부", ["아니오", "예"])

if st.button("🧠 이탈 예측하기"):
    # 입력값 초기 6개
    input_df = pd.DataFrame([{
        "Tenure": tenure,
        "HourSpendOnApp": hour,
        "SatisfactionScore": satisfaction,
        "OrderCount": orders,
        "DaySinceLastOrder": last_order_days,
        "Complain": 1 if complain == "예" else 0
    }])

    # ✅ 모델이 요구하는 전체 피처 목록
    required_features = [
        'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
        'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
        'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
        'DaySinceLastOrder', 'CashbackAmount',
        'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone',
        'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery',
        'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
        'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI',
        'Gender_Male',
        'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
        'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone',
        'MaritalStatus_Married', 'MaritalStatus_Single'
    ]

    # ✅ 누락된 피처 0으로 채우기
    for col in required_features:
        if col not in input_df.columns:
            input_df[col] = 0

    # ✅ 컬럼 순서 맞추기
    input_df = input_df[required_features]

    try:
        model = load_xgboost_model2()
        predictor = ChurnPredictor2(external_model=model)

        # 예측 수행
        y_pred, y_proba = predictor.predict(input_df)
        prob_pct = float(y_proba[0]) * 100

        # ✅ 디버깅용 출력
        st.subheader("✅ [디버깅] 예측 상태 확인")
        st.write("🔍 모델 타입:", type(predictor.model))
        st.write("🔍 입력 컬럼 수:", input_df.shape[1])
        st.write("🔍 입력 데이터프레임:", input_df)
        st.write("🔍 예측 확률 결과:", y_proba)

        # 📈 게이지 시각화
        st.header("2️⃣ 이탈 확률 예측 결과")
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
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'red'}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        # 🔍 중요도 계산 및 시각화
        processed = predictor._preprocess_data(input_df)
        _ = predictor._compute_feature_importance(processed)
        fi = predictor.get_feature_importance()

        st.header("3️⃣ 예측에 영향을 준 주요 요인")
        fi_df = pd.DataFrame(fi.items(), columns=["Feature", "Importance"]) \
                 .sort_values("Importance", ascending=False)

        fig_bar = go.Figure(go.Bar(
            x=fi_df["Feature"],
            y=fi_df["Importance"]
        ))
        fig_bar.update_layout(xaxis_title="입력 변수", yaxis_title="중요도")
        st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 예측 오류 발생: {str(e)}")
