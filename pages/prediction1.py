import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# 경로 설정
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.churn_model import load_xgboost_model2, ChurnPredictor2

st.set_page_config(page_title="고객 이탈 예측", layout="wide")
st.title("📊 고객 이탈 예측 시스템")

# --------------------------
# 1️⃣ UI 입력 섹션 (총 18개)
# --------------------------
st.subheader("1️⃣ 고객 데이터 입력")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("이용 기간 (개월)", min_value=0, value=12)
    city_tier = st.selectbox("거주 도시 등급 (1~3)", [1, 2, 3], index=1)
    warehouse_dist = st.number_input("창고-집 거리 (km)", min_value=0.0, value=20.0)
    app_hour = st.number_input("앱 사용 시간 (시간)", min_value=0.0, value=2.5)
    num_devices = st.number_input("등록된 기기 수", min_value=0, value=2)

with col2:
    satisfaction = st.slider("만족도 점수 (1~5)", 1, 5, 3)
    num_address = st.number_input("배송지 등록 수", min_value=0, value=1)
    complain = st.selectbox("불만 제기 여부", ["예", "아니오"])
    order_hike = st.number_input("주문금액 상승률 (%)", value=10.0)
    coupon_used = st.number_input("쿠폰 사용 횟수", value=2)

with col3:
    orders = st.number_input("주문 횟수", value=8)
    last_order_days = st.number_input("마지막 주문 후 경과일", value=10)
    cashback = st.number_input("캐시백 금액", value=150)

    # ✅ 범주형 변수 5개
    login_device = st.selectbox("선호 로그인 기기", ["Mobile Phone", "Phone"])
    payment_mode = st.selectbox("선호 결제 방식", [
        "Credit Card", "Debit Card", "Cash on Delivery", "COD",
        "E wallet", "UPI"
    ])
    gender = st.selectbox("성별", ["Male", "Female"])
    order_cat = st.selectbox("선호 주문 카테고리", [
        "Mobile", "Mobile Phone", "Laptop & Accessory", "Grocery"
    ])
    marital = st.selectbox("결혼 여부", ["Single", "Married"])

# --------------------------
# 2️⃣ 예측 버튼 누르면 실행
# --------------------------
if st.button("🧠 이탈 예측하기"):

    # 기본 수치형 + 범주형 코드화 전
    raw_input = {
        "Tenure": tenure,
        "CityTier": city_tier,
        "WarehouseToHome": warehouse_dist,
        "HourSpendOnApp": app_hour,
        "NumberOfDeviceRegistered": num_devices,
        "SatisfactionScore": satisfaction,
        "NumberOfAddress": num_address,
        "Complain": 1 if complain == "예" else 0,
        "OrderAmountHikeFromlastYear": order_hike,
        "CouponUsed": coupon_used,
        "OrderCount": orders,
        "DaySinceLastOrder": last_order_days,
        "CashbackAmount": cashback,
        "PreferredLoginDevice": login_device,
        "PreferredPaymentMode": payment_mode,
        "Gender": gender,
        "PreferedOrderCat": order_cat,
        "MaritalStatus": marital
    }

    df_input = pd.DataFrame([raw_input])

    # ✅ 원-핫 인코딩 대상
    one_hot_cols = [
        "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
        "PreferedOrderCat", "MaritalStatus"
    ]
    df_encoded = pd.get_dummies(df_input, columns=one_hot_cols)

    # ✅ 모델 요구 피처 목록
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

    # 누락된 피처는 0으로 채움
    for col in required_features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # 순서 맞춤
    df_encoded = df_encoded[required_features]

    try:
        model = load_xgboost_model2()
        predictor = ChurnPredictor2(external_model=model)
        y_pred, y_proba = predictor.predict(df_encoded)
        prob_pct = float(y_proba[0]) * 100

        # 📈 게이지 차트
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

        # 📊 주요 변수 영향 시각화
        processed = predictor._preprocess_data(df_encoded)
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
        st.error(f"❌ 예측 실패: {str(e)}")
