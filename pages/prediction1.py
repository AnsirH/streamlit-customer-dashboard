# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from churn_model import ChurnPredictor

st.set_page_config(page_title="고객 이탈 예측", layout="wide")
st.title("고객 이탈 예측 페이지")

# 1) 고객 정보 입력
st.header("1) 고객 정보 입력")
col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("거래 기간 (개월)", min_value=0, max_value=120, value=12)
    warehouse = st.number_input("창고-집 거리 (km)", min_value=0.0, max_value=100.0, value=10.0)
with col2:
    hour = st.number_input("앱 사용 시간 (시간)", min_value=0.0, max_value=24.0, value=1.0)
    hike = st.number_input("작년 대비 주문 금액 증가율 (%)", min_value=0.0, max_value=200.0, value=10.0)
with col3:
    coupon = st.number_input("쿠폰 사용 횟수", min_value=0, max_value=100, value=2)
    cashback = st.number_input("캐시백 금액 (원)", min_value=0.0, max_value=10000.0, value=150.0)

# 예측 버튼
if st.button("이탈 예측하기"):
    # 입력값을 DataFrame으로 변환
    input_df = pd.DataFrame([{
        "Tenure": tenure,
        "WarehouseToHome": warehouse,
        "HourSpendOnApp": hour,
        "OrderAmountHikeFromlastYear": hike,
        "CouponUsed": coupon,
        "CashbackAmount": cashback
    }])

    # 모델 예측
    predictor = ChurnPredictor()
    _, proba = predictor.predict(input_df)
    prob = float(proba[0])

    # 2) 이탈율 위험도 게이지바
    st.header("2) 이탈율 위험도 게이지바")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%'},
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
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 3) 주요 영향 요인
    st.header("3) 주요 영향 요인")
    fi = predictor.model.feature_importances_
    fi_df = pd.DataFrame({
        'feature': input_df.columns,
        'importance': fi
    }).sort_values('importance', ascending=False)

    fig_bar = go.Figure(go.Bar(
        x=fi_df['feature'],
        y=fi_df['importance']
    ))
    fig_bar.update_layout(xaxis_title="피처", yaxis_title="중요도")
    st.plotly_chart(fig_bar, use_container_width=True)
