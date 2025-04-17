# pages/prediction1.py

import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.churn_model import ChurnPredictor

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="고객 이탈 예측", layout="wide")
st.title("고객 이탈 예측 페이지")

# 1) 고객 정보 입력
st.header("1) 고객 정보 입력")
c1, c2, c3 = st.columns(3)
with c1:
    tenure = st.number_input("거래 기간 (개월)", 0, 120, 12)
    warehouse = st.number_input("창고-집 거리 (km)", 0.0, 100.0, 10.0)
with c2:
    hour = st.number_input("앱 사용 시간 (시간)", 0.0, 24.0, 1.0)
    hike = st.number_input("작년 대비 주문 금액 증가율 (%)", 0.0, 200.0, 10.0)
with c3:
    coupon = st.number_input("쿠폰 사용 횟수", 0, 100, 2)
    cashback = st.number_input("캐시백 금액 (원)", 0.0, 10000.0, 150.0)

if st.button("이탈 예측하기"):
    # 입력값 DataFrame 생성
    input_df = pd.DataFrame([{
        "Tenure": tenure,
        "WarehouseToHome": warehouse,
        "HourSpendOnApp": hour,
        "OrderAmountHikeFromlastYear": hike,
        "CouponUsed": coupon,
        "CashbackAmount": cashback
    }])

    # 예측 수행
    predictor = ChurnPredictor()
    pred, proba = predictor.predict(input_df)
    prob_pct = float(proba[0]) * 100

    # 2) 이탈율 위험도 게이지바
    st.header("2) 이탈율 위험도 게이지바")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'},
            ],
            'bar': {'color': 'darkblue'}
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 3) 주요 영향 요인
    st.header("3) 주요 영향 요인")
    fi = predictor.get_feature_importance()
    if isinstance(fi, dict):
        items = fi.items()
    else:
        items = zip(input_df.columns, fi)

    fi_df = pd.DataFrame(items, columns=['feature', 'importance']) \
                 .sort_values('importance', ascending=False)
    fig_bar = go.Figure(go.Bar(x=fi_df['feature'], y=fi_df['importance']))
    fig_bar.update_layout(xaxis_title="피처", yaxis_title="중요도")
    st.plotly_chart(fig_bar, use_container_width=True)
