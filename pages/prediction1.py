# pages/prediction1.py

import sys
from pathlib import Path

# 1. 경로 설정: 루트 디렉토리 등록
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 2. 모델 관련 함수 및 클래스 불러오기
from models.churn_model import load_xgboost_model2, ChurnPredictor

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 3. 페이지 UI 설정
st.set_page_config(page_title="고객 이탈 예측", layout="wide")
st.title("📊 고객 이탈 예측 대시보드")

# 4. 입력 섹션
st.header("1️⃣ 고객 정보 입력")

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.number_input("거래 기간 (개월)", 0, 120, 12)
    warehouse = st.number_input("창고-집 거리 (km)", 0.0, 100.0, 10.0)
with col2:
    hour = st.number_input("앱 사용 시간 (시간)", 0.0, 24.0, 1.0)
    hike = st.number_input("작년 대비 주문 금액 증가율 (%)", 0.0, 200.0, 10.0)
with col3:
    coupon = st.number_input("쿠폰 사용 횟수", 0, 100, 2)
    cashback = st.number_input("캐시백 금액 (원)", 0.0, 10000.0, 150.0)

# 5. 예측 버튼
if st.button("🧠 이탈 예측 실행"):
    # 입력 데이터프레임 생성
    input_df = pd.DataFrame([{
        "Tenure": tenure,
        "WarehouseToHome": warehouse,
        "HourSpendOnApp": hour,
        "OrderAmountHikeFromlastYear": hike,
        "CouponUsed": coupon,
        "CashbackAmount": cashback
    }])

    try:
        # 6. 모델 로드 및 예측 수행
        model = load_xgboost_model2()
        predictor = ChurnPredictor(model_path=None)
        predictor.model = model  # 수동 주입

        pred, proba = predictor.predict(input_df)
        prob_pct = float(proba[0]) * 100

        # 7. 게이지 차트 시각화
        st.header("2️⃣ 이탈율 위험도 게이지")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            number={'suffix': '%'},
            title={"text": "예상 이탈 확률"},
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

        # 8. 피처 중요도 재계산 (입력 기반)
        processed_input = predictor._preprocess_data(input_df)
        _ = predictor._compute_feature_importance(processed_input)
        fi = predictor.get_feature_importance()

        # 9. 바 차트 시각화
        st.header("3️⃣ 주요 영향 요인")
        if isinstance(fi, dict):
            items = fi.items()
        else:
            items = zip(input_df.columns, fi)

        fi_df = pd.DataFrame(items, columns=["feature", "importance"]) \
                   .sort_values("importance", ascending=False)

        fig_bar = go.Figure(go.Bar(
            x=fi_df["feature"],
            y=fi_df["importance"]
        ))
        fig_bar.update_layout(xaxis_title="입력 변수", yaxis_title="중요도")
        st.plotly_chart(fig_bar, use_container_width=True)

    except Exception as e:
        st.error(f"❌ 예측 중 오류 발생: {str(e)}")
