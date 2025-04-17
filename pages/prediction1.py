# pages/prediction1.py

import sys
from pathlib import Path

# 프로젝트 루트 경로 등록
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 로컬 모듈 임포트
from models.churn_model import ChurnPredictor, load_xgboost_model2

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="고객 이탈 예측", layout="wide")
st.title("고객 이탈 예측 페이지")

if st.button("🧠 이탈 예측 실행"):
    input_df = pd.DataFrame([{
        "Tenure": tenure,
        "WarehouseToHome": warehouse,
        "HourSpendOnApp": hour,
        "OrderAmountHikeFromlastYear": hike,
        "CouponUsed": coupon,
        "CashbackAmount": cashback
    }])

    # 1. 모델 수동 로드 및 주입
    model = load_xgboost_model2()
    predictor = ChurnPredictor(model_path=None)
    predictor.model = model

    # 2. 예측
    pred, proba = predictor.predict(input_df)
    prob_pct = float(proba[0]) * 100

    # 3. 게이지 차트
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

    # 4. 중요도 재계산 후 시각화
    _ = predictor._compute_feature_importance(predictor._preprocess_data(input_df))
    fi = predictor.get_feature_importance()

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
