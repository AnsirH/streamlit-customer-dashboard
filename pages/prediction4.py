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

    # 고객 ID 선택 및 입력 UI
    st.header("4️⃣ 고객 ID 기반 시뮬레이션")
    selected_id = st.selectbox("분석할 고객ID 선택", df.index.astype(str))
    selected_row = df.loc[int(selected_id)]

    st.subheader("🛠 고객 데이터 튜닝")
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.number_input("이용 기간 (개월)", value=int(selected_row["Tenure"]))
        hour = st.number_input("앱 사용 시간", value=float(selected_row["HourSpendOnApp"]))
    with col2:
        satisfaction = st.slider("만족도 점수 (1~5)", 1, 5, int(selected_row["SatisfactionScore"]))
        order_count = st.number_input("주문 횟수", value=int(selected_row["OrderCount"]))
    with col3:
        last_order = st.number_input("마지막 주문 후 경과일", value=int(selected_row["DaySinceLastOrder"]))
        complain = st.selectbox("불만 제기 여부", ["아니오", "예"], index=int(selected_row["Complain"]))

    col4, col5, col6 = st.columns(3)
    with col4:
        gender = st.selectbox("성별", ["Male", "Female"], index=0 if selected_row["Gender"] == "Male" else 1)
        marital = st.selectbox("결혼 여부", ["Single", "Married"], index=0 if selected_row["MaritalStatus"] == "Single" else 1)
    with col5:
        order_cat = st.selectbox("선호 주문 카테고리", df["PreferedOrderCat"].unique(), index=0)
        login = st.selectbox("선호 로그인 기기", df["PreferredLoginDevice"].unique(), index=0)
    with col6:
        pay = st.selectbox("선호 결제 방식", df["PreferredPaymentMode"].unique(), index=0)

    modified = pd.DataFrame([{
        "Tenure": tenure,
        "CityTier": selected_row["CityTier"],
        "WarehouseToHome": selected_row["WarehouseToHome"],
        "HourSpendOnApp": hour,
        "NumberOfDeviceRegistered": selected_row["NumberOfDeviceRegistered"],
        "SatisfactionScore": satisfaction,
        "NumberOfAddress": selected_row["NumberOfAddress"],
        "Complain": 1 if complain == "예" else 0,
        "OrderAmountHikeFromlastYear": selected_row["OrderAmountHikeFromlastYear"],
        "CouponUsed": selected_row["CouponUsed"],
        "OrderCount": order_count,
        "DaySinceLastOrder": last_order,
        "CashbackAmount": selected_row["CashbackAmount"],
        "PreferredLoginDevice": login,
        "PreferredPaymentMode": pay,
        "Gender": gender,
        "PreferedOrderCat": order_cat,
        "MaritalStatus": marital
    }])

    if st.button("변동 예측하기"):
        df_encoded_mod = pd.get_dummies(modified)
        for col in model_features:
            if col not in df_encoded_mod.columns:
                df_encoded_mod[col] = 0
        df_encoded_mod = df_encoded_mod[model_features]

        _, new_proba = predictor.predict(df_encoded_mod)
        new_pct = float(new_proba[0]) * 100

        fig_new = go.Figure(go.Indicator(
            mode="gauge+number",
            value=new_pct,
            number={'suffix': '%'},
            title={"text": "변경 후 이탈 확률"},
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
        st.plotly_chart(fig_new, use_container_width=True)