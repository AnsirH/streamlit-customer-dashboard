import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import uuid
import sys

# 안전한 변환 함수 선언
def safe_int(value, minimum=0, fallback=0):
    try:
        val = int(float(value))
        return val if val >= minimum else fallback
    except:
        return fallback

def safe_float(value, minimum=0.0, fallback=0.0):
    try:
        val = float(value)
        return val if val >= minimum else fallback
    except:
        return fallback


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

    # 복원된 한글 컬럼 데이터 생성
    df_recovered = reverse_one_hot_columns(df_encoded)
    df_recovered["고객ID"] = df["고객ID"]
    df_recovered["이탈확률"] = df["이탈확률"]
    df_recovered["위험군"] = df["위험군"]

    # 위험군별 고객 ID (상위 10개씩)
    st.subheader("📌 위험군별 고객 ID (상위 10개)")
    for group in ["초고위험군", "고위험군", "주의단계", "관찰단계"]:
        st.markdown(f"**{group}**")
        top_ids = df[df["위험군"] == group].nlargest(10, "이탈확률")["고객ID"].tolist()
        st.write(top_ids)

    st.subheader("👤 고객 ID 선택")
    selected_id = st.selectbox("고객 ID 선택", df_recovered["고객ID"].unique())
    selected_row = df_recovered[df_recovered["고객ID"] == selected_id].iloc[0]

    # 게이지 시각화
    st.subheader("📈 이탈 확률 게이지")
    prob_pct = float(selected_row["이탈확률"] * 100)
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

    # 컬럼값 표시 및 수정 UI
    
    # 줄 별 입력 구성
        # ————————————————————————————
    # ⚙ 고객 데이터 튜닝
    # ————————————————————————————
    st.subheader("⚙ 고객 데이터 튜닝")
    updated_values = {}
    rows = [st.columns(3) for _ in range(6)]  # 3열 × 6행

    # 1행
    updated_values["이용 기간"] = rows[0][0].number_input(
        "이용 기간 (개월)",
        min_value=0,
        value=safe_int(selected_row["이용 기간"])
    )
    updated_values["거주 도시 등급"] = rows[0][1].selectbox(
        "거주 도시 등급 (1~3)",
        [1, 2, 3],
        index=max(safe_int(selected_row["거주 도시 등급"], 1, 1) - 1, 0)
    )
    updated_values["창고-집 거리"] = rows[0][2].number_input(
        "창고-집 거리 (km)",
        min_value=0.0,
        value=safe_float(selected_row["창고-집 거리"])
    )

    # 2행
    updated_values["앱 사용 시간"] = rows[1][0].number_input(
        "앱 사용 시간 (시간)",
        min_value=0.0,
        value=safe_float(selected_row["앱 사용 시간"])
    )
    updated_values["등록된 기기 수"] = rows[1][1].number_input(
        "등록된 기기 수",
        min_value=0,
        value=safe_int(selected_row["등록된 기기 수"])
    )
    updated_values["만족도 점수"] = rows[1][2].slider(
        "만족도 점수 (1~5)",
        1, 5,
        safe_int(selected_row["만족도 점수"], 1, 3)
    )

    # 3행
    updated_values["배송지 등록 수"] = rows[2][0].number_input(
        "배송지 등록 수",
        min_value=0,
        value=safe_int(selected_row["배송지 등록 수"])
    )
    # 0/1 → "예"/"아니오" 변환
    complain_label = "예" if safe_int(selected_row["불만 제기 여부"], 0, 0) == 1 else "아니오"
    updated_values["불만 제기 유무"] = rows[2][1].selectbox(
        "불만 제기 유무",
        ["예", "아니오"],
        index=0 if complain_label == "예" else 1
    )
    updated_values["주문금액 상승률"] = rows[2][2].number_input(
        "주문금액 상승률 (%)",
        min_value=0.0,
        value=safe_float(selected_row["주문금액 상승률"])
    )

    # 4행
    updated_values["쿠폰 사용 횟수"] = rows[3][0].number_input(
        "쿠폰 사용 횟수",
        min_value=0,
        value=safe_int(selected_row["쿠폰 사용 횟수"])
    )
    updated_values["주문 횟수"] = rows[3][1].number_input(
        "주문 횟수",
        min_value=0,
        value=safe_int(selected_row["주문 횟수"])
    )
    updated_values["마지막 주문 후 경과일"] = rows[3][2].number_input(
        "마지막 주문 후 경과일",
        min_value=0,
        value=safe_int(selected_row["마지막 주문 후 경과일"])
    )

    # 5행
    updated_values["캐시백 금액"] = rows[4][0].number_input(
        "캐시백 금액",
        min_value=0,
        value=safe_int(selected_row["캐시백 금액"])
    )
    updated_values["선호 로그인 기기"] = rows[4][1].selectbox(
        "선호 로그인 기기",
        ["Mobile Phone", "Phone"],
        index=["Mobile Phone", "Phone"].index(selected_row["선호 로그인 기기"])
    )
    updated_values["선호 결제 방식"] = rows[4][2].selectbox(
        "선호 결제 방식",
        ["Credit Card", "Debit Card", "Cash on Delivery", "COD", "E wallet", "UPI"],
        index=["Credit Card", "Debit Card", "Cash on Delivery", "COD", "E wallet", "UPI"].index(selected_row["선호 결제 방식"])
    )

    # 6행
    updated_values["성별"] = rows[5][0].selectbox(
        "성별",
        ["Male", "Female"],
        index=["Male", "Female"].index(selected_row["성별"])
    )
    updated_values["선호 주문 카테고리"] = rows[5][1].selectbox(
        "선호 주문 카테고리",
        ["Mobile", "Mobile Phone", "Laptop & Accessory", "Grocery"],
        index=["Mobile", "Mobile Phone", "Laptop & Accessory", "Grocery"].index(selected_row["선호 주문 카테고리"])
    )
    updated_values["결혼 유무"] = rows[5][2].selectbox(
        "결혼 유무",
        ["Single", "Married"],
        index=["Single", "Married"].index(selected_row["결혼 여부"])
    )


    for col in df_recovered.columns:
        if col in ["고객ID", "이탈확률", "위험군"]:
            continue
        if df_recovered[col].dtype == object:
            # updated_values[col] = st.selectbox(col, sorted(df_recovered[col].unique()), index=sorted(df_recovered[col].unique()).index(selected_row[col]))
        else:
            # updated_values[col] = st.number_input(col, value=float(selected_row[col]))

    if st.button("변동 예측하기"):
        df_updated = pd.DataFrame([updated_values])
        df_updated_encoded = pd.get_dummies(df_updated)
        for col in model_features:
            if col not in df_updated_encoded.columns:
                df_updated_encoded[col] = 0
        df_updated_encoded = df_updated_encoded[model_features]

        _, new_proba = predictor.predict(df_updated_encoded)
        new_pct = float(new_proba[0]) * 100

        st.success(f"변경된 이탈 확률: {new_pct:.2f}%")

        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=new_pct,
            number={'suffix': '%'},
            title={"text": "이탈 가능성 (변경 후)"},
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