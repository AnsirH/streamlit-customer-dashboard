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
st.subheader("1\ufe0f\ufe0f \uace0\uac1d \ub370\uc774\ud130 \uc785\ub825")

row1 = st.columns(3)
row2 = st.columns(3)
row3 = st.columns(3)
row4 = st.columns(3)
row5 = st.columns(3)
row6 = st.columns(3)

# 1~3
tenure         = row1[0].number_input("\uc774\uc6a9 \uae30\uac04 (\uac1c\uc6d4)", min_value=0, value=12)
city_tier      = row1[1].selectbox("\uac70\uc8fc \ub3c4\uc2dc \ub4f1\uae09 (1~3)", [1, 2, 3], index=1)
warehouse_dist = row1[2].number_input("\ucc3d\uace0-\uc9d1 \uac70\ub9ac (km)", min_value=0.0, value=20.0)

# 4~6
app_hour    = row2[0].number_input("\uc571 \uc0ac\uc6a9 \uc2dc\uac04 (\uc2dc\uac04)", min_value=0.0, value=2.5)
num_devices = row2[1].number_input("\ub4f1\ub85d\ub41c \uae30\uae30 \uc218", min_value=0, value=2)
satisfaction= row2[2].slider("\ub9cc\uc871\ub3c4 \uc810\uc218 (1~5)", 1, 5, 3)

# 7~9
num_address = row3[0].number_input("\ubc30\uc1a1\uc9c0 \ub4f1\ub85d \uc218", min_value=0, value=1)
complain    = row3[1].selectbox("\ubd88\ub9cc \uc81c\uae30 \uc720\ubb34", ["\uc608", "\uc544\ub2c8\uc624"])
order_hike  = row3[2].number_input("\uc8fc\ubb38\uae08\uc561 \uc0c1\uc2b9\ub960 (%)", value=10.0)

# 10~12
coupon_used = row4[0].number_input("\ucfe0\ud3f0 \uc0ac\uc6a9 \ud69f\uc218", value=2)
orders      = row4[1].number_input("\uc8fc\ubb38 \ud69f\uc218", value=8)
last_order_days = row4[2].number_input("\ub9c8\uc9c0\ub9c9 \uc8fc\ubb38 \ud6c4 \uac74\uc640\uc77c", value=10)

# 13~15
cashback     = row5[0].number_input("\uce90\uc2dc\ubca1 \uae08\uc561", value=150)
login_device = row5[1].selectbox("\uc120\ud638 \ub85c\uadf8\uc778 \uae30\uae00", ["Mobile Phone", "Phone"])
payment_mode = row5[2].selectbox("\uc120\ud638 \uacb0\uc81c \ubc29\uc2dd", [
    "Credit Card", "Debit Card", "Cash on Delivery", "COD", "E wallet", "UPI"])

# 16~18
gender      = row6[0].selectbox("\uc131\ubcc4", ["Male", "Female"])
order_cat   = row6[1].selectbox("\uc120\ud638 \uc8fc\ubb38 \uce74\ud14c\uace0\ub9ac", [
    "Mobile", "Mobile Phone", "Laptop & Accessory", "Grocery"])
marital     = row6[2].selectbox("\uacb0\ud63c \uc720\ubb34", ["Single", "Married"])

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

        # 3️⃣ 예측에 영향을 준 주요 요인
        st.header("3️⃣ 예측에 영향을 준 주요 요인")

        # 피처 이름 맵
        feature_name_map = {
            f"feature_{i+1}": name for i, name in enumerate([
                "이용 기간", "거주 도시 등급", "창고-집 거리", "앱 사용 시간", "등록된 기기 수",
                "만족도 점수", "배송지 등록 수", "불만 제기 여부", "주문금액 상승률", "쿠폰 사용 횟수",
                "주문 횟수", "마지막 주문 후 경과일", "캐시백 금액", "선호 로그인 기기", "선호 로그인 기기",
                "선호 결제 방식", "선호 결제 방식", "선호 결제 방식", "선호 결제 방식", "선호 결제 방식",
                "선호 결제 방식", "성별", "선호 주문 카테고리", "선호 주문 카테고리", "선호 주문 카테고리",
                "선호 주문 카테고리", "결혼 여부", "결혼 여부"
            ])
        }

        # 중요도 가져오기
        importance_raw = predictor.get_feature_importance()

        # 한글 이름 적용
        importance_named = {
            feature_name_map.get(k, k): v for k, v in importance_raw.items()
        }

        # 정리
        fi_df_all = pd.DataFrame(importance_named.items(), columns=["Feature", "Importance"]) \
                    .groupby("Feature").sum().sort_values("Importance", ascending=False).reset_index()

        # 📌 등급 함수
        def map_importance_level(value):
            if value >= 0.12: return "매우 높음"
            elif value >= 0.08: return "높음"
            elif value >= 0.05: return "중간"
            elif value >= 0.02: return "낮음"
            else: return "매우 낮음"

        debug_info = [
            {"원본 이름": k, "한글 이름": feature_name_map.get(k, "❌ 매핑 안됨")}
            for k in importance_raw
        ]

        st.subheader("🧩 입력 변수 이름 매핑 확인 (디버그)")
        st.table(debug_info)  # 또는 st.dataframe(debug_info)



        # ✅ 상위 5개 시각화
        top5 = fi_df_all.head(5)
        fig_top = go.Figure(go.Bar(
            x=top5["Feature"],
            y=top5["Importance"],
            marker_color='skyblue'
        ))
        fig_top.update_layout(
            xaxis_title="입력 변수", yaxis_title="중요도",
            title="📊 상위 5개 중요 변수 (입력값 기준)", height=400
        )
        st.plotly_chart(fig_top, use_container_width=True)

        # 해석 출력
        st.markdown("📌 **높은 연관성성:**")
        for _, row in top5.iterrows():
            level = map_importance_level(row["Importance"])
            st.markdown(f"- `{row['Feature']}` 변수의 영향도는 **{level}** 수준입니다.")

        # ✅ 하위 5개 시각화
        bottom5 = fi_df_all.tail(5)
        fig_bottom = go.Figure(go.Bar(
            x=bottom5["Feature"],
            y=bottom5["Importance"],
            marker_color='lightgrey'
        ))
        fig_bottom.update_layout(
            xaxis_title="입력 변수", yaxis_title="중요도",
            title="📉 미관여 하위 5개 변수", height=400
        )
        st.plotly_chart(fig_bottom, use_container_width=True)

        # 해석 출력
        st.markdown("📌 **낮은 연관성성**")
        for _, row in bottom5.iterrows():
            level = map_importance_level(row["Importance"])
            st.markdown(f"- `{row['Feature']}` 변수의 영향도는 **{level}** 수준입니다.")
    except Exception as e:
        st.error(f"❌ 예측 실패: {str(e)}")
