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

def show():
    """고객 이탈 예측 페이지를 표시합니다."""
    # st.set_page_config() 제거 - app.py에서 이미 호출됨
    
    st.title("📊 고객 이탈 예측 시스템")
    
    # --------------------------
    # 1️⃣ UI 입력 섹션 (총 18개)
    # --------------------------
    st.subheader("1️⃣ 고객 데이터 입력")
    
    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(3)
    row5 = st.columns(3)
    row6 = st.columns(3)
    
    # 1~3
    tenure         = row1[0].number_input("이용 기간 (개월)", min_value=0, value=12)
    city_tier      = row1[1].selectbox("거주 도시 등급 (1~3)", [1, 2, 3], index=1)
    warehouse_dist = row1[2].number_input("창고-집 거리 (km)", min_value=0.0, value=20.0)
    
    # 4~6
    app_hour    = row2[0].number_input("앱 사용 시간 (시간)", min_value=0.0, value=2.5)
    num_devices = row2[1].number_input("등록된 기기 수", min_value=0, value=2)
    satisfaction= row2[2].slider("만족도 점수 (1~5)", 1, 5, 3)
    
    # 7~9
    num_address = row3[0].number_input("배송지 등록 수", min_value=0, value=1)
    complain    = row3[1].selectbox("불만 제기 유무", ["예", "아니오"])
    order_hike  = row3[2].number_input("주문금액 상승률 (%)", value=10.0)
    
    # 10~12
    coupon_used = row4[0].number_input("쿠폰 사용 횟수", value=2)
    orders      = row4[1].number_input("주문 횟수", value=8)
    last_order_days = row4[2].number_input("마지막 주문 후 경과일", value=10)
    
    # 13~15
    cashback     = row5[0].number_input("캐시백 금액", value=150)
    login_device = row5[1].selectbox("선호 로그인 기글", ["Mobile Phone", "Phone"])
    payment_mode = row5[2].selectbox("선호 결제 방식", [
        "Credit Card", "Debit Card", "Cash on Delivery", "COD", "E wallet", "UPI"])
    
    # 16~18
    gender      = row6[0].selectbox("성별", ["Male", "Female"])
    order_cat   = row6[1].selectbox("선호 주문 카테고리", [
        "Mobile", "Mobile Phone", "Laptop & Accessory", "Grocery"])
    marital     = row6[2].selectbox("결혼 유무", ["Single", "Married"])
    
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
                'Tenure': '이용 기간',
                'CityTier': '거주 도시 등급',
                'WarehouseToHome': '창고-집 거리',
                'HourSpendOnApp': '앱 사용 시간',
                'NumberOfDeviceRegistered': '등록된 기기 수',
                'SatisfactionScore': '만족도 점수',
                'NumberOfAddress': '배송지 등록 수',
                'Complain': '불만 제기 여부',
                'OrderAmountHikeFromlastYear': '주문금액 상승률',
                'CouponUsed': '쿠폰 사용 횟수',
                'OrderCount': '주문 횟수',
                'DaySinceLastOrder': '마지막 주문 후 경과일',
                'CashbackAmount': '캐시백 금액',
                'PreferredLoginDevice_Mobile Phone': '휴대전화',
                'PreferredLoginDevice_Phone': '전화',
                'PreferredPaymentMode_COD': '착불',
                'PreferredPaymentMode_Cash on Delivery': '배송',
                'PreferredPaymentMode_Credit Card': '신용카드',
                'PreferredPaymentMode_Debit Card': '선불카드',
                'PreferredPaymentMode_E wallet': '인터넷뱅킹',
                'PreferredPaymentMode_UPI': '인터페이스',
                'Gender_Male': '성별',
                'PreferedOrderCat_Grocery': '선호 주문_잡화',
                'PreferedOrderCat_Laptop & Accessory': '선호 주문_노트북&장신구',
                'PreferedOrderCat_Mobile': '선호 주문_전화',
                'PreferedOrderCat_Mobile Phone': '선호 주문_휴대전화',
                'MaritalStatus_Married': '기혼',
                'MaritalStatus_Single': '미혼'
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
    
            # 중요 피처 선택 (최대 8개)
            fi_df = fi_df_all.iloc[:8].copy()
            fi_df["Level"] = fi_df["Importance"].apply(map_importance_level)
            
            # 레벨별 색상
            level_colors = {
                "매우 높음": "#ff4b4b",  # 빨강
                "높음": "#ff9d4b",      # 주황
                "중간": "#79c3f8",      # 파랑
                "낮음": "#a3a0a0",      # 회색
                "매우 낮음": "#c9c9c9"  # 연한 회색
            }
            
            # 색상 컬럼 추가
            fi_df["Color"] = fi_df["Level"].apply(lambda x: level_colors.get(x))
            
            # 데이터 시각화
            fig = go.Figure()
            
            # 바차트 추가
            fig.add_trace(go.Bar(
                x=fi_df["Feature"],
                y=fi_df["Importance"],
                marker_color=fi_df["Color"],
                text=fi_df["Level"],
                textposition="outside"
            ))
            
            fig.update_layout(
                title="주요 특성 중요도",
                xaxis_title="특성",
                yaxis_title="중요도",
                height=500
            )
            
            # 차트 표시
            st.plotly_chart(fig, use_container_width=True)
            
            # 표도 함께 표시
            with st.expander("주요 특성 상세 정보"):
                # 색상 변환을 위해 스타일 함수 정의
                def color_importance(val):
                    if val >= 0.12: return "background-color: #ff4b4b; color: white"
                    elif val >= 0.08: return "background-color: #ff9d4b; color: white"
                    elif val >= 0.05: return "background-color: #79c3f8; color: white"
                    elif val >= 0.02: return "background-color: #a3a0a0; color: white"
                    else: return "background-color: #c9c9c9; color: black"
                
                # 정렬된 데이터프레임 표시 (상위 15개)
                styled_df = fi_df_all.head(15).style.format({
                    "Importance": "{:.4f}"
                }).applymap(
                    lambda x: color_importance(x), subset=["Importance"]
                )
                
                st.dataframe(styled_df, use_container_width=True)
            
            # 추가 분석: 시그모이드 변환 확률 (비선형 변환)
            st.header("4️⃣ 확률 조정: 시그모이드 변환")
            
            # 변곡점 조정 슬라이더
            col1, col2 = st.columns([1, 2])
            with col1:
                x0 = st.slider(
                    "변곡점 위치 조정",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.15,
                    step=0.05,
                    help="값이 낮을수록 더 많은 고객이 이탈 위험이 높은 것으로 분류됩니다."
                )
            
            # 고위험/저위험 임계값 설정
            with col2:
                col1, col2 = st.columns(2)
                with col1:
                    low_risk = st.slider("저위험 임계값", 0.0, 0.5, 0.3, 0.05)
                with col2:
                    high_risk = st.slider("고위험 임계값", 0.5, 1.0, 0.7, 0.05)
            
            # 시그모이드 변환 함수
            def sigmoid_transform(p, k=15, x0=0.15):
                import numpy as np
                return 1 / (1 + np.exp(-k * (p - x0)))
            
            # 원래 확률과 조정된 확률
            raw_prob = float(y_proba[0])
            sigmoid_prob = sigmoid_transform(raw_prob, k=15, x0=x0)
            
            # 변환 전/후 비교
            col1, col2 = st.columns(2)
            with col1:
                st.metric("원래 확률", f"{raw_prob:.1%}")
            with col2:
                # 변화량 계산
                change = (sigmoid_prob - raw_prob) * 100
                delta_str = f"{change:+.1f}%p" if change != 0 else "변화 없음"
                st.metric("조정된 확률 (시그모이드 변환)", f"{sigmoid_prob:.1%}", delta=delta_str)
            
            # 변환 정보 표시
            st.info(f"📈 시그모이드 변환 적용됨: 변환 강도 = 15, 변곡점 = {x0:.2f}")
            
            # 위험도 평가
            risk_msg = ""
            if sigmoid_prob <= low_risk:
                risk_msg = "😀 저위험: 이 고객은 이탈 가능성이 낮습니다."
            elif sigmoid_prob >= high_risk:
                risk_msg = "⚠️ 고위험: 이 고객은 이탈 가능성이 매우 높습니다!"
            else:
                risk_msg = "🔍 중간 위험: 이 고객은 이탈 가능성이 있으며 추가 분석이 필요합니다."
            
            st.markdown(f"### {risk_msg}")
            
            # 디버그 정보
            with st.expander("디버그: 원시 예측값"):
                st.write(f"클래스 예측값: {y_pred[0]}")
                st.write(f"확률 예측값 (raw): {y_proba[0]:.6f}")
                st.write(f"확률 예측값 (sigmoid): {sigmoid_prob:.6f}")
                # 매우 낮은 예측값에 대한 경고
                if raw_prob < 0.01:
                    st.warning("원시 예측값이 매우 낮음 (< 1%): 모델 재검토 필요")
            
        except Exception as e:
            st.error(f"예측 오류 발생: {str(e)}")
            st.write("오류 상세 정보:", e)
            
if __name__ == "__main__":
    show() 