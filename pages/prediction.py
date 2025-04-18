import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys
import datetime
import traceback

# 경로 설정
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# 디버깅 로그 함수
def debug_log(message):
    """디버깅용 로그 기록 함수"""
    with open(f"{ROOT}/debug_log.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] [prediction.py] {message}\n")

try:
    debug_log("모듈 임포트 시작")
    from models.churn_model import load_xgboost_model2, ChurnPredictor2
    debug_log("모듈 임포트 완료")
except Exception as e:
    debug_log(f"모듈 임포트 오류: {str(e)}")
    debug_log(traceback.format_exc())

def show():
    """고객 이탈 예측 페이지를 표시합니다."""
    debug_log("prediction.py의 show() 함수 호출됨")
    
    try:
        # st.set_page_config(page_title="고객 이탈 예측", layout="wide")
        # 주의: app.py에서 이미 st.set_page_config가 호출되었으므로 여기서는 제거함
        
        st.title("📊 고객 이탈 예측 시스템")
        debug_log("페이지 타이틀 설정 완료")
        
        # 디버깅 정보 표시
        with st.expander("🔍 디버깅 정보"):
            st.write(f"현재 파일 경로: {__file__}")
            st.write(f"ROOT 경로: {ROOT}")
            st.write(f"sys.path: {sys.path}")
            st.write(f"현재 시간: {datetime.datetime.now()}")
        
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
        login_device = row5[1].selectbox("선호 로그인 기기", ["Mobile Phone", "Phone"])
        payment_mode = row5[2].selectbox("선호 결제 방식", [
            "Credit Card", "Debit Card", "Cash on Delivery", "COD", "E wallet", "UPI"])
        
        # 16~18
        gender      = row6[0].selectbox("성별", ["Male", "Female"])
        order_cat   = row6[1].selectbox("선호 주문 카테고리", [
            "Mobile", "Mobile Phone", "Laptop & Accessory", "Grocery"])
        marital     = row6[2].selectbox("결혼 유무", ["Single", "Married"])
        
        debug_log("UI 필드 설정 완료")
        
        # --------------------------
        # 2️⃣ 예측 버튼 누르면 실행
        # --------------------------
        if st.button("🧠 이탈 예측하기"):
            debug_log("예측 버튼 클릭됨")
        
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
            
            debug_log("입력 데이터 준비 완료")
        
            df_input = pd.DataFrame([raw_input])
        
            # ✅ 원-핫 인코딩 대상
            one_hot_cols = [
                "PreferredLoginDevice", "PreferredPaymentMode", "Gender",
                "PreferedOrderCat", "MaritalStatus"
            ]
            df_encoded = pd.get_dummies(df_input, columns=one_hot_cols)
            debug_log("원-핫 인코딩 완료")
        
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
                    debug_log(f"누락된 피처 추가: {col}")
        
            # 순서 맞춤
            df_encoded = df_encoded[required_features]
            debug_log("피처 정렬 완료")
        
            try:
                debug_log("모델 로딩 시작")
                model = load_xgboost_model2()
                debug_log("모델 로딩 완료")
                
                predictor = ChurnPredictor2(external_model=model)
                debug_log("예측기 초기화 완료")
                
                y_pred, y_proba = predictor.predict(df_encoded)
                debug_log(f"예측 완료: class={y_pred[0]}, prob={y_proba[0]}")
                
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
                debug_log("게이지 차트 표시 완료")
        
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
                debug_log("피처 중요도 계산 완료")
        
                # 한글 이름 적용
                importance_named = {
                    feature_name_map.get(k, k): v for k, v in importance_raw.items()
                }
        
                # 정리
                fi_df_all = pd.DataFrame(importance_named.items(), columns=["Feature", "Importance"]) \
                            .groupby("Feature").sum().sort_values("Importance", ascending=False).reset_index()
        
                # 중요 피처 상위 5개 시각화
                top5 = fi_df_all.head(5)
                fig_top = go.Figure(go.Bar(
                    x=top5["Feature"],
                    y=top5["Importance"],
                    marker_color='skyblue'
                ))
                fig_top.update_layout(
                    xaxis_title="입력 변수", yaxis_title="중요도",
                    title="📊 상위 5개 중요 변수", height=400
                )
                st.plotly_chart(fig_top, use_container_width=True)
                debug_log("피처 중요도 차트 표시 완료")
                
                # 디버그 정보
                with st.expander("디버그: 원시 예측값"):
                    st.write(f"클래스 예측값: {y_pred[0]}")
                    st.write(f"확률 예측값 (raw): {y_proba[0]:.6f}")
                    # 매우 낮은 예측값에 대한 경고
                    if float(y_proba[0]) < 0.01:
                        st.warning("원시 예측값이 매우 낮음 (< 1%): 모델 재검토 필요")
                
            except Exception as e:
                debug_log(f"예측 과정에서 오류 발생: {str(e)}")
                debug_log(traceback.format_exc())
                st.error(f"❌ 예측 실패: {str(e)}")
                st.write("오류 상세 정보:", e)
                
    except Exception as e:
        debug_log(f"show() 함수 실행 중 오류 발생: {str(e)}")
        debug_log(traceback.format_exc())
        st.error(f"페이지 로딩 중 오류가 발생했습니다: {str(e)}")
        st.write("오류 상세 정보:", e)

if __name__ == "__main__":
    debug_log("prediction.py가 직접 실행됨")
    # 이 파일이 직접 실행될 때만 set_page_config 호출
    try:
        st.set_page_config(page_title="고객 이탈 예측", layout="wide")
        debug_log("직접 실행 시 페이지 설정 완료")
        show()
    except Exception as e:
        debug_log(f"직접 실행 시 오류 발생: {str(e)}")
        debug_log(traceback.format_exc())
