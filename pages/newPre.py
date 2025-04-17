import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import os
import sys
from pathlib import Path

# 현재 스크립트의 경로를 기준으로 프로젝트 루트 경로 설정
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir
sys.path.append(str(project_root))

# 페이지 설정
st.set_page_config(
    page_title="고객 이탈 예측 시뮬레이터",
    page_icon="🔮",
    layout="wide"
)

# ChurnPredictor 클래스 임포트
try:
    from models.churn_model import ChurnPredictor
except ImportError:
    st.error("모델을 불러오는 데 실패했습니다. 경로를 확인해주세요.")
    sys.exit(1)

# 게이지 차트 생성 함수
def create_churn_gauge(value: float) -> go.Figure:
    """이탈 확률을 게이지 차트로 시각화합니다."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={"text": "이탈 가능성 (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}
            ],
            'bar': {'color': 'darkblue'},
            'threshold': {'line': {'color': 'red', 'width': 4}, 'value': value * 100}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# 특성 중요도 막대 차트 생성 함수
def create_feature_importance_chart(features_dict):
    """특성 중요도를 막대 차트로 시각화합니다."""
    # 딕셔너리를 데이터프레임으로 변환
    df = pd.DataFrame({
        "특성": list(features_dict.keys()),
        "중요도": list(features_dict.values())
    })
    
    # 중요도 기준 정렬
    df = df.sort_values(by="중요도", ascending=False)
    
    # 상위 10개 특성만 선택
    df = df.head(10)
    
    # 막대 차트 생성
    fig = px.bar(
        df,
        x="중요도",
        y="특성",
        orientation='h',
        title="특성 중요도 (상위 10개)",
        color="중요도",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(height=400)
    return fig

def main():
    """메인 애플리케이션 함수"""
    # 헤더 섹션
    st.title("🔮 고객 이탈 예측 시뮬레이터")
    st.markdown("""
    이 애플리케이션은 고객 데이터를 기반으로 이탈 확률을 예측합니다.
    아래 입력 필드에 고객 정보를 입력하고 '예측하기' 버튼을 클릭하세요.
    """)
    
    # 사이드바 - 설정 옵션
    st.sidebar.title("설정")
    show_debug = st.sidebar.checkbox("디버그 정보 표시", value=False)
    
    # 세션 상태 초기화
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    
    # 고객 데이터 입력 섹션
    st.header("고객 데이터 입력")
    
    # 입력 필드 생성 (3열로 구성)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        customer_id = st.text_input("고객 ID", value=f"CUST-{np.random.randint(10000, 100000)}")
        tenure = st.number_input("거래기간 (개월)", min_value=0, value=12)
        preferred_login = st.selectbox("선호 로그인 기기", options=["Mobile", "Desktop", "Tablet"])
        city_tier = st.number_input("도시 등급", min_value=1, max_value=3, value=1)
        warehouse_to_home = st.number_input("창고-집 거리 (km)", min_value=0, value=20)
        gender = st.selectbox("성별", options=["Male", "Female"])
    
    with col2:
        preferred_payment = st.selectbox("선호 결제 방식", options=["Credit Card", "Debit Card", "UPI", "Cash on Delivery"])
        hour_spend_app = st.number_input("앱 사용 시간 (시간)", min_value=0.0, value=3.0)
        devices_registered = st.number_input("등록된 기기 수", min_value=0, value=2)
        preferred_order_cat = st.selectbox("선호 주문 카테고리", options=["Electronics", "Fashion", "Grocery", "Home"])
        satisfaction_score = st.slider("만족도 점수", min_value=1, max_value=5, value=3)
        marital_status = st.selectbox("결혼 상태", options=["Single", "Married", "Divorced"])
    
    with col3:
        number_of_address = st.number_input("주소 개수", min_value=0, value=2)
        complain = st.selectbox("불만 제기 여부", options=["아니오", "예"])
        order_amount_hike = st.number_input("작년 대비 주문 금액 증가율 (%)", min_value=0.0, value=15.0)
        coupon_used = st.number_input("쿠폰 사용 횟수", min_value=0, value=3)
        order_count = st.number_input("주문 횟수", min_value=0, value=10)
        days_since_last_order = st.number_input("마지막 주문 후 경과일", min_value=0, value=15)
        cashback_amount = st.number_input("캐시백 금액 (원)", min_value=0.0, value=150.0)
        
    # 예측 버튼
    predict_button = st.button("예측하기", type="primary", use_container_width=True)
    
    # 새로운 고객 데이터 생성
    if predict_button:
        new_customer_data = pd.DataFrame({
            'CustomerID': [customer_id],
            'Tenure': [tenure],
            'PreferredLoginDevice': [preferred_login],
            'CityTier': [city_tier],
            'WarehouseToHome': [warehouse_to_home],
            'PreferredPaymentMode': [preferred_payment],
            'Gender': [gender],
            'HourSpendOnApp': [hour_spend_app],
            'NumberOfDeviceRegistered': [devices_registered],
            'PreferedOrderCat': [preferred_order_cat],
            'SatisfactionScore': [satisfaction_score],
            'MaritalStatus': [marital_status],
            'NumberOfAddress': [number_of_address],
            'Complain': [complain],
            'OrderAmountHikeFromlastYear': [order_amount_hike],
            'CouponUsed': [coupon_used],
            'OrderCount': [order_count],
            'DaySinceLastOrder': [days_since_last_order],
            'CashbackAmount': [cashback_amount]
        })
        
        # 예측 수행
        try:
            # 로딩 스피너 표시
            with st.spinner("예측 중..."):
                # 예측기 인스턴스 생성
                predictor = ChurnPredictor()
                
                # 예측 수행
                y_pred, y_proba = predictor.predict(new_customer_data)
                
                # 세션 상태에 결과 저장
                st.session_state.prediction_result = {
                    'prediction': y_pred[0],
                    'probability': y_proba[0],
                    'customer_data': new_customer_data
                }
            
            # 성공 메시지
            st.success("예측이 완료되었습니다! 아래에서 결과를 확인하세요.")
        
        except Exception as e:
            # 오류 메시지
            st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
            if show_debug:
                st.exception(e)
    
    # 예측 결과 표시
    if st.session_state.prediction_result is not None:
        st.markdown("---")
        
        # 결과 데이터 가져오기
        result = st.session_state.prediction_result
        prob_value = result['probability']
        is_churn = result['prediction'] == 1
        customer_data = result['customer_data']
        
        # 결과 헤더
        st.header("예측 결과")
        st.markdown(f"고객 ID: **{customer_data['CustomerID'].values[0]}**")
        
        # 결과 시각화 - 2열 레이아웃
        col1, col2 = st.columns(2)
        
        with col1:
            # 이탈 확률 게이지 차트
            st.plotly_chart(create_churn_gauge(prob_value), use_container_width=True)
            
            # 위험도 분석
            risk_level = "높음" if prob_value >= 0.7 else ("중간" if prob_value >= 0.3 else "낮음")
            risk_color = "red" if risk_level == "높음" else ("orange" if risk_level == "중간" else "green")
            
            st.markdown(f"""
            ### 위험도 분석
            - **이탈 확률**: {prob_value * 100:.2f}%
            - **위험도**: <span style='color:{risk_color};'>{risk_level}</span>
            - **예측 결과**: {'이탈' if is_churn else '유지'}
            """, unsafe_allow_html=True)
            
            # 권장 조치
            st.subheader("권장 조치")
            if risk_level == "높음":
                st.markdown("""
                - 즉각적인 고객 응대 필요
                - 특별 할인 또는 혜택 제공 고려
                - 맞춤형 제안으로 고객 관계 강화
                """)
            elif risk_level == "중간":
                st.markdown("""
                - 고객 만족도 점검
                - 추가 서비스나 혜택 제안
                - 정기적인 소통 강화
                """)
            else:
                st.markdown("""
                - 현재 관리 방식 유지
                - 정기적인 프로모션 안내
                """)
        
        with col2:
            # 입력된 주요 특성 표시
            st.subheader("주요 고객 특성")
            
            # 핵심 특성과 한글 이름 매핑
            key_features = {
                'Tenure': '거래기간 (개월)',
                'SatisfactionScore': '만족도 점수 (1-5)',
                'OrderCount': '주문 횟수',
                'HourSpendOnApp': '앱 사용 시간 (시간)',
                'DaySinceLastOrder': '마지막 주문 후 경과일',
                'CouponUsed': '쿠폰 사용 횟수'
            }
            
            # 주요 특성 데이터 테이블
            key_data = pd.DataFrame({
                '특성': list(key_features.values()),
                '값': [customer_data[k].values[0] for k in key_features.keys()]
            })
            
            st.dataframe(key_data, use_container_width=True)
        
        # 고급 분석 섹션
        st.markdown("---")
        st.subheader("고급 분석")
        
        # 특성 중요도 시각화 (가상 데이터)
        importance_data = {
            '거래기간': 0.28,
            '마지막 주문 후 경과일': 0.21,
            '앱 사용 시간': 0.18,
            '주문 횟수': 0.15,
            '만족도 점수': 0.12,
            '캐시백 금액': 0.08,
            '도시 등급': 0.07,
            '선호 결제 방식_Credit Card': 0.05,
            '선호 로그인 기기_Mobile': 0.04,
            '불만 제기 여부': 0.03
        }
        
        st.plotly_chart(create_feature_importance_chart(importance_data), use_container_width=True)
        
        # 디버그 정보 (선택적 표시)
        if show_debug:
            with st.expander("🔧 디버그 정보"):
                st.write("### 원본 입력 데이터")
                st.dataframe(customer_data)
                
                # 전처리된 데이터 (예시)
                st.write("### 전처리 후 데이터 (원핫인코딩 적용)")
                try:
                    # 예측기 인스턴스 생성
                    predictor = ChurnPredictor()
                    processed_df = predictor._preprocess_data(customer_data)
                    st.dataframe(processed_df)
                except Exception as e:
                    st.warning(f"전처리된 데이터를 표시할 수 없습니다: {str(e)}")

if __name__ == "__main__":
    main() 