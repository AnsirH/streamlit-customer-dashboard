import streamlit as st
import pandas as pd
from components.header import show_header
from components.animations import add_page_transition
from utils.visualizer import Visualizer
from pages.customer_dashboard import show_customer_churn_analysis
from models.customer_analyzer import analyze_customers, load_customer_data

def show():
    # 애니메이션 적용
    add_page_transition()

    show_header()

    # 상단에 고객 이탈 예측 결과 표시 (대시보드)
    show_customer_churn_analysis()
    
    # 선택된 고객 ID 가져오기
    if 'selected_customer_id' not in st.session_state:
        st.warning("고객을 선택해주세요.")
        return
    
    customer_id = st.session_state['selected_customer_id']
    
    # 구분선 추가
    st.markdown("---")
    st.subheader(f"고객 {customer_id}번 상세 분석")
    
    try:
        # 전체 고객 데이터 로드
        full_data = load_customer_data()
        # 이탈 예측 결과 로드
        prediction_data = analyze_customers()
        
        # 선택된 고객의 전체 데이터 찾기
        customer_full_data = full_data[full_data['CustomerID'] == customer_id]
        if customer_full_data.empty:
            st.error(f"고객 ID {customer_id}에 대한 데이터를 찾을 수 없습니다.")
            return
        customer_full_data = customer_full_data.iloc[0]
        
        # 선택된 고객의 예측 데이터 찾기
        customer_prediction = prediction_data[prediction_data['CustomerID'] == customer_id]
        if customer_prediction.empty:
            st.error(f"고객 ID {customer_id}에 대한 예측 데이터를 찾을 수 없습니다.")
            return
        customer_prediction = customer_prediction.iloc[0]
        
    except Exception as e:
        st.error(f"데이터 로드 중 오류가 발생했습니다: {str(e)}")
        return
    
    # 메인 컨테이너
    main_container = st.container()
    
    # 레이아웃 설정 - 4:6 비율
    with main_container:
        left_col, right_col = st.columns([4, 6])
        
        # 오른쪽 열에 이탈 확률 게이지 먼저 표시
        with right_col:
            st.markdown("#### 이탈 확률")
            st.markdown('70% 이상의 이탈 확률을 가진 고객은 이탈 위험이 높습니다.')
            
            # 이탈 확률 표시
            churn_risk = customer_prediction['Churn Risk']
            
            st.plotly_chart(
                Visualizer.create_churn_gauge(churn_risk),
                use_container_width=True
            )

            # 주요 이탈 요인 표시
            st.markdown("##### 주요 이탈 요인")
            for i in range(1, 4):
                factor = customer_prediction[f'Top Feature {i}']
                st.markdown(f"**{i}. {factor}**")

        # 왼쪽 열에 고객 상세 정보 표시
        with left_col:
            st.markdown(f"### 고객번호: {customer_id}")
            
            st.markdown("##### 고객 기본 정보")
            info_data = {
                '거래기간': f"{customer_full_data['Tenure']}개월",
                '선호 로그인 기기': customer_full_data['PreferredLoginDevice'],
                '선호 결제 수단': customer_full_data['PreferredPaymentMode'],
                '성별': customer_full_data['Gender']
            }
            st.write(pd.Series(info_data))

            st.markdown("##### 주문 정보")
            order_data = {
                '주문 횟수': customer_full_data['OrderCount'],
                '마지막 주문': f"{customer_full_data['DaySinceLastOrder']}일 전",
                '주문 증가율': f"{customer_full_data['OrderAmountHikeFromlastYear']}%",
                '쿠폰 사용': f"{customer_full_data['CouponUsed']}회"
            }
            st.write(pd.Series(order_data))
            
            st.markdown("##### 만족도 정보")
            satisfaction_data = {
                '만족도': f"{customer_full_data['SatisfactionScore']}/5",
                '불만 제기': '있음' if customer_full_data['Complain'] else '없음',
                '앱 사용': f"{customer_full_data['HourSpendOnApp']}시간"
            }
            st.write(pd.Series(satisfaction_data))
    
    # 페이지 구분선
    st.markdown("---")
    
    # 상관계수 분석
    st.subheader("각 칼럼 별 이탈 여부와의 상관관계")
    st.plotly_chart(
        Visualizer.create_correlation_bar(customer_id),
        use_container_width=True
    ) 