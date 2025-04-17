import streamlit as st
import pandas as pd
from components.header import show_header
from components.animations import add_page_transition
from utils.visualizer import Visualizer
from utils.data_generator import generate_sample_data

def show():
    # 애니메이션 적용
    add_page_transition()

    show_header()

    # 임시 데이터 생성
    df = generate_sample_data()
    
    # 고객 선택
    customer_id = st.selectbox(
        "고객 ID 선택",
        df['CustomerID'].tolist()
    )

    # # 선택된 고객 ID 가져오기
    # customer_id = st.session_state.get('selected_customer_id')
    # if not customer_id:
    #     st.warning("고객을 선택해주세요.")
    #     return
    
    # 선택된 고객 데이터
    customer_data = df[df['CustomerID'] == customer_id].iloc[0]
    
    # 메인 컨테이너
    main_container = st.container()
    
    # 레이아웃 설정 - 4:6 비율
    with main_container:
        left_col, right_col = st.columns([4, 6])
        
        # 오른쪽 열에 이탈 확률 게이지 먼저 표시
        with right_col:
            st.markdown("#### 이탈 확률")
            st.markdown('70% 이상의 이탈 확률을 가진 고객은 이탈 위험이 높습니다.')
            st.plotly_chart(
                Visualizer.create_churn_gauge(customer_data['churn_probability']),
                use_container_width=True
            )

            # 주요 이탈 요인과 개선 방안을 카드 형태로 표시
            st.markdown("##### 주요 이탈 요인")
            st.markdown("""
            <div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin-top: 10px; margin-bottom: 20px;'>
                <p style='color: white; font-size: 15px; margin: 0;'>
                    마지막 주문 후 일수: {days}일
                </p>
            </div>
            """.format(days=customer_data['DaySinceLastOrder']), unsafe_allow_html=True)

            st.markdown("##### 개선 방안")
            st.markdown("""
            <div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin-top: 10px;'>
                <ul style='color: white; font-size: 13px; margin: 0; padding-left: 20px;'>
                    <li>개인화된 할인 쿠폰 발송</li>
                    <li>관심 상품 재입고 알림 서비스</li>
                    <li>최근 트렌드 상품 추천</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # 왼쪽 열에 고객 정보 표시
        with left_col:
            # 고객 번호 표시
            st.markdown(f"### 고객번호: {customer_id}")
            
            st.markdown("##### 고객 기본 정보")
            info_data = {
                '거래기간': f"{customer_data['Tenure']}개월",
                '선호 로그인 기기': customer_data['PreferredLoginDevice'],
                '도시 등급': f"Tier {customer_data['CityTier']}",
                '성별': customer_data['Gender']
            }
            st.write(pd.Series(info_data))

            # 주문정보와 만족도정보를 순차적으로 표시
            st.markdown("##### 주문 정보")
            order_data = {
                '주문 횟수': customer_data['OrderCount'],
                '마지막 주문': f"{customer_data['DaySinceLastOrder']}일 전",
                '주문 증가율': f"{customer_data['OrderAmountHikeFromlastYear']}%",
                '캐쉬백': f"${customer_data['CashbackAmount']:.2f}"
            }
            st.write(pd.Series(order_data))
            
            st.markdown("##### 만족도 정보")
            satisfaction_data = {
                '만족도': f"{customer_data['SatisfactionScore']}/5",
                '불만 제기': '있음' if customer_data['Complain'] else '없음',
                '앱 사용': f"{customer_data['HourSpendOnApp']}시간"
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