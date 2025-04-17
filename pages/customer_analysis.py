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
    
    # 선택된 고객 데이터
    customer_data = df[df['CustomerID'] == customer_id].iloc[0]
    
    # 레이아웃 설정
    col1, col2 = st.columns(2)
    
    with col1:
        # 이탈 확률 게이지
        st.plotly_chart(
            Visualizer.create_churn_gauge(customer_data['churn_probability']),
            use_container_width=True
        )
        
        # 고객 기본 정보
        st.subheader("고객 기본 정보")
        info_data = {
            '거래기간': f"{customer_data['Tenure']}개월",
            '선호 로그인 기기': customer_data['PreferredLoginDevice'],
            '도시 등급': f"Tier {customer_data['CityTier']}",
            '성별': customer_data['Gender']
        }
        st.write(pd.Series(info_data))
    
    with col2:
        # 주문 관련 정보
        st.subheader("주문 정보")
        order_data = {
            '주문 횟수': customer_data['OrderCount'],
            '마지막 주문 후 일수': customer_data['DaySinceLastOrder'],
            '주문 금액 상승률': f"{customer_data['OrderAmountHikeFromlastYear']}%",
            '캐쉬백 금액': f"${customer_data['CashbackAmount']:.2f}"
        }
        st.write(pd.Series(order_data))
        
        # 만족도 정보
        st.subheader("만족도 정보")
        satisfaction_data = {
            '만족도 점수': customer_data['SatisfactionScore'],
            '불만 제기 여부': '예' if customer_data['Complain'] else '아니오',
            '앱 사용 시간': f"{customer_data['HourSpendOnApp']}시간"
        }
        st.write(pd.Series(satisfaction_data))
    
    # 추가 시각화
    st.subheader("고객 행동 분석")
    
    # 주문 카테고리 선호도
    category_data = df[df['CustomerID'] == customer_id][['PreferedOrderCat']].value_counts().reset_index()
    st.plotly_chart(
        Visualizer.create_bar_chart(
            category_data,
            x='PreferedOrderCat',
            y='count',
            title='선호 주문 카테고리'
        ),
        use_container_width=True
    ) 