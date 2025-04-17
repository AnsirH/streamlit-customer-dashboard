import streamlit as st
from components.header import show_header
from components.animations import add_page_transition
from utils.visualizer import Visualizer
from utils.data_generator import generate_sample_data
import pandas as pd

def show():
    # 애니메이션 적용
    add_page_transition()

    show_header()

    st.title("예측")
    st.write("예측 관련 내용이 여기에 표시됩니다.")

    # 임시 데이터 생성
    df = generate_sample_data()
    
    # 레이아웃 설정
    col1, col2 = st.columns(2)
    
    with col1:
        # 이탈 위험도 분포
        st.plotly_chart(
            Visualizer.create_risk_distribution(df),
            use_container_width=True
        )
        
        # 도시 등급별 이탈률
        city_churn = df.groupby('CityTier')['Churn'].mean().reset_index()
        st.plotly_chart(
            Visualizer.create_bar_chart(
                city_churn,
                x='CityTier',
                y='Churn',
                title='도시 등급별 이탈률'
            ),
            use_container_width=True
        )
    
    with col2:
        # 만족도 점수별 이탈률
        satisfaction_churn = df.groupby('SatisfactionScore')['Churn'].mean().reset_index()
        st.plotly_chart(
            Visualizer.create_bar_chart(
                satisfaction_churn,
                x='SatisfactionScore',
                y='Churn',
                title='만족도 점수별 이탈률'
            ),
            use_container_width=True
        )
        
        # 선호 결제 방식별 이탈률
        payment_churn = df.groupby('PreferredPaymentMode')['Churn'].mean().reset_index()
        st.plotly_chart(
            Visualizer.create_bar_chart(
                payment_churn,
                x='PreferredPaymentMode',
                y='Churn',
                title='결제 방식별 이탈률'
            ),
            use_container_width=True
        )
    
    # 추가 분석
    st.subheader("이탈 위험 요인 분석")
    
    # 거래기간과 이탈률의 관계
    tenure_churn = df.groupby('Tenure')['Churn'].mean().reset_index()
    st.plotly_chart(
        Visualizer.create_bar_chart(
            tenure_churn,
            x='Tenure',
            y='Churn',
            title='거래기간별 이탈률'
        ),
        use_container_width=True
    )
    
    # 앱 사용 시간과 이탈률의 관계
    app_usage_churn = df.groupby(pd.cut(df['HourSpendOnApp'], bins=5))['Churn'].mean().reset_index()
    st.plotly_chart(
        Visualizer.create_bar_chart(
            app_usage_churn,
            x='HourSpendOnApp',
            y='Churn',
            title='앱 사용 시간별 이탈률'
        ),
        use_container_width=True
    ) 