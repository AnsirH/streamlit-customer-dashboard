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

    st.title("전체 데이터")
    st.write("전체 데이터 관련 내용이 여기에 표시됩니다.")

    # 임시 데이터 생성
    df = generate_sample_data()
    
    # 데이터 미리보기
    st.subheader("데이터 미리보기")
    st.dataframe(df.head())
    
    # 기본 통계 정보
    st.subheader("기본 통계 정보")
    st.write(df.describe())
    
    # 이탈 고객 vs 유지 고객 비교
    st.subheader("이탈 고객 vs 유지 고객 비교")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 거래기간 비교
        tenure_comparison = df.groupby('Churn')['Tenure'].mean().reset_index()
        st.plotly_chart(
            Visualizer.create_bar_chart(
                tenure_comparison,
                x='Churn',
                y='Tenure',
                title='평균 거래기간 비교'
            ),
            use_container_width=True
        )
        
        # 앱 사용 시간 비교
        app_usage_comparison = df.groupby('Churn')['HourSpendOnApp'].mean().reset_index()
        st.plotly_chart(
            Visualizer.create_bar_chart(
                app_usage_comparison,
                x='Churn',
                y='HourSpendOnApp',
                title='평균 앱 사용 시간 비교'
            ),
            use_container_width=True
        )
    
    with col2:
        # 만족도 점수 비교
        satisfaction_comparison = df.groupby('Churn')['SatisfactionScore'].mean().reset_index()
        st.plotly_chart(
            Visualizer.create_bar_chart(
                satisfaction_comparison,
                x='Churn',
                y='SatisfactionScore',
                title='평균 만족도 점수 비교'
            ),
            use_container_width=True
        )
        
        # 주문 횟수 비교
        order_comparison = df.groupby('Churn')['OrderCount'].mean().reset_index()
        st.plotly_chart(
            Visualizer.create_bar_chart(
                order_comparison,
                x='Churn',
                y='OrderCount',
                title='평균 주문 횟수 비교'
            ),
            use_container_width=True
        )
    
    # 상관관계 분석
    st.subheader("상관관계 분석")
    
    # 수치형 변수들 간의 상관관계
    numeric_cols = ['Tenure', 'HourSpendOnApp', 'OrderCount', 'CashbackAmount', 'SatisfactionScore']
    correlation = df[numeric_cols].corr().round(2)  # 소수점 둘째 자리까지 반올림
    
    # 상관관계 히트맵
    try:
        st.plotly_chart(
            Visualizer.create_correlation_heatmap(correlation),
            use_container_width=True
        )
    except Exception as e:
        st.error(f"상관관계 히트맵 생성 중 오류가 발생했습니다: {str(e)}")
        st.write("상관관계 행렬:")
        st.dataframe(correlation) 