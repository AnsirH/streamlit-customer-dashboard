import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from models.customer_analyzer import analyze_customers
from components.header import show_header
from components.animations import add_page_transition
from st_aggrid import AgGrid, GridOptionsBuilder

def create_importance_gauge(importance):
    """중요도를 게이지 차트로 표시"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = importance,
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
        },
        number = {'suffix': "%"}
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def show_customer_churn_analysis():
    """
    고객 이탈 예측 결과를 표시합니다.
    - 고객 ID
    - 이탈률
    - 이탈 예측에 영향을 많이 끼친 상위 3개 컬럼
    """
    try:
        # 데이터 분석 실행 (캐시 사용)
        @st.cache_data
        def load_analysis_data():
            with st.spinner("데이터 분석을 시작합니다..."):
                return analyze_customers()
        
        result_df = load_analysis_data()
        
        # 결과 표시
        st.subheader("고객 이탈 예측 결과")
        
        # 통계 정보 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            high_risk = len(result_df[result_df['Churn Risk'] >= 0.7])
            st.metric("고위험 고객 수", f"{high_risk}명", "이탈 위험 70% 이상")
        with col2:
            med_risk = len(result_df[(result_df['Churn Risk'] >= 0.4) & (result_df['Churn Risk'] < 0.7)])
            st.metric("중위험 고객 수", f"{med_risk}명", "이탈 위험 40~70%")
        with col3:
            low_risk = len(result_df[result_df['Churn Risk'] < 0.4])
            st.metric("저위험 고객 수", f"{low_risk}명", "이탈 위험 40% 미만")
        
        # 필터 컨테이너 생성
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 이탈률 필터 추가
            filter_options = ["전체", "20% 이상", "50% 이상", "70% 이상", "90% 이상"]
            selected_filter = st.selectbox("이탈률 필터", filter_options)
        
        with col2:
            # 고객 ID 검색 기능 추가
            search_id = st.text_input("고객 ID 검색", placeholder="고객 ID를 입력하세요")
        
        # 필터링된 데이터프레임 생성
        display_df = result_df[['CustomerID', 'Churn Risk', 
                              'Top Feature 1', 'Importance 1',
                              'Top Feature 2', 'Importance 2',
                              'Top Feature 3', 'Importance 3']].copy()
        
        # 고객 ID 검색이 있는 경우 이탈률 필터를 "전체"로 설정
        if search_id:
            selected_filter = "전체"
            try:
                search_id = int(search_id)
                display_df = display_df[display_df['CustomerID'] == search_id]
                
                if not display_df.empty:
                    st.subheader("선택된 고객의 이탈 영향 요인 분석")
                    factor_cols = st.columns(3)
                    
                    for i, col in enumerate(factor_cols):
                        with col:
                            feature = display_df.iloc[0][f'Top Feature {i+1}']
                            importance = display_df.iloc[0][f'Importance {i+1}']
                            st.markdown(f"**{i+1}순위 영향 요인**")
                            st.markdown(feature)
                            st.plotly_chart(create_importance_gauge(importance))
                
            except ValueError:
                st.warning("올바른 고객 ID를 입력해주세요.")
        # 고객 ID 검색이 없는 경우에만 이탈률 필터 적용
        elif selected_filter != "전체":
            threshold = float(selected_filter.split("%")[0]) / 100
            display_df = display_df[display_df['Churn Risk'] >= threshold]
            if display_df.empty:
                st.warning("해당 이탈률 기준에 맞는 고객이 없습니다.")
            else:
                display_df = display_df.sort_values('Churn Risk', ascending=False)
        else:
            # 전체 데이터는 이탈 위험도 기준으로 내림차순 정렬
            display_df = display_df.sort_values('CustomerID', ascending=True)
        
        # 컬럼명 변경
        display_df.columns = ['고객 ID', '이탈 위험도',
                            '영향 요인 1', '중요도 1',
                            '영향 요인 2', '중요도 2',
                            '영향 요인 3', '중요도 3']
        
        # 이탈 위험도를 퍼센트로 변환하고 색상 적용
        def format_risk(risk):
            risk_float = float(risk)
            if risk_float >= 0.7:
                return f'🔴 {risk_float:.1%}'
            elif risk_float >= 0.4:
                return f'🟡 {risk_float:.1%}'
            else:
                return f'🟢 {risk_float:.1%}'
        
        display_df['이탈 위험도'] = display_df['이탈 위험도'].apply(format_risk)
                
        for i in range(1, 4):
            display_df[f'영향 요인 {i}'] = display_df.apply(
                lambda x: f"{x[f'영향 요인 {i}']} ({x[f'중요도 {i}']:.1f}%)", axis=1)
        
        # 불필요한 중요도 컬럼 제거
        display_df = display_df.drop(['중요도 1', '중요도 2', '중요도 3'], axis=1)
        
        # 필터링된 결과 개수 표시
        st.write(f"총 {len(display_df)}명의 고객이 선택되었습니다.")
        
        # 참고사항 표시
        st.info("""
        **이탈 위험도 표시:**
        - 🔴 70% 이상: 고위험 고객
        - 🟡 40~70%: 중위험 고객
        - 🟢 40% 미만: 저위험 고객
        
        **방향성 설명:**
        - (부정): 해당 특성이 이탈 확률을 높이는 방향으로 작용
        - (긍정): 해당 특성이 이탈 확률을 낮추는 방향으로 작용
        - 괄호 안의 %는 전체 예측에서 해당 특성이 차지하는 상대적 중요도입니다.
        """)
        
        # 데이터프레임 표시
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "고객 ID": st.column_config.TextColumn(
                    "고객 ID",
                    help="고객의 고유 식별 번호",
                    width="small"
                ),
                "이탈 위험도": st.column_config.TextColumn(
                    "이탈 위험도",
                    help="고객의 이탈 가능성을 나타내는 지표",
                    width="small"
                ),
                "영향 요인 1": st.column_config.TextColumn(
                    "주요 영향 요인",
                    help="이탈에 가장 큰 영향을 미치는 요인",
                    width="medium"
                ),
                "영향 요인 2": st.column_config.TextColumn(
                    "2순위 영향 요인",
                    help="이탈에 두 번째로 큰 영향을 미치는 요인",
                    width="medium"
                ),
                "영향 요인 3": st.column_config.TextColumn(
                    "3순위 영향 요인",
                    help="이탈에 세 번째로 큰 영향을 미치는 요인",
                    width="medium"
                )
            },
            height=400
        )

        # 고객 ID 입력 UI
        col1, col2 = st.columns([1, 2])
        
        with col1:
            input_customer_id = st.text_input(
                "분석할 고객 ID",
                placeholder="예: CUST_000001"
            )

        with col2:
            # 입력된 ID가 실제로 존재하는지 확인
            if input_customer_id in display_df['고객 ID'].values:
                customer_info = display_df[display_df['고객 ID'] == input_customer_id].iloc[0]
                st.info(f"선택된 고객의 이탈 위험도: {customer_info['이탈 위험도']}")
                
                if st.button('선택한 고객 상세 분석'):
                    st.session_state['selected_customer_id'] = input_customer_id
            elif input_customer_id:  # 입력값이 있지만 유효하지 않은 경우
                st.error("해당 고객 ID가 존재하지 않습니다.")

    except Exception as e:
        st.error(f"데이터 분석 중 오류가 발생했습니다: {str(e)}")

