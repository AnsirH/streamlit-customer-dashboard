import streamlit as st
import pandas as pd
from utils.visualizer import Visualizer
from utils.model_predictor import ModelPredictor
# from components.header import show_header
# from components.animations import add_page_transition
# from utils.data_generator import generate_sample_data

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
            except ValueError:
                st.warning("올바른 고객 ID를 입력해주세요.")
        # 고객 ID 검색이 없는 경우에만 이탈률 필터 적용
        elif selected_filter != "전체":
            threshold = float(selected_filter.split("%")[0]) / 100
            display_df = display_df[display_df['Churn Risk'] >= threshold]
            # 이탈률 기준으로 내림차순 정렬
            display_df = display_df.sort_values('Churn Risk', ascending=False)
        else:
            # 전체 데이터는 고객 ID 기준으로 정렬
            display_df = display_df.sort_values('CustomerID', ascending=True)
        
        # 컬럼명 변경
        display_df.columns = ['고객 ID', '이탈 위험도',
                            '영향 요인 1', '중요도 1',
                            '영향 요인 2', '중요도 2',
                            '영향 요인 3', '중요도 3']
        
        # 이탈 위험도와 중요도를 퍼센트로 변환
        display_df['이탈 위험도'] = display_df['이탈 위험도'].apply(lambda x: f"{x:.1%}")
        
        # 영향 요인과 중요도 결합
        display_df['영향 요인 1'] = display_df.apply(
            lambda x: f"{x['영향 요인 1']} ({x['중요도 1']:.1f}%)", axis=1)
        display_df['영향 요인 2'] = display_df.apply(
            lambda x: f"{x['영향 요인 2']} ({x['중요도 2']:.1f}%)", axis=1)
        display_df['영향 요인 3'] = display_df.apply(
            lambda x: f"{x['영향 요인 3']} ({x['중요도 3']:.1f}%)", axis=1)
        
        # 불필요한 중요도 컬럼 제거
        display_df = display_df.drop(['중요도 1', '중요도 2', '중요도 3'], axis=1)
        
        # 필터링된 결과 개수 표시
        st.write(f"총 {len(display_df)}명의 고객이 선택되었습니다.")
        
        # 참고사항 표시
        st.info("""
        **참고사항:**
        - **증가**: 해당 특성이 고객의 이탈 확률을 높이는 방향으로 작용합니다.
          - 예: "마지막 주문 후 경과일 (증가)"는 경과일이 길수록 이탈 확률이 높아짐
        - **감소**: 해당 특성이 고객의 이탈 확률을 낮추는 방향으로 작용합니다.
          - 예: "만족도 점수 (감소)"는 만족도가 높을수록 이탈 확률이 낮아짐
        - 괄호 안의 숫자는 해당 특성이 이탈 확률에 미치는 영향의 크기를 나타냅니다.
        """)
        
        # 테이블 스타일 설정
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "고객 ID": st.column_config.TextColumn("고객 ID", width="small"),
                "이탈 위험도": st.column_config.TextColumn("이탈 위험도", width="small"),
                "영향 요인 1": st.column_config.TextColumn("영향 요인 1", width="medium"),
                "영향 요인 2": st.column_config.TextColumn("영향 요인 2", width="medium"),
                "영향 요인 3": st.column_config.TextColumn("영향 요인 3", width="medium")
            }
        )
        
    except Exception as e:
        st.error(f"데이터 분석 중 오류가 발생했습니다: {str(e)}")

def display_customer_info(customer_data):
    """
    고객의 기본 정보를 표시합니다.
    
    Args:
        customer_data (pd.Series): 고객 정보를 포함하는 Series
    """
    # 고객 번호 표시
    st.markdown(f"### 고객번호: {customer_data['CustomerID']}")
    
    # 고객 기본 정보
    st.markdown("##### 고객 기본 정보")
    info_data = {
        '거래기간': f"{customer_data['Tenure']}개월",
        '선호 로그인 기기': customer_data['PreferredLoginDevice'],
        '도시 등급': f"Tier {customer_data['CityTier']}",
        '성별': customer_data['Gender']
    }
    st.write(pd.Series(info_data))

def display_order_info(customer_data):
    """
    고객의 주문 정보를 표시합니다.
    
    Args:
        customer_data (pd.Series): 고객 정보를 포함하는 Series
    """
    st.markdown("##### 주문 정보")
    order_data = {
        '주문 횟수': customer_data['OrderCount'],
        '마지막 주문': f"{customer_data['DaySinceLastOrder']}일 전",
        '주문 증가율': f"{customer_data['OrderAmountHikeFromlastYear']}%",
        '캐쉬백': f"${customer_data['CashbackAmount']:.2f}"
    }
    st.write(pd.Series(order_data))

def display_satisfaction_info(customer_data):
    """
    고객의 만족도 정보를 표시합니다.
    
    Args:
        customer_data (pd.Series): 고객 정보를 포함하는 Series
    """
    st.markdown("##### 만족도 정보")
    satisfaction_data = {
        '만족도': f"{customer_data['SatisfactionScore']}/5",
        '불만 제기': '있음' if customer_data['Complain'] else '없음',
        '앱 사용': f"{customer_data['HourSpendOnApp']}시간"
    }
    st.write(pd.Series(satisfaction_data))

def display_churn_analysis(customer_data):
    """
    고객의 이탈 위험도와 관련 정보를 표시합니다.
    
    Args:
        customer_data (pd.Series): 고객 정보를 포함하는 Series
    """
    st.markdown("#### 이탈 확률")
    st.markdown('70% 이상의 이탈 확률을 가진 고객은 이탈 위험이 높습니다.')
    st.plotly_chart(
        Visualizer.create_churn_gauge(customer_data['churn_probability']),
        use_container_width=True
    )

def display_churn_factors(customer_data):
    """
    고객의 주요 이탈 요인을 표시합니다.
    
    Args:
        customer_data (pd.Series): 고객 정보를 포함하는 Series
    """
    st.markdown("##### 주요 이탈 요인")
    st.markdown(f"""
    <div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin-top: 10px; margin-bottom: 20px;'>
        <p style='color: white; font-size: 15px; margin: 0;'>
            마지막 주문 후 일수: {customer_data['DaySinceLastOrder']}일
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_improvement_suggestions():
    """
    이탈 위험을 줄이기 위한 개선 방안을 표시합니다.
    """
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

def show_customer_dashboard(customer_data):
    """
    고객 대시보드를 표시합니다.
    
    Args:
        customer_data (pd.Series): 고객 정보를 포함하는 Series
    """
    # 메인 컨테이너
    main_container = st.container()
    
    # 레이아웃 설정 - 4:6 비율
    with main_container:
        left_col, right_col = st.columns([4, 6])
        
        # 왼쪽 열에 고객 정보 표시
        with left_col:
            display_customer_info(customer_data)
            display_order_info(customer_data)
            display_satisfaction_info(customer_data)
        
        # 오른쪽 열에 이탈 분석 정보 표시
        with right_col:
            display_churn_analysis(customer_data)
            display_churn_factors(customer_data)
            display_improvement_suggestions()

# def show():
#     """고객 대시보드 페이지를 표시하는 함수"""
#     ModelPredictor.show()

# def show():
#     # 애니메이션 적용
#     add_page_transition()

#     show_header()

#     # 임시 데이터 생성
#     df = generate_sample_data()
    
#     # 고객 목록 표시
#     st.subheader("고객 목록")
    
#     # 고객 ID를 클릭 가능한 링크로 표시
#     for customer_id in df['CustomerID'].unique():
#         if st.button(f"고객 ID: {customer_id}"):
#             st.session_state['selected_customer_id'] = customer_id
#             st.switch_page("pages/customer_analysis.py")