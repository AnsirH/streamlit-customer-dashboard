import streamlit as st
import pandas as pd
from components.header import show_header
from components.animations import add_page_transition
from utils.visualizer import Visualizer
from utils.customer_analyzer import CustomerAnalyzer
from pages.customer_dashboard import show_customer_churn_analysis
from models.customer_analyzer import analyze_customers, load_customer_data


def show():
    """고객 분석 페이지를 표시합니다."""
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

####### 개인 분석 코드 #######
            
    # CustomerAnalyzer 인스턴스 생성
    analyzer = CustomerAnalyzer()
    
    # 데이터 로드
    if not analyzer.load_data():
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 고객 ID 선택
    customer_ids = analyzer.get_customer_ids()
    if not customer_ids:
        st.error("고객 데이터가 없습니다.")
        return
    
    customer_id = st.selectbox("고객 ID 선택", customer_ids)
    
    # 선택된 고객 분석
    analysis = analyzer.analyze_customer(customer_id)
    if analysis['customer_data'] is None:
        st.error(f"고객 ID {customer_id}의 데이터를 찾을 수 없습니다.")
        return
    
    # 메인 컨테이너
    main_container = st.container()
    
    # 레이아웃 설정 - 4:6 비율
    with main_container:
        left_col, right_col = st.columns([4, 6])
        
        # 오른쪽 열에 이탈 확률 게이지 먼저 표시
        with right_col:

            st.plotly_chart(
                analyzer.visualizer.create_churn_gauge(analysis['churn_prob']),
                use_container_width=True
            )
            
            # 주요 이탈 요인과 개선 방안을 카드 형태로 표시
            st.markdown("""
            <div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px;'>
            <h5 style='margin-top: 0; margin-bottom: 10px;'>주요 이탈 요인</h5>
            """, unsafe_allow_html=True)
            
            # 주요 이탈 요인 버튼
            top_issues = analyzer.get_top_issues(customer_id)
            cols = st.columns(3)
            for i, issue in enumerate(top_issues):
                with cols[i]:
                    if st.button(issue, key=f"btn{i}"):
                        st.session_state['selected_issue'] = issue

            # 솔루션 카드
            st.markdown("<h5 style='margin-top: 15px; margin-bottom: 10px;'>개선 방안</h5>", unsafe_allow_html=True)
            solution_card = st.container()
            with solution_card:
                # 선택된 이탈 요인에 따른 솔루션 표시
                selected_issue = st.session_state.get('selected_issue')
                if selected_issue:
                    solutions = {
                        '장기간 주문 없음': [
                            '개인화된 할인 쿠폰 발송',
                            '관심 상품 재입고 알림 서비스',
                            '최근 트렌드 상품 추천'
                        ],
                        '낮은 만족도': [
                            '전담 상담원 배정',
                            '맞춤형 서비스 제공',
                            '만족도 향상 프로그램 참여'
                        ],
                        '불만 제기 이력': [
                            '불만 사항 즉시 처리',
                            '사과 및 보상 프로그램',
                            '서비스 개선 피드백 수집'
                        ],
                        '낮은 주문 빈도': [
                            '주문 빈도 기반 할인 혜택',
                            '정기 구독 서비스 추천',
                            '자동 주문 설정 안내'
                        ],
                        '낮은 캐시백 사용': [
                            '캐시백 적립 이벤트 안내',
                            '캐시백 사용 방법 가이드',
                            '캐시백 전용 상품 추천'
                        ],
                        '낮은 앱 사용 시간': [
                            '앱 사용 보상 프로그램',
                            '앱 기능 활용 가이드',
                            '앱 전용 혜택 안내'
                        ],
                        '짧은 거래 기간': [
                            '신규 고객 전용 혜택',
                            '장기 고객 혜택 안내',
                            '충성도 프로그램 소개'
                        ],
                        '낮은 주문 금액 증가율': [
                            '구매 금액대별 할인',
                            '프리미엄 상품 추천',
                            '구매 금액 목표 달성 보상'
                        ],
                        '낮은 쿠폰 사용': [
                            '맞춤형 쿠폰 발급',
                            '쿠폰 사용 가이드',
                            '쿠폰 만료 알림 서비스'
                        ]
                    }
                    
                    if selected_issue in solutions:
                        st.markdown("""
                        <ul style='color: white; font-size: 13px; margin: 0; padding-left: 20px;'>
                        """, unsafe_allow_html=True)
                        
                        for solution in solutions[selected_issue]:
                            st.markdown(f"""
                            <li>{solution}</li>
                            """, unsafe_allow_html=True)
                            
                        st.markdown("</ul>", unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <p style='color: white; font-size: 13px; margin: 0; text-align: center;'>
                        이탈 요인을 선택하면 개선 방안이 표시됩니다.
                    </p>
                    """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # 왼쪽 열에 고객 정보 표시
        with left_col:
            # 고객 정보 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("고객ID", f"{customer_id}")
            with col2:
                try:
                    formatted_prob = f"{analysis['churn_prob']:.1%}" if analysis['churn_prob'] is not None else "계산 불가"
                    st.metric("이탈 확률", formatted_prob)
                except (ValueError, TypeError):
                    st.metric("이탈 확률", "계산 불가")
            with col3:
                # 이탈 위험도 계산
                if analysis['churn_prob'] >= 0.7:
                    risk_level = "위험"
                    bg_color = "#FF4B4B"  # 빨간색
                elif analysis['churn_prob'] >= 0.3:
                    risk_level = "보통"
                    bg_color = "#FFA500"  # 주황색
                else:
                    risk_level = "낮음"
                    bg_color = "#32CD32"  # 더 진한 연두색 (LimeGreen)
                
                st.markdown(
                    f"""
                    <div style="
                        background-color: {bg_color};
                        padding: 10px;
                        border-radius: 5px;
                        text-align: center;
                        color: white;
                    ">
                        <p style="margin: 0;">이탈 위험도</p>
                        <h3 style="margin: 0;">{risk_level}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("##### 고객 기본 정보")
            customer_data = analysis['customer_data'].iloc[0]
            info_data = pd.Series(
                data=[
                    f"{int(customer_data['Tenure'])}개월",
                    customer_data['PreferredLoginDevice'],
                    f"Tier {customer_data['CityTier']}",
                    '남성' if customer_data['Gender'] in ['M', 'Male'] else '여성'
                ],
                index=['거래기간', '선호 로그인 기기', '도시 등급', '성별'],
                name='data'
            )
            st.write(info_data)
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

            order_data = pd.Series(
                data=[
                    f"{int(customer_data['OrderCount'])}회",
                    f"{int(customer_data['DaySinceLastOrder'])}일 전",
                    f"{customer_data['OrderAmountHikeFromlastYear']}%",
                    f"${customer_data['CashbackAmount']:.2f}"
                ],
                index=['주문 횟수', '마지막 주문', '주문 증가율', '캐쉬백'],
                name='data'
            )
            st.write(order_data)
            
            st.markdown("##### 만족도 정보")
            satisfaction_data = pd.Series(
                data=[
                    f"{customer_data['SatisfactionScore']}/5",
                    '있음' if customer_data['Complain'] else '없음',
                    f"{int(customer_data['HourSpendOnApp'])}시간"
                ],
                index=['만족도', '불만 제기', '앱 사용'],
                name='data'
            )
            st.write(satisfaction_data)

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
    
    # 이탈 확률 계산
    if analyzer.df is not None:
        # 모든 고객에 대한 이탈 확률 계산 (디버그 메시지 없이)
        all_customers_data = analyzer.df.copy()
        
        # CustomerID를 인덱스로 설정
        all_customers_data.set_index('CustomerID', inplace=True)
        
        # 디버그 메시지 없이 조용히 예측 수행
        def predict_without_debug(row):
            try:
                return analyzer.predict(pd.DataFrame([row]), debug=False) or 0
            except:
                return 0
                
        all_customers_data['churn_prob'] = all_customers_data.apply(
            predict_without_debug,
            axis=1
        )
        
        # 상관관계 그래프 표시
        correlation_fig = analyzer.visualizer.create_correlation_bar(all_customers_data, customer_id)
        if correlation_fig is not None:
            st.plotly_chart(correlation_fig, use_container_width=True)
        else:
            st.error("상관관계 그래프를 생성할 수 없습니다.")