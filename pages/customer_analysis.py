import streamlit as st
import pandas as pd
from components.header import show_header
from components.animations import add_page_transition
from utils.visualizer import Visualizer
from utils.customer_analyzer import CustomerAnalyzer
from pages.customer_dashboard import show_customer_churn_analysis
from models.customer_analyzer import analyze_customers, load_customer_data


def display_data_as_table(data_dict, title=None):
    """딕셔너리 데이터를 DataFrame으로 변환하여 표로 표시하는 함수"""
    df = pd.DataFrame({
        '항목': list(data_dict.keys()),
        '값': list(data_dict.values())
    })
    if title:
        st.markdown(f"##### {title}")
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )


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

####### 개인 분석 코드 #######
            
    # CustomerAnalyzer 인스턴스 생성
    analyzer = CustomerAnalyzer()
    
    # 고객 데이터 로드
    df = load_customer_data()
    if df is None or df.empty:
        st.error("고객 데이터를 로드할 수 없습니다.")
        return
    
    # CustomerAnalyzer에 데이터 설정
    analyzer.df = df
    
    # 선택된 고객 분석
    analysis = analyzer.analyze_customer(customer_id)
    if analysis['customer_data'] is None:
        st.error(f"고객 ID {customer_id}의 데이터를 찾을 수 없습니다.")
        return
    
    # 메인 컨테이너
    main_container = st.container()
    
    with main_container:
        # 1. 고객 기본 정보와 이탈 확률 게이지
        st.markdown("##### 고객 기본 정보")
        col1, col2 = st.columns([4, 6])
        
        with col1:
            # 이탈 확률과 위험도 표시를 위한 내부 열
            prob_col, risk_col = st.columns(2)
            
            with prob_col:
                try:
                    formatted_prob = f"{analysis['churn_prob']:.1%}" if analysis['churn_prob'] is not None else "계산 불가"
                    st.metric("이탈 확률", formatted_prob)
                except (ValueError, TypeError):
                    st.metric("이탈 확률", "계산 불가")
            
            with risk_col:
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
                        margin-bottom: 20px;
                    ">
                        <p style="margin: 0;">이탈 위험도</p>
                        <h3 style="margin: 0;">{risk_level}</h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            customer_data = analysis['customer_data'].iloc[0]
            
            # 거래기간 계산 (스케일링된 값을 원래 값으로 변환)
            # StandardScaler로 스케일링된 값을 원래 값으로 변환
            # 평균: 0.041596, 표준편차: 1.017154
            scaled_tenure = customer_data['Tenure']
            original_tenure = (scaled_tenure * 1.017154) + 0.041596
            
            # 음수 값 처리 (스케일링 오류로 인한 음수 값은 0으로 처리)
            original_tenure = max(0, original_tenure)
            
            if original_tenure < 1:
                # 1개월 미만인 경우 일 수로 변환 (30일 기준)
                tenure_days = int(original_tenure * 30)
                tenure_display = f"{tenure_days}일"
            else:
                # 1개월 이상인 경우 개월 수로 표시
                tenure_months = int(original_tenure)
                tenure_display = f"{tenure_months}개월"
            
            # 결제 수단 찾기 (스케일링된 값 중 가장 큰 값을 선택)
            payment_modes = {
                'PreferredPaymentMode_COD': 'COD',
                'PreferredPaymentMode_Cash on Delivery': '현금 결제',
                'PreferredPaymentMode_Credit Card': '신용카드',
                'PreferredPaymentMode_Debit Card': '체크카드',
                'PreferredPaymentMode_E wallet': '전자지갑',
                'PreferredPaymentMode_UPI': 'UPI'
            }
            
            # 각 결제 수단의 스케일링된 값을 가져와서 가장 큰 값을 가진 결제 수단 선택
            payment_values = {mode: customer_data[mode] for mode in payment_modes.keys()}
            selected_payment = max(payment_values.items(), key=lambda x: x[1])[0]
            
            info_data = {
                '성별': '남성' if customer_data['Gender_Male'] > 0 else '여성',
                '선호 로그인 기기': '모바일' if customer_data['PreferredLoginDevice_Mobile Phone'] > customer_data['PreferredLoginDevice_Phone'] else '전화',
                '선호 결제 수단': payment_modes[selected_payment],
                '거래기간': tenure_display
            }
            display_data_as_table(info_data, "고객 기본 정보")
            
            # 주요 영향 요인 표시
            st.markdown("##### 주요 영향 요인")
            
            # 상관관계 분석 데이터 가져오기 (캐싱 사용)
            @st.cache_data
            def get_correlation_data():
                return analyze_customers()
            
            correlation_data = get_correlation_data()
            if correlation_data is not None and not correlation_data.empty:
                # 해당 고객의 상관관계 데이터 필터링
                customer_corr = correlation_data[correlation_data['CustomerID'] == customer_id]
                if not customer_corr.empty:
                    # 표 형식으로 데이터 준비
                    influence_data = pd.DataFrame({
                        '순위': ['1순위', '2순위', '3순위'],
                        '영향 요인': [
                            customer_corr.iloc[0]['Top Feature 1'],
                            customer_corr.iloc[0]['Top Feature 2'],
                            customer_corr.iloc[0]['Top Feature 3']
                        ],
                        '중요도': [
                            f"{customer_corr.iloc[0]['Importance 1']:.1f}%",
                            f"{customer_corr.iloc[0]['Importance 2']:.1f}%",
                            f"{customer_corr.iloc[0]['Importance 3']:.1f}%"
                        ]
                    })
                    
                    # 표시할 때 인덱스 숨기기
                    st.dataframe(
                        influence_data,
                        hide_index=True,
                        use_container_width=True
                    )
        
        with col2:
            st.plotly_chart(
                analyzer.visualizer.create_churn_gauge(analysis['churn_prob']),
                use_container_width=True
            )
        
        # 구분선 추가
        st.markdown("---")
        
        # 분석 결과와 이탈 요인을 2개의 컬럼으로 나누기
        st.markdown("""
        <div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; height: 100%;'>
            <h5 style='margin: 0 0 15px 0; color: white;'>분석 결과 해석</h5>
            <p style='color: white; font-size: 14px; line-height: 1.5;'>
        """, unsafe_allow_html=True)

        # 해석 규칙 로드
        interpretation_rules = load_interpretation_rules()
        
        # 고객의 주요 이슈 표시
        top_issues = analyzer.get_top_issues(customer_id)
        if top_issues and len(top_issues) > 0:  # top_issues가 비어있지 않은지 확인
            st.subheader("주요 이슈")
            
            # 이슈 이름 매핑
            issue_mapping = {
                "마지막 주문 후 경과일 (⚠️ 부정)": "장기간 주문 없음",
                "주문 횟수 (⚠️ 부정)": "낮은 주문 빈도",
                "주문 횟수 (✅ 긍정)": "높은 주문 빈도",
                "캐시백 금액 (⚠️ 부정)": "낮은 캐시백 사용",
                "캐시백 금액 (✅ 긍정)": "높은 캐시백 사용",
                "앱 사용 시간 (⚠️ 부정)": "낮은 앱 사용 시간",
                "앱 사용 시간 (✅ 긍정)": "높은 앱 사용 시간",
                "거래 기간 (⚠️ 부정)": "짧은 거래 기간",
                "거래 기간 (✅ 긍정)": "긴 거래 기간",
                "작년 대비 주문 증가율 (⚠️ 부정)": "낮은 주문 증가율",
                "작년 대비 주문 증가율 (✅ 긍정)": "높은 주문 증가율",
                "쿠폰 사용 횟수 (⚠️ 부정)": "낮은 쿠폰 사용",
                "쿠폰 사용 횟수 (✅ 긍정)": "높은 쿠폰 사용",
                "배송 거리 (⚠️ 부정)": "먼 배송 거리",
                "배송 거리 (✅ 긍정)": "가까운 배송 거리",
                "선호 로그인 기기 (⚠️ 부정)": "비선호 로그인 기기 사용",
                "선호 로그인 기기 (✅ 긍정)": "선호 로그인 기기 사용",
                "선호 결제 수단 (E wallet) (⚠️ 부정)": "비선호 결제 수단 사용",
                "선호 결제 수단 (E wallet) (✅ 긍정)": "선호 결제 수단 사용",
                "성별 (⚠️ 부정)": "성별 관련 이슈",
                "성별 (✅ 긍정)": "성별 관련 이슈",
                "선호 주문 카테고리 (⚠️ 부정)": "비선호 카테고리 주문",
                "선호 주문 카테고리 (✅ 긍정)": "선호 카테고리 주문",
                "결혼 여부 (⚠️ 부정)": "결혼 상태 관련 이슈",
                "결혼 여부 (✅ 긍정)": "결혼 상태 관련 이슈",
                "도시 등급 (⚠️ 부정)": "도시 등급 관련 이슈",
                "도시 등급 (✅ 긍정)": "도시 등급 관련 이슈",
                "주소 개수 (⚠️ 부정)": "적은 주소 등록",
                "주소 개수 (✅ 긍정)": "많은 주소 등록",
                "등록된 기기 수 (⚠️ 부정)": "적은 기기 등록",
                "등록된 기기 수 (✅ 긍정)": "많은 기기 등록",
                "만족도 점수 (⚠️ 부정)": "낮은 만족도",
                "만족도 점수 (✅ 긍정)": "높은 만족도",
                "불만 제기 (⚠️ 부정)": "높은 불만 제기"
            }
            
            # 매핑 적용
            top_issues = [issue_mapping.get(issue, issue) for issue in top_issues]
            
            # 해석 내용 표시
            for issue in top_issues:
                interpretation_df = interpretation_rules.loc[interpretation_rules['issue'] == issue]
                if not interpretation_df.empty:
                    interpretation = interpretation_df.iloc[0]['interpretation']
                    st.write(f"**{issue}**: {interpretation}")
                else:
                    st.write(f"**{issue}**: 해석 정보 없음")
        else:
            st.warning("이 고객에 대한 주요 이슈 정보가 없습니다.")

        st.markdown("""
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # 이탈 요인
        st.markdown("##### 이탈 요인")
        
        # 버튼을 가로로 배치하기 위한 컬럼 생성
        cols = st.columns([1, 1, 1])
        
        for i, issue in enumerate(top_issues):
            with cols[i]:
                mapped_issue = issue_mapping.get(issue, issue)
                if st.button(mapped_issue, key=f"btn{i}", use_container_width=True):
                    st.session_state['selected_issue'] = mapped_issue
        
        # 솔루션 카드
        st.markdown("<h5 style='margin-top: 15px; margin-bottom: 10px; color: white;'>개선 방안</h5>", unsafe_allow_html=True)
        solution_card = st.container()
        with solution_card:
            # 선택된 이탈 요인에 따른 솔루션 표시
            selected_issue = st.session_state.get('selected_issue')
            if selected_issue:
                # 솔루션 규칙 로드
                solution_rules = load_solution_rules()
                
                solutions = solution_rules[solution_rules['issue'] == selected_issue]
                
                if not solutions.empty:
                    st.markdown("""
                    <ul style='color: white; font-size: 13px; margin: 0; padding-left: 20px;'>
                    """, unsafe_allow_html=True)
                    
                    for i in range(1, 4):
                        solution = solutions[f'solution{i}'].values[0]
                        st.markdown(f"""
                        <li>{solution}</li>
                        """, unsafe_allow_html=True)
                            
                    st.markdown("</ul>", unsafe_allow_html=True)
                else:
                    st.error(f"이슈 '{selected_issue}'에 대한 솔루션을 찾을 수 없습니다.")
            else:
                st.markdown("""
                <p style='color: white; font-size: 13px; margin: 0; text-align: center;'>
                    이탈 요인을 선택하면 개선 방안이 표시됩니다.
                </p>
                """, unsafe_allow_html=True)
        
        # 구분선 추가
        st.markdown("---")
        
        # 3. 고객 상세 정보
        st.markdown(f"### 고객번호: {customer_id}")
        
        # 주문 정보 표시
        order_data = {
            '주문 횟수': f"{int(customer_data['OrderCount'])}회",
            '마지막 주문': f"{int(customer_data['DaySinceLastOrder'])}일 전",
            '주문 증가율': f"{customer_data['OrderAmountHikeFromlastYear']:.1f}%",
            '캐쉬백': f"${customer_data['CashbackAmount']:.2f}"
        }
        
        # 만족도 정보 표시
        satisfaction_data = {
            '만족도': f"{customer_data['SatisfactionScore']}/5",
            '불만 제기': '있음' if customer_data['Complain'] else '없음',
            '앱 사용': f"{int(customer_data['HourSpendOnApp'])}시간"
        }
        
        # 스케일링된 값을 원래 값으로 변환
        # StandardScaler로 스케일링된 값을 원래 값으로 변환
        # 평균과 표준편차는 모델 학습 시 사용된 값으로 대체
        scaler_params = {
            'OrderCount': {'mean': 50, 'std': 20},  # 예시 값
            'DaySinceLastOrder': {'mean': 30, 'std': 15},  # 예시 값
            'OrderAmountHikeFromlastYear': {'mean': 15, 'std': 5},  # 예시 값
            'CashbackAmount': {'mean': 50, 'std': 25},  # 예시 값
            'SatisfactionScore': {'mean': 3, 'std': 1},  # 1-5 범위의 만족도
            'HourSpendOnApp': {'mean': 3, 'std': 2}  # 앱 사용 시간
        }
        
        # 스케일링된 값을 원래 값으로 변환
        for key, params in scaler_params.items():
            original_value = (customer_data[key] * params['std']) + params['mean']
            if key == 'OrderCount':
                order_data['주문 횟수'] = f"{int(original_value)}회"
            elif key == 'DaySinceLastOrder':
                order_data['마지막 주문'] = f"{int(original_value)}일 전"
            elif key == 'OrderAmountHikeFromlastYear':
                order_data['주문 증가율'] = f"{original_value:.1f}%"
            elif key == 'CashbackAmount':
                order_data['캐쉬백'] = f"${original_value:.2f}"
            elif key == 'SatisfactionScore':
                # 만족도는 1-5 범위로 제한
                satisfaction_score = max(1, min(5, round(original_value)))
                satisfaction_data['만족도'] = f"{satisfaction_score}/5"
            elif key == 'HourSpendOnApp':
                # 앱 사용 시간은 0 이상의 정수로 제한
                app_usage = max(0, round(original_value))
                satisfaction_data['앱 사용'] = f"{app_usage}시간"
        
        display_data_as_table(order_data, "주문 정보")
        display_data_as_table(satisfaction_data, "만족도 정보")

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
                print("예측 오류 발생:", row.name)  # 예외 발생 시 고객 ID 출력
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

# 해석 규칙 CSV 파일 로드
@st.cache_data
def load_interpretation_rules():
    try:
        # CSV 파일을 읽을 때 구분자를 명시적으로 지정하고, 컬럼 이름을 유지
        df = pd.read_csv('data/formatted_data.csv', 
                        encoding='utf-8', 
                        sep=',', 
                        keep_default_na=False)  # NaN 값을 빈 문자열로 처리
        
        # 컬럼 이름이 올바르게 로드되었는지 확인
        if 'issue' not in df.columns or 'interpretation' not in df.columns:
            st.error("해석 규칙 파일의 컬럼 이름이 올바르지 않습니다.")
            return pd.DataFrame(columns=['issue', 'interpretation'])
        
        return df
    except Exception as e:
        st.error(f"해석 규칙 파일을 로드하는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame(columns=['issue', 'interpretation'])

@st.cache_data
def load_solution_rules():
    try:
        return pd.read_csv('data/solution_rules.csv')
    except Exception as e:
        st.error(f"솔루션 규칙 파일을 로드하는 중 오류가 발생했습니다: {e}")
        return pd.DataFrame(columns=['issue', 'solution1', 'solution2', 'solution3'])