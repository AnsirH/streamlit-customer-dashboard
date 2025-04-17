import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.header import show_header
from components.animations import add_page_transition
from models.churn_model import ChurnPredictor

# 캐시된 모델 로딩
@st.cache_resource
def get_predictor():
    return ChurnPredictor()

# 게이지 차트 생성
def create_churn_gauge(value: float) -> go.Figure:
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

def show():
    # 애니메이션 적용
    add_page_transition()

    # 헤더 표시
    show_header()
    
    # 앱 시작
    st.title("고객 이탈 예측")
    
    # 모델 특성 확인 (디버그용)
    with st.expander("모델 특성 확인"):
        predictor = ChurnPredictor()
        if predictor.model is not None and hasattr(predictor.model, 'feature_names_in_'):
            st.write("### 모델이 사용하는 특성 이름 목록")
            feature_names = predictor.model.feature_names_in_
            
            # 원핫인코딩된 특성 그룹 분석
            encoded_features = predictor.get_onehot_encoded_features()
            
            # 원핫인코딩된 특성 그룹 표시
            st.write("### 원핫인코딩 된 특성 그룹")
            if encoded_features:
                total_encoded = sum(len(cols) for cols in encoded_features.values())
                st.write(f"원본 변수 {len(encoded_features)}개가 원핫인코딩되어 총 {total_encoded}개 변수로 확장됨")
                
                for prefix, columns in encoded_features.items():
                    with st.expander(f"{prefix} → {len(columns)}개 변수"):
                        # 예상되는 원본 값 추출 (접두사 제거)
                        original_values = [col.replace(f"{prefix}_", "") for col in columns]
                        st.write("예상되는 원본 값들:", ", ".join(sorted(original_values)))
                        st.write("원핫인코딩된 컬럼들:", sorted(columns))
            else:
                st.warning("원핫인코딩된 특성을 찾을 수 없습니다.")
            
            # 전체 특성 목록
            st.write("### 전체 특성 목록")
            non_encoded = [f for f in sorted(feature_names) if not any(f in cols for cols in encoded_features.values())]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("원핫인코딩 되지 않은 특성:")
                st.write(non_encoded)
            
            with col2:
                st.write("특성 수 요약:")
                st.write(f"- 총 특성 수: {len(feature_names)}")
                st.write(f"- 원핫인코딩 특성 수: {total_encoded if encoded_features else 0}")
                st.write(f"- 일반 특성 수: {len(non_encoded)}")
        else:
            st.error("모델이 로드되지 않았거나 feature_names_in_ 속성이 없습니다.")
    
    st.write("고객 정보를 입력하여 이탈 가능성을 예측해보세요.")

    # 입력 컬럼 정의 (+ CustomerID 추가, Churn 제외)
    columns = [
        'CustomerID',                  # 고객 ID (추가)
        'Tenure',                      # 거래기간
        'PreferredLoginDevice',        # 선호 로그인 기기
        'CityTier',                    # 도시 등급
        'WarehouseToHome',             # 창고에서 집까지 거리
        'PreferredPaymentMode',        # 선호 결제 방식
        'Gender',                      # 성별
        'HourSpendOnApp',              # 앱 사용 시간
        'NumberOfDeviceRegistered',    # 등록된 기기 수
        'PreferedOrderCat',            # 선호 주문 카테고리
        'SatisfactionScore',           # 만족도 점수
        'MaritalStatus',               # 결혼 상태
        'NumberOfAddress',             # 주소 개수
        'Complain',                    # 불만 제기 여부
        'OrderAmountHikeFromlastYear', # 작년 대비 주문 금액 증가율
        'CouponUsed',                  # 쿠폰 사용 횟수
        'OrderCount',                  # 주문 횟수
        'DaySinceLastOrder',           # 마지막 주문 후 경과일
        'CashbackAmount'               # 캐시백 금액
    ]

    # 필수 입력 필드 지정 (나머지는 선택 사항)
    required_columns = [
        'CustomerID',              # 고객 ID (필수 추가)
        'Tenure',                  # 거래기간
        'SatisfactionScore',       # 만족도 점수
        'OrderCount',              # 주문 횟수
        'HourSpendOnApp',          # 앱 사용 시간
        'DaySinceLastOrder'        # 마지막 주문 후 경과일
    ]

    # 컬럼별 한글 이름 매핑
    column_korean_names = {
        'CustomerID': '고객 ID',
        'Tenure': '거래기간 (개월)',
        'PreferredLoginDevice': '선호 로그인 기기',
        'CityTier': '도시 등급',
        'WarehouseToHome': '창고-집 거리 (km)',
        'PreferredPaymentMode': '선호 결제 방식',
        'Gender': '성별',
        'HourSpendOnApp': '앱 사용 시간 (시간)',
        'NumberOfDeviceRegistered': '등록된 기기 수',
        'PreferedOrderCat': '선호 주문 카테고리',
        'SatisfactionScore': '만족도 점수 (1-5)',
        'MaritalStatus': '결혼 상태',
        'NumberOfAddress': '주소 개수',
        'Complain': '불만 제기 여부',
        'OrderAmountHikeFromlastYear': '작년 대비 주문 금액 증가율 (%)',
        'CouponUsed': '쿠폰 사용 횟수',
        'OrderCount': '주문 횟수',
        'DaySinceLastOrder': '마지막 주문 후 경과일',
        'CashbackAmount': '캐시백 금액 (원)'
    }

    # 타입과 선택지 매핑
    column_types = {
        'CustomerID': 'text',       # 고객 ID는 텍스트 타입
        'Tenure': 'number',
        'PreferredLoginDevice': 'select',
        'CityTier': 'number',
        'WarehouseToHome': 'number',
        'PreferredPaymentMode': 'select',
        'Gender': 'select',
        'HourSpendOnApp': 'number',
        'NumberOfDeviceRegistered': 'number',
        'PreferedOrderCat': 'select',
        'SatisfactionScore': 'number',
        'MaritalStatus': 'select',
        'NumberOfAddress': 'number',
        'Complain': 'select',
        'OrderAmountHikeFromlastYear': 'number',
        'CouponUsed': 'number',
        'OrderCount': 'number',
        'DaySinceLastOrder': 'number',
        'CashbackAmount': 'number'
    }

    # 선택 항목 매핑
    select_options = {
        'PreferredLoginDevice': ['Mobile', 'Desktop', 'Tablet'],
        'PreferredPaymentMode': ['Credit Card', 'Debit Card', 'UPI', 'Cash on Delivery'],
        'Gender': ['Male', 'Female'],
        'PreferedOrderCat': ['Electronics', 'Fashion', 'Grocery', 'Home'],
        'MaritalStatus': ['Single', 'Married', 'Divorced'],
        'Complain': ['예', '아니오']
    }

    # 기본값 매핑
    default_values = {
        'CustomerID': 'CUST-' + ''.join(['0' + str(np.random.randint(10000, 100000))]),  # 랜덤 고객 ID 
        'Tenure': 12,
        'PreferredLoginDevice': 'Mobile',
        'CityTier': 1,
        'WarehouseToHome': 20,
        'PreferredPaymentMode': 'Credit Card',
        'Gender': 'Male',
        'HourSpendOnApp': 3.0,
        'NumberOfDeviceRegistered': 2,
        'PreferedOrderCat': 'Electronics',
        'SatisfactionScore': 3,
        'MaritalStatus': 'Single',
        'NumberOfAddress': 2,
        'Complain': '아니오',
        'OrderAmountHikeFromlastYear': 15.0,
        'CouponUsed': 3,
        'OrderCount': 10,
        'DaySinceLastOrder': 15,
        'CashbackAmount': 150.0
    }

    # 필수 입력 필드 표시
    st.markdown("""
    <style>
    .required-field::after {
        content: " *";
        color: red;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("**참고:** 빨간색 별표(\*) 표시가 있는 필드는 필수 입력 사항입니다.")

    # 입력값을 저장할 딕셔너리
    input_data = {}

    # 필수 입력 필드 검증용 변수
    required_fields_filled = {col: False for col in required_columns}

    # 필드 카테고리로 그룹화 (필수/선택)
    required_fields = []
    optional_fields = []

    for col in columns:
        if col in required_columns:
            required_fields.append(col)
        else:
            optional_fields.append(col)

    # 탭 생성
    tab1, tab2 = st.tabs(["필수 입력 필드", "선택 입력 필드"])

    with tab1:
        # 필수 입력 필드 그리드
        st.markdown("### 필수 입력 필드")
        
        # 그리드 열 수 계산 (최대 3개 컬럼)
        req_cols = 3
        req_rows = (len(required_fields) + req_cols - 1) // req_cols
        
        for row in range(req_rows):
            cols = st.columns(req_cols)
            for col in range(req_cols):
                idx = row * req_cols + col
                
                if idx < len(required_fields):
                    column = required_fields[idx]
                    korean_name = column_korean_names.get(column, column)
                    # 필수 입력 표시 추가
                    label = f"<div class='required-field'>{korean_name}</div>"
                    col_type = column_types.get(column, 'text')
                    
                    with cols[col]:
                        st.markdown(label, unsafe_allow_html=True)
                        
                        # 컬럼 타입에 따라 적절한 입력 위젯 생성
                        if col_type == 'text':
                            # 텍스트 타입 입력 (고객 ID 등)
                            value = st.text_input(
                                "",  # 레이블은 위에서 이미 표시
                                value=default_values.get(column, ""),
                                key=f"input_{column}"
                            )
                            input_data[column] = value
                            if value and column in required_columns:
                                required_fields_filled[column] = True
                            
                        elif col_type == 'number':
                            # 숫자 타입 입력
                            min_val = 0 if column in ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 
                                                   'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                                                   'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 
                                                   'DaySinceLastOrder', 'CashbackAmount'] else 0.0
                            
                            # 특별 범위 설정
                            if column == 'SatisfactionScore':
                                min_val, max_val = 1, 5
                            else:
                                max_val = None
                            
                            if column in ['HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CashbackAmount']:
                                # 소수점이 필요한 경우
                                value = st.number_input(
                                    "",  # 레이블은 위에서 이미 표시
                                    min_value=float(min_val) if min_val is not None else None,
                                    max_value=float(max_val) if max_val is not None else None,
                                    value=float(default_values.get(column, 0.0)),
                                    step=0.1,
                                    key=f"input_{column}"
                                )
                                input_data[column] = value
                                if value > 0 or column not in required_columns:
                                    required_fields_filled[column] = True
                            else:
                                # 정수 입력
                                value = st.number_input(
                                    "",  # 레이블은 위에서 이미 표시
                                    min_value=int(min_val) if min_val is not None else None,
                                    max_value=int(max_val) if max_val is not None else None,
                                    value=int(default_values.get(column, 0)),
                                    step=1,
                                    key=f"input_{column}"
                                )
                                input_data[column] = value
                                if value > 0 or column not in required_columns:
                                    required_fields_filled[column] = True
                        
                        elif col_type == 'select':
                            # 선택형 입력
                            options = select_options.get(column, [])
                            default_idx = options.index(default_values.get(column)) if default_values.get(column) in options else 0
                            
                            if column == 'Complain':
                                # 예/아니오 선택의 경우 boolean으로 변환
                                selected = st.selectbox(
                                    "",  # 레이블은 위에서 이미 표시
                                    options,
                                    index=0 if default_values.get(column) == '아니오' else 1,
                                    key=f"input_{column}"
                                )
                                input_data[column] = selected  # 문자열 '예'/'아니오' 그대로 유지
                            else:
                                input_data[column] = st.selectbox(
                                    "",  # 레이블은 위에서 이미 표시
                                    options,
                                    index=default_idx,
                                    key=f"input_{column}"
                                )
                            
                            # 선택형 필드는 항상 값이 있으므로 필수 필드 충족 처리
                            if column in required_columns:
                                required_fields_filled[column] = True

    with tab2:
        # 선택 입력 필드 그리드
        st.markdown("### 선택 입력 필드")
        
        # 그리드 열 수 계산 (최대 3개 컬럼)
        opt_cols = 3
        opt_rows = (len(optional_fields) + opt_cols - 1) // opt_cols
        
        for row in range(opt_rows):
            cols = st.columns(opt_cols)
            for col in range(opt_cols):
                idx = row * opt_cols + col
                
                if idx < len(optional_fields):
                    column = optional_fields[idx]
                    korean_name = column_korean_names.get(column, column)
                    col_type = column_types.get(column, 'text')
                    
                    with cols[col]:
                        # 컬럼 타입에 따라 적절한 입력 위젯 생성
                        if col_type == 'text':
                            # 텍스트 타입 입력
                            input_data[column] = st.text_input(
                                korean_name,
                                value=default_values.get(column, ""),
                                key=f"input_{column}"
                            )
                        
                        elif col_type == 'number':
                            # 숫자 타입 입력
                            min_val = 0 if column in ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp', 
                                                   'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                                                   'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 
                                                   'DaySinceLastOrder', 'CashbackAmount'] else 0.0
                            
                            # 특별 범위 설정
                            if column == 'SatisfactionScore':
                                min_val, max_val = 1, 5
                            else:
                                max_val = None
                            
                            if column in ['HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CashbackAmount']:
                                # 소수점이 필요한 경우
                                input_data[column] = st.number_input(
                                    korean_name,
                                    min_value=float(min_val) if min_val is not None else None,
                                    max_value=float(max_val) if max_val is not None else None,
                                    value=float(default_values.get(column, 0.0)),
                                    step=0.1,
                                    key=f"input_{column}"
                                )
                            else:
                                # 정수 입력
                                input_data[column] = st.number_input(
                                    korean_name,
                                    min_value=int(min_val) if min_val is not None else None,
                                    max_value=int(max_val) if max_val is not None else None,
                                    value=int(default_values.get(column, 0)),
                                    step=1,
                                    key=f"input_{column}"
                                )
                        
                        elif col_type == 'select':
                            # 선택형 입력
                            options = select_options.get(column, [])
                            default_idx = options.index(default_values.get(column)) if default_values.get(column) in options else 0
                            
                            if column == 'Complain':
                                # 예/아니오 선택의 경우 boolean으로 변환
                                selected = st.selectbox(
                                    korean_name,
                                    options,
                                    index=0 if default_values.get(column) == '아니오' else 1,
                                    key=f"input_{column}"
                                )
                                input_data[column] = selected  # 문자열 '예'/'아니오' 그대로 유지
                            else:
                                input_data[column] = st.selectbox(
                                    korean_name,
                                    options,
                                    index=default_idx,
                                    key=f"input_{column}"
                                )

    # 예측 버튼
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        predict_button = st.button("이탈 예측하기", use_container_width=True, type="primary")

    # 예측 결과 표시
    if predict_button:
        # 필수 필드 검증
        all_required_filled = all(required_fields_filled.values())
        
        if not all_required_filled:
            # 필수 입력 필드 미입력 시 경고 표시
            missing_fields = [column_korean_names.get(col, col) for col, filled in required_fields_filled.items() if not filled]
            st.error(f"다음 필수 입력 항목을 입력해주세요: {', '.join(missing_fields)}")
        else:
            # 로딩 표시
            with st.spinner("예측 중..."):
                try:
                    # 입력 데이터를 DataFrame으로 변환
                    input_df = pd.DataFrame([input_data])
                    
                    # 고객 ID 표시
                    st.markdown(f"### 고객 ID: {input_data['CustomerID']}")
                    
                    # 이탈 예측 모델 로드 및 예측
                    predictor = ChurnPredictor()
                    
                    _, y_proba = predictor.predict(input_df)
                    
                    prob_value = y_proba[0]  # 이탈 확률

                    # 결과 표시
                    st.markdown("---")
                    st.subheader("예측 결과")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 이탈 확률 게이지
                        st.plotly_chart(
                            create_churn_gauge(prob_value),
                            use_container_width=True
                        )
                    
                    with col2:
                        # 위험도 수준
                        if prob_value < 0.3:
                            risk_level = "low"
                            risk_text = "낮음"
                            risk_color = "#4CAF50"
                            action_text = "정기적인 마케팅 이메일을 보내고 일반적인 고객 관리를 유지하세요."
                        elif prob_value < 0.7:
                            risk_level = "medium"
                            risk_text = "중간"
                            risk_color = "#FFC107"
                            action_text = "특별 할인이나 개인화된 제안을 통해 참여도를 높이는 것이 좋습니다."
                        else:
                            risk_level = "high"
                            risk_text = "높음"
                            risk_color = "#F44336"
                            action_text = "즉각적인 고객 응대와 특별 혜택 제공이 필요합니다."
                        
                        # 결과 요약
                        st.markdown(f"""
                        ### 예측 결과 요약
                        - **이탈 확률**: {prob_value:.2%}
                        - **위험도**: <span style='color:{risk_color};font-weight:bold'>{risk_text}</span>
                        
                        ### 권장 조치
                        {action_text}
                        """, unsafe_allow_html=True)
                    
                    # 주요 영향 요인
                    st.subheader("주요 영향 요인")
                    
                    # 영향 요인 계산 (모델 기반)
                    feature_importance = predictor.get_feature_importance()
                    
                    # NumPy 배열에 대한 직접 불리언 평가 방지
                    is_valid_dict = isinstance(feature_importance, dict)
                    
                    if is_valid_dict:
                        # 특성 중요도를 리스트로 변환
                        factors = [
                            {"name": column_korean_names.get(feature, feature), 
                             "value": float(importance), 
                             "weight": 1.0,  # 이미 중요도에 가중치가 반영되어 있음
                             "description": f"{column_korean_names.get(feature, feature)}이(가) 이탈 확률에 영향을 미칩니다."}
                            for feature, importance in feature_importance.items()
                        ]
                        
                        # 중요도로 정렬
                        factors.sort(key=lambda x: x["value"], reverse=True)
                        
                        # 상위 3개 요인
                        top_factors = factors[:3]
                    else:
                        # 모델에서 특성 중요도를 얻지 못한 경우: 입력값 기반 계산
                        # 필요한 계수 다시 계산
                        tenure_factor = 1 - min(input_data.get('Tenure', 0), 60) / 60
                        app_factor = 1 - min(input_data.get('HourSpendOnApp', 0), 10) / 10
                        satisfaction_factor = 1 - input_data.get('SatisfactionScore', 3) / 5
                        order_factor = 1 - min(input_data.get('OrderCount', 0), 50) / 50
                        dayslast_factor = min(input_data.get('DaySinceLastOrder', 0), 90) / 90
                        
                        factors = [
                            {"name": "거래기간", "value": tenure_factor, "weight": 0.2, "description": "거래기간이 길수록 이탈 확률이 낮아집니다."},
                            {"name": "앱 사용 시간", "value": app_factor, "weight": 0.2, "description": "앱 사용 시간이 길수록 이탈 확률이 낮아집니다."},
                            {"name": "만족도 점수", "value": satisfaction_factor, "weight": 0.3, "description": "만족도가 높을수록 이탈 확률이 낮아집니다."},
                            {"name": "주문 횟수", "value": order_factor, "weight": 0.1, "description": "주문 횟수가 많을수록 이탈 확률이 낮아집니다."},
                            {"name": "마지막 주문 후 경과일", "value": dayslast_factor, "weight": 0.2, "description": "마지막 주문 이후 시간이 길수록 이탈 확률이 높아집니다."}
                        ]
                        factors.sort(key=lambda x: x["value"] * x["weight"], reverse=True)
                        top_factors = factors[:3]
                    
                    # 요인별 가중 영향력 계산
                    weighted_factors = [(f["name"], f["value"] * f["weight"], f["description"]) for f in top_factors]
                    
                    # 막대 그래프 데이터
                    factor_df = pd.DataFrame({
                        '요인': [f[0] for f in weighted_factors],
                        '영향력': [f[1] for f in weighted_factors]
                    })
                    
                    # 막대 그래프 표시
                    st.bar_chart(factor_df, x='요인', y='영향력')
                    
                    # 요인별 설명
                    for name, impact, desc in weighted_factors:
                        st.markdown(f"**{name}**: {desc}")
                    
                    # 디버그 정보 섹션 추가
                    with st.expander("🔧 디버그 정보"):
                        debug_tabs = st.tabs(["입력 데이터", "모델 정보", "예측 과정", "로그"])
                        
                        with debug_tabs[0]:
                            st.write("### 원본 입력 데이터")
                            st.dataframe(input_df)
                            
                            # 전처리 과정 확인을 위해 전처리 함수 직접 호출
                            processed_df = predictor._preprocess_data(input_df)
                            st.write("### 전처리 후 데이터 (원핫인코딩 적용)")
                            st.dataframe(processed_df)
                            
                            # 원핫인코딩 정보 표시
                            encoded_features = predictor.get_onehot_encoded_features()
                            
                            st.write("### 원핫인코딩 변환 정보")
                            if encoded_features:
                                # 범주형 변수별로 원본 값과 변환된 값 표시
                                for prefix, columns in encoded_features.items():
                                    with st.expander(f"{prefix} 변수의 원핫인코딩"):
                                        # 입력 데이터에서 해당 변수 값 확인
                                        if prefix in input_df.columns:
                                            original_value = input_df[prefix].iloc[0]
                                            st.write(f"입력값: {original_value}")
                                            
                                            # 변환 후 데이터에서 해당 원핫인코딩 컬럼들 값 확인
                                            encoded_values = {}
                                            for col in columns:
                                                if col in processed_df.columns:
                                                    encoded_values[col] = processed_df[col].iloc[0]
                                            
                                            encoded_df = pd.DataFrame([encoded_values])
                                            st.write("원핫인코딩 결과:")
                                            st.dataframe(encoded_df.T)  # 전치하여 보기 좋게 표시
                            
                            st.write("### 컬럼 변환 정보")
                            st.write(f"- 원본 입력 컬럼 수: {len(input_df.columns)}")
                            st.write(f"- 전처리 후 컬럼 수: {len(processed_df.columns)}")
                            st.write(f"- 차이: {len(processed_df.columns) - len(input_df.columns)}개 컬럼 추가됨")
                            
                            if hasattr(predictor.model, 'feature_names_in_'):
                                expected_columns = set(predictor.model.feature_names_in_)
                                actual_columns = set(processed_df.columns)
                                
                                # 모델과 입력 데이터의 컬럼 일치 여부 확인
                                if expected_columns == actual_columns:
                                    st.success("✅ 모델이 기대하는 컬럼과 전처리 후 컬럼이 정확히 일치합니다.")
                                else:
                                    st.error("❌ 모델이 기대하는 컬럼과 전처리 후 컬럼이 일치하지 않습니다.")
                                    missing = expected_columns - actual_columns
                                    extra = actual_columns - expected_columns
                                    
                                    if missing:
                                        st.write("누락된 컬럼:", missing)
                                    if extra:
                                        st.write("추가된 컬럼:", extra)
                            
                            st.write("### JSON 형식 입력 데이터")
                            st.json(input_data)
                        
                        with debug_tabs[1]:
                            st.write("### 모델 정보")
                            st.write(f"**모델 경로:** {predictor.model_path}")
                            st.write(f"**모델 로드 상태:** {'성공' if predictor.model is not None else '실패'}")
                            
                            # NumPy 배열 직접 평가 방지
                            has_cache = predictor.feature_importance_cache is not None
                            st.write(f"**특성 중요도 캐시:** {'있음' if has_cache else '없음'}")
                            
                            if hasattr(predictor.model, 'feature_importances_'):
                                st.write("### 모델 특성 중요도")
                                feature_importances = predictor.model.feature_importances_
                                
                                # 처리된 데이터프레임의 컬럼 가져오기
                                processed_df = predictor._preprocess_data(input_df)
                                
                                # 컬럼과 특성 중요도 길이 확인 및 조정
                                if len(feature_importances) == len(processed_df.columns):
                                    # 컬럼 수와 특성 중요도 수가 일치할 때
                                    importance_df = pd.DataFrame({
                                        '특성': processed_df.columns,
                                        '중요도': feature_importances
                                    })
                                    st.dataframe(importance_df.sort_values('중요도', ascending=False))
                                elif hasattr(predictor.model, 'feature_names_in_'):
                                    # 모델이 feature_names_in_ 속성을 가지고 있을 때
                                    importance_df = pd.DataFrame({
                                        '특성': predictor.model.feature_names_in_,
                                        '중요도': feature_importances
                                    })
                                    st.dataframe(importance_df.sort_values('중요도', ascending=False))
                                else:
                                    # 길이 불일치 시 인덱스만 사용
                                    st.warning(f"특성 중요도({len(feature_importances)})와 입력 컬럼 수({len(processed_df.columns)})가 일치하지 않습니다.")
                                    importance_df = pd.DataFrame({
                                        '특성 인덱스': range(len(feature_importances)),
                                        '중요도': feature_importances
                                    })
                                    st.dataframe(importance_df.sort_values('중요도', ascending=False))
                        
                        with debug_tabs[2]:
                            st.write("### 예측 과정")
                            st.write(f"**이탈 확률 값:** {prob_value}")
                            st.write(f"**이탈 위험도:** {risk_text}")
                            
                            # NumPy 배열 직접 평가 방지
                            is_dict_feature = isinstance(feature_importance, dict)
                            st.write(f"**특성 중요도 계산 방법:** {'모델 기반' if is_dict_feature else '기본값 기반'}")
                            
                            st.write("### 모든 영향 요인")
                            
                            # 안전하게 DataFrame 생성
                            try:
                                # 데이터가 존재하고 모든 필드가 있는지 확인
                                if factors and all(key in factors[0] for key in ['name', 'value', 'weight', 'description']):
                                    # 필요한 필드만 추출하여 DataFrame 생성
                                    factor_data = {
                                        '요인': [f.get('name', '') for f in factors],
                                        '값': [f.get('value', 0) for f in factors],
                                        '가중치': [f.get('weight', 0) for f in factors],
                                        '설명': [f.get('description', '') for f in factors]
                                    }
                                    all_factors_df = pd.DataFrame(factor_data)
                                    st.dataframe(all_factors_df)
                                else:
                                    st.warning("영향 요인 데이터를 표시할 수 없습니다.")
                                    st.write(factors)  # 원본 데이터 표시
                            except Exception as e:
                                st.error(f"영향 요인 처리 중 오류 발생: {str(e)}")
                                st.write("원본 데이터:", factors)
                        
                        with debug_tabs[3]:
                            st.write("### 로그 정보")
                            st.code(f"""
# 모델 로드 시도
model_path: {predictor.model_path}
model_loaded: {predictor.model is not None}

# 입력 데이터 처리
input_rows: {len(input_df)}
input_columns: {len(input_df.columns)}

# 예측 수행
probability: {prob_value:.4f}
risk_level: {risk_level}
                            """)
                        
                except Exception as e:
                    st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
                    import traceback
                    st.write("상세 오류:")
                    st.code(traceback.format_exc())

# 메인 함수 호출이 필요하면 추가
if __name__ == "__main__":
    show() 