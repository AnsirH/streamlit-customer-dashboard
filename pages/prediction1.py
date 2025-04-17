import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib
import os
from components.header import show_header
from components.animations import add_page_transition

# 모델 경로 설정
MODEL_PATH = Path(os.path.dirname(__file__)) / ".." / "models" / "xgb_best_model.pkl"

# 모델과 전처리 클래스 정의
class ChurnPredictor:
    """고객 이탈 예측을 위한 모델 클래스"""
    
    def __init__(self, model_path=None):
        """모델을 로드하고 초기화합니다."""
        self.model = None
        if model_path is None:
            self.model_path = MODEL_PATH
        else:
            self.model_path = model_path
        self.feature_importance_cache = None
        self.load_model()
        
    def load_model(self):
        """모델 파일을 로드합니다."""
        try:
            if not self.model_path.exists():
                st.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                return False
            
            self.model = joblib.load(self.model_path)
            st.write(f"🔍 모델 로드 성공: {self.model_path}")
            return True
        except Exception as e:
            st.error(f"모델 로드 실패: {str(e)}")
            return False
    
    def predict(self, input_df):
        """입력 데이터에 대한 이탈 예측을 수행합니다."""
        try:
            # 모델이 없으면 로드 시도
            if self.model is None:
                if not self.load_model():
                    return np.array([0]), np.array([0.5])  # 기본값 반환
            
            # 데이터 전처리
            processed_df = self._preprocess_data(input_df)
            
            # 예측 수행
            with st.expander("디버그: 전처리된 데이터"):
                st.dataframe(processed_df)
                if hasattr(self.model, 'feature_names_in_'):
                    st.write("모델의 특성:", self.model.feature_names_in_)
                    st.write(f"모델 특성 수: {len(self.model.feature_names_in_)}")
                    st.write(f"입력 특성 수: {len(processed_df.columns)}")
                    
                    # 특성 매칭 확인
                    missing = set(self.model.feature_names_in_) - set(processed_df.columns)
                    extra = set(processed_df.columns) - set(self.model.feature_names_in_)
                    
                    if missing:
                        st.error(f"누락된 특성: {missing}")
                    if extra:
                        st.warning(f"추가 특성: {extra}")
            
            try:
                # 예측 및 확률 계산
                y_pred = self.model.predict(processed_df)
                y_proba = self.model.predict_proba(processed_df)[:, 1]  # 이탈 확률
                
                # 특성 중요도 계산
                self._compute_feature_importance(processed_df)
                
                return y_pred, y_proba
            except Exception as e:
                st.error(f"예측 오류: {str(e)}")
                return np.array([0]), np.array([0.5])  # 기본값 반환
                
        except Exception as e:
            st.error(f"전체 예측 과정 오류: {str(e)}")
            return np.array([0]), np.array([0.5])  # 기본값 반환
    
    def _preprocess_data(self, input_df):
        """입력 데이터를 전처리합니다. (19개 컬럼 -> 28개 컬럼)"""
        # 입력 데이터 복사
        df = input_df.copy()
        
        # 컬럼명을 모두 소문자로 변경
        df.columns = [col.lower() for col in df.columns]
        
        # CustomerID 제거 (예측에 사용되지 않음)
        columns_to_remove = ['customerid', 'customer_id', 'cust_id', 'id']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Boolean 타입 변환
        if 'complain' in df.columns and isinstance(df['complain'].iloc[0], str):
            df['complain'] = df['complain'].apply(lambda x: 1 if x == '예' else 0)
        
        # 원핫인코딩이 필요한 범주형 특성들
        categorical_features = {
            'preferred_login_device': ['mobile', 'computer'],
            'gender': ['male', 'female'],
            'preferred_payment_method': ['credit card', 'debit card', 'upi', 'cash on delivery'],
            'preferred_order_category': ['fashion', 'grocery', 'electronics', 'others'],
            'marital_status': ['single', 'married', 'divorced']
        }
        
        # 원핫인코딩 수행
        for feature, categories in categorical_features.items():
            if feature in df.columns:
                # 현재 값 (소문자로 변환)
                current_value = str(df[feature].iloc[0]).lower()
                
                # 원본 컬럼 제거
                df = df.drop(feature, axis=1)
                
                # 각 카테고리에 대한 더미 변수 생성
                for category in categories:
                    col_name = f"{feature}_{category}"
                    df[col_name] = 1 if current_value == category else 0
        
        # 모델에 필요한 컬럼 확인 및 조정
        if hasattr(self.model, 'feature_names_in_'):
            expected_columns = [col.lower() for col in self.model.feature_names_in_]
            
            # 누락된 컬럼 추가
            for col in expected_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # 불필요한 컬럼 제거
            for col in list(df.columns):
                if col not in expected_columns:
                    df = df.drop(col, axis=1)
            
            # 컬럼 순서 맞추기
            df = df[expected_columns]
        
        return df
    
    def _compute_feature_importance(self, input_data):
        """특성 중요도를 계산합니다."""
        try:
            if self.model is None:
                return None
                
            # 모델의 feature_importances_ 속성 이용
            if hasattr(self.model, 'feature_importances_'):
                importances = {}
                for i, feature in enumerate(self.model.feature_names_in_):
                    importances[feature] = float(self.model.feature_importances_[i])
                
                # 중요도 값으로 정렬
                self.feature_importance_cache = dict(sorted(
                    importances.items(), key=lambda x: x[1], reverse=True
                ))
            
            return self.feature_importance_cache
        except Exception as e:
            st.error(f"특성 중요도 계산 오류: {str(e)}")
            return None
    
    def get_feature_importance(self):
        """계산된 특성 중요도를 반환합니다."""
        return self.feature_importance_cache

# 게이지 차트 생성
def create_churn_gauge(value):
    """이탈 가능성 게이지 차트를 생성합니다."""
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
    """예측 페이지 표시"""
    # 애니메이션 적용
    add_page_transition()
    
    # 헤더 표시
    show_header()
    
    st.title("고객 이탈 예측")
    st.write("고객 정보를 입력하여 이탈 가능성을 예측해보세요.")
    
    # 원본 데이터셋의 컬럼명 및 설명
    columns = {
        'customer_id': '고객 ID',
        'tenure': '거래기간 (개월)',
        'preferred_login_device': '선호 로그인 기기',
        'city_tier': '도시 등급',
        'warehouse_to_home': '창고-집 거리 (km)',
        'preferred_payment_method': '선호 결제 방식',
        'gender': '성별',
        'hour_spend_on_app': '앱 사용 시간 (시간)',
        'number_of_device_registered': '등록된 기기 수',
        'preferred_order_category': '선호 주문 카테고리',
        'satisfaction_score': '만족도 점수 (1-5)',
        'marital_status': '결혼 상태',
        'number_of_address': '등록 주소 수',
        'complain': '불만 제기 여부',
        'order_amount_hike': '작년 대비 주문 금액 증가율 (%)',
        'coupon_used': '쿠폰 사용 횟수',
        'order_count': '주문 횟수',
        'days_since_last_order': '마지막 주문 후 경과일',
        'cashback_amount': '캐시백 금액 (원)'
    }
    
    # 필수 입력 필드
    required_columns = [
        'tenure', 
        'satisfaction_score',
        'order_count',
        'hour_spend_on_app',
        'days_since_last_order'
    ]
    
    # 입력값을 저장할 딕셔너리
    input_data = {}
    
    # 선택지가 있는 범주형 변수
    category_options = {
        'preferred_login_device': ['Mobile', 'Computer'],
        'gender': ['Male', 'Female'],
        'preferred_payment_method': ['Credit Card', 'Debit Card', 'UPI', 'Cash on Delivery'],
        'preferred_order_category': ['Fashion', 'Grocery', 'Electronics', 'Others'],
        'marital_status': ['Single', 'Married', 'Divorced'],
        'complain': ['예', '아니오']
    }
    
    # 기본값
    default_values = {
        'customer_id': f'CUST-{np.random.randint(10000, 99999)}',
        'tenure': 12,
        'preferred_login_device': 'Mobile',
        'city_tier': 1,
        'warehouse_to_home': 20,
        'preferred_payment_method': 'Credit Card',
        'gender': 'Male',
        'hour_spend_on_app': 3.0,
        'number_of_device_registered': 2,
        'preferred_order_category': 'Electronics',
        'satisfaction_score': 3,
        'marital_status': 'Single',
        'number_of_address': 2,
        'complain': '아니오',
        'order_amount_hike': 15.0,
        'coupon_used': 3,
        'order_count': 10,
        'days_since_last_order': 15,
        'cashback_amount': 150.0
    }
    
    # 입력 폼 생성
    with st.form("customer_form"):
        st.markdown("## 고객 정보 입력")
        
        # 필수/선택 입력 필드 구분
        tab1, tab2 = st.tabs(["필수 입력 필드", "선택 입력 필드"])
        
        with tab1:
            cols = st.columns(3)
            
            # 고객 ID
            with cols[0]:
                input_data['customer_id'] = st.text_input(
                    "고객 ID", 
                    value=default_values['customer_id'],
                    key="customer_id"
                )
            
            # 거래기간
            with cols[1]:
                input_data['tenure'] = st.number_input(
                    "거래기간 (개월) *", 
                    min_value=0, 
                    value=default_values['tenure'],
                    key="tenure"
                )
            
            # 만족도 점수
            with cols[2]:
                input_data['satisfaction_score'] = st.slider(
                    "만족도 점수 (1-5) *", 
                    min_value=1, 
                    max_value=5, 
                    value=default_values['satisfaction_score'],
                    key="satisfaction_score"
                )
            
            cols = st.columns(3)
            
            # 주문 횟수
            with cols[0]:
                input_data['order_count'] = st.number_input(
                    "주문 횟수 *", 
                    min_value=0, 
                    value=default_values['order_count'],
                    key="order_count"
                )
            
            # 앱 사용 시간
            with cols[1]:
                input_data['hour_spend_on_app'] = st.number_input(
                    "앱 사용 시간 (시간) *", 
                    min_value=0.0, 
                    step=0.1,
                    value=default_values['hour_spend_on_app'],
                    key="hour_spend_on_app"
                )
            
            # 마지막 주문 후 경과일
            with cols[2]:
                input_data['days_since_last_order'] = st.number_input(
                    "마지막 주문 후 경과일 *", 
                    min_value=0, 
                    value=default_values['days_since_last_order'],
                    key="days_since_last_order"
                )
        
        with tab2:
            cols = st.columns(3)
            
            # 선호 로그인 기기
            with cols[0]:
                input_data['preferred_login_device'] = st.selectbox(
                    "선호 로그인 기기", 
                    options=category_options['preferred_login_device'],
                    index=0,
                    key="preferred_login_device"
                )
            
            # 성별
            with cols[1]:
                input_data['gender'] = st.selectbox(
                    "성별", 
                    options=category_options['gender'],
                    index=0,
                    key="gender"
                )
            
            # 결혼 상태
            with cols[2]:
                input_data['marital_status'] = st.selectbox(
                    "결혼 상태", 
                    options=category_options['marital_status'],
                    index=0,
                    key="marital_status"
                )
            
            cols = st.columns(3)
            
            # 도시 등급
            with cols[0]:
                input_data['city_tier'] = st.number_input(
                    "도시 등급", 
                    min_value=1, 
                    max_value=3, 
                    value=default_values['city_tier'],
                    key="city_tier"
                )
            
            # 창고-집 거리
            with cols[1]:
                input_data['warehouse_to_home'] = st.number_input(
                    "창고-집 거리 (km)", 
                    min_value=0, 
                    value=default_values['warehouse_to_home'],
                    key="warehouse_to_home"
                )
            
            # 등록된 기기 수
            with cols[2]:
                input_data['number_of_device_registered'] = st.number_input(
                    "등록된 기기 수", 
                    min_value=1, 
                    value=default_values['number_of_device_registered'],
                    key="number_of_device_registered"
                )
            
            cols = st.columns(3)
            
            # 선호 결제 방식
            with cols[0]:
                input_data['preferred_payment_method'] = st.selectbox(
                    "선호 결제 방식", 
                    options=category_options['preferred_payment_method'],
                    index=0,
                    key="preferred_payment_method"
                )
            
            # 선호 주문 카테고리
            with cols[1]:
                input_data['preferred_order_category'] = st.selectbox(
                    "선호 주문 카테고리", 
                    options=category_options['preferred_order_category'],
                    index=0,
                    key="preferred_order_category"
                )
            
            # 불만 제기 여부
            with cols[2]:
                input_data['complain'] = st.selectbox(
                    "불만 제기 여부", 
                    options=category_options['complain'],
                    index=1,  # 기본값 '아니오'
                    key="complain"
                )
            
            cols = st.columns(3)
            
            # 등록 주소 수
            with cols[0]:
                input_data['number_of_address'] = st.number_input(
                    "등록 주소 수", 
                    min_value=0, 
                    value=default_values['number_of_address'],
                    key="number_of_address"
                )
            
            # 작년 대비 주문 금액 증가율
            with cols[1]:
                input_data['order_amount_hike'] = st.number_input(
                    "작년 대비 주문 금액 증가율 (%)", 
                    min_value=-100.0, 
                    value=default_values['order_amount_hike'],
                    step=0.1,
                    key="order_amount_hike"
                )
            
            # 쿠폰 사용 횟수
            with cols[2]:
                input_data['coupon_used'] = st.number_input(
                    "쿠폰 사용 횟수", 
                    min_value=0, 
                    value=default_values['coupon_used'],
                    key="coupon_used"
                )
            
            # 캐시백 금액
            input_data['cashback_amount'] = st.number_input(
                "캐시백 금액 (원)", 
                min_value=0.0, 
                value=default_values['cashback_amount'],
                step=0.1,
                key="cashback_amount"
            )
        
        # 예측 버튼
        submit = st.form_submit_button("이탈 예측하기", use_container_width=True)
    
    # 예측 수행
    if submit:
        # 필수 필드 검증
        missing_fields = [columns[col] for col in required_columns if not input_data.get(col)]
        
        if missing_fields:
            st.error(f"다음 필수 입력 항목을 입력해주세요: {', '.join(missing_fields)}")
        else:
            # 로딩 표시
            with st.spinner("예측 중..."):
                try:
                    # 입력 데이터를 DataFrame으로 변환
                    input_df = pd.DataFrame([input_data])
                    
                    # 고객 ID 표시
                    st.markdown(f"### 고객 ID: {input_data['customer_id']}")
                    
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
                    
                    # 영향 요인 계산
                    feature_importance = predictor.get_feature_importance()
                    
                    if feature_importance:
                        # 원핫인코딩 피처 변환 (예: preferred_login_device_mobile -> 선호 로그인 기기: Mobile)
                        readable_features = {}
                        for feature, importance in feature_importance.items():
                            if '_' in feature:
                                # 원핫인코딩된 특성인 경우
                                parts = feature.split('_')
                                prefix = '_'.join(parts[:-1])  # 접두사 부분
                                value = parts[-1]  # 값 부분
                                
                                # 원본 컬럼 이름 찾기
                                for col, desc in columns.items():
                                    if col == prefix:
                                        readable_features[f"{desc}: {value.capitalize()}"] = importance
                                        break
                                else:
                                    readable_features[feature] = importance
                            else:
                                # 일반 특성인 경우
                                for col, desc in columns.items():
                                    if col == feature:
                                        readable_features[desc] = importance
                                        break
                                else:
                                    readable_features[feature] = importance
                        
                        # 상위 5개 특성 추출
                        top_features = dict(list(readable_features.items())[:5])
                        
                        # 바 차트로 시각화
                        fig = px.bar(
                            x=list(top_features.values()),
                            y=list(top_features.keys()),
                            orientation='h',
                            title="주요 영향 요인 (Top 5)",
                            labels={'x': '중요도', 'y': '특성'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 주요 피처별 설명
                        st.write("### 영향 요인 분석")
                        for feature, importance in list(top_features.items())[:3]:
                            st.write(f"**{feature}** (중요도: {importance:.4f})")
                            
                            # 특성별 설명
                            if "거래기간" in feature:
                                st.write("거래기간이 길수록 이탈 확률이 낮아지는 경향이 있습니다.")
                            elif "앱 사용 시간" in feature:
                                st.write("앱 사용 시간이 길수록 이탈 확률이 낮아지는 경향이 있습니다.")
                            elif "만족도" in feature:
                                st.write("만족도 점수가 높을수록 이탈 확률이 낮아지는 경향이 있습니다.")
                            elif "주문 횟수" in feature:
                                st.write("주문 횟수가 많을수록 이탈 확률이 낮아지는 경향이 있습니다.")
                            elif "마지막 주문" in feature:
                                st.write("마지막 주문 이후 시간이 길수록 이탈 확률이 높아지는 경향이 있습니다.")
                            elif "불만" in feature:
                                st.write("불만을 제기한 고객은 이탈 확률이 높아지는 경향이 있습니다.")
                            else:
                                st.write("이 특성은 고객의 이탈 가능성에 영향을 미치는 주요 요인입니다.")
                    
                    # 디버그 정보
                    with st.expander("🔧 디버그 정보"):
                        debug_tabs = st.tabs(["입력 데이터", "모델 정보", "특성 중요도"])
                        
                        with debug_tabs[0]:
                            st.write("### 원본 입력 데이터")
                            st.dataframe(input_df)
                        
                        with debug_tabs[1]:
                            st.write("### 모델 정보")
                            st.write(f"**모델 경로:** {predictor.model_path}")
                            st.write(f"**모델 로드 상태:** {'성공' if predictor.model is not None else '실패'}")
                            
                            if hasattr(predictor.model, 'feature_names_in_'):
                                st.write("### 모델 특성 목록")
                                st.write(f"특성 수: {len(predictor.model.feature_names_in_)}")
                                st.write(sorted(predictor.model.feature_names_in_))
                        
                        with debug_tabs[2]:
                            st.write("### 모든 특성 중요도")
                            if feature_importance:
                                importance_df = pd.DataFrame({
                                    '특성': list(feature_importance.keys()),
                                    '중요도': list(feature_importance.values())
                                })
                                st.dataframe(importance_df)
                            else:
                                st.write("특성 중요도 정보가 없습니다.")
                
                except Exception as e:
                    st.error(f"예측 중 오류가 발생했습니다: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

# 메인 함수 호출
if __name__ == "__main__":
    show() 