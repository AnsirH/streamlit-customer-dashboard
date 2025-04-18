import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from utils.visualizer import Visualizer
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

class CustomerAnalyzer:
    """고객 분석을 위한 클래스"""
    
    def __init__(self):
        self.df = None
        self.model = None
        self.visualizer = Visualizer()
        self.feature_importance_cache = None
        
        try:
            # 모델 파일 경로 설정
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = Path(current_dir).parent / "models"
            
            # 모델 파일 경로
            model_path = models_dir / "xgboost_best_model.pkl"
            st.write(f"🔍 디버그: 모델 파일 경로 - {model_path}")
            st.write(f"🔍 디버그: 모델 파일 존재 여부 - {model_path.exists()}")
            
            # 모델 로드
            if model_path.exists():
                st.write("🔍 디버그: 모델 파일 로드 시도 중...")
                try:
                    # pickle로 시도
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    st.write("🔍 디버그: pickle로 모델 로드 성공")
                except Exception as e:
                    st.write(f"🔍 디버그: pickle 로드 실패 - {str(e)}")
                    try:
                        # joblib로 시도
                        self.model = joblib.load(model_path)
                        st.write("🔍 디버그: joblib로 모델 로드 성공")
                    except Exception as e:
                        st.write(f"🔍 디버그: joblib 로드 실패 - {str(e)}")
                        st.error(f"🔍 디버그: 모델 파일이 손상되었거나 올바른 형식이 아닙니다.")
                        st.error(f"🔍 디버그: 모델 파일을 다시 생성하거나 올바른 파일을 사용해주세요.")
                        self.model = None
                
                if self.model is not None:
                    # 모델 정보 출력
                    st.write(f"🔍 디버그: 모델 타입 - {type(self.model)}")
                    if hasattr(self.model, 'feature_importances_'):
                        st.write(f"🔍 디버그: 모델 특성 수 - {len(self.model.feature_importances_)}")
            else:
                st.error(f"🔍 디버그: 모델 파일을 찾을 수 없습니다. 경로: {model_path}")
                st.error(f"🔍 디버그: models 디렉토리 내용:")
                if models_dir.exists():
                    for file in models_dir.iterdir():
                        st.write(f"- {file.name}")
                else:
                    st.error("models 디렉토리가 존재하지 않습니다.")
        except Exception as e:
            st.error(f"🔍 디버그: 모델 로드 중 오류 발생")
            st.error(f"🔍 디버그: 오류 타입 - {type(e).__name__}")
            st.error(f"🔍 디버그: 오류 메시지 - {str(e)}")
            st.error(f"🔍 디버그: 현재 작업 디렉토리 - {os.getcwd()}")
            st.error(f"🔍 디버그: 파일 경로 - {os.path.abspath(__file__)}")
    
    def load_data(self):
        """고객 데이터를 로드하고 전처리합니다."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = Path(current_dir).parent / "models"
            file_path = models_dir / "E Commerce Dataset2.xlsx"
            
            st.write(f"🔍 디버그: 데이터 파일 경로 - {file_path}")
            st.write(f"🔍 디버그: 데이터 파일 존재 여부 - {file_path.exists()}")
            
            if not file_path.exists():
                st.error(f"🔍 디버그: 데이터 파일을 찾을 수 없습니다. 경로: {file_path}")
                st.error(f"🔍 디버그: models 디렉토리 내용:")
                if models_dir.exists():
                    for file in models_dir.iterdir():
                        st.write(f"- {file.name}")
                else:
                    st.error("models 디렉토리가 존재하지 않습니다.")
                return False
            
            # 데이터 로드
            df = pd.read_excel(file_path)
            
            # 결측치 처리
            # 수치형 컬럼의 결측치는 중앙값으로 대체
            numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            for col in numeric_columns:
                df[col] = df[col].fillna(df[col].median())
            
            # 범주형 컬럼의 결측치는 최빈값으로 대체
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0])
            
            # 결측치 처리 후 데이터 저장
            self.df = df
            
            st.write(f"🔍 디버그: 데이터 로드 성공 - {len(self.df)}행")
            return True
            
        except Exception as e:
            st.error(f"🔍 디버그: 데이터 로드 중 오류 발생")
            st.error(f"🔍 디버그: 오류 타입 - {type(e).__name__}")
            st.error(f"🔍 디버그: 오류 메시지 - {str(e)}")
            return False
    
    def predict(self, input_data, debug=True):
        """입력 데이터에 대한 이탈 확률을 예측합니다."""
        if self.model is None:
            if debug:
                st.error("모델이 로드되지 않았습니다.")
            return None
        
        try:
            # 데이터 전처리
            processed_data = self._preprocess_data(input_data)
            
            # 필요한 특성만 선택 (28개 특성)
            required_features = [
                'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                'OrderCount', 'DaySinceLastOrder', 'CashbackAmount',
                'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone',
                'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery',
                'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
                'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI',
                'Gender_Male',
                'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
                'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone',
                'MaritalStatus_Married', 'MaritalStatus_Single'
            ]
            
            # 누락된 특성이 있다면 0으로 채움
            for feature in required_features:
                if feature not in processed_data.columns:
                    processed_data[feature] = 0
            
            # 특성 순서 맞추기
            processed_data = processed_data[required_features]
            
            # 예측 수행
            churn_prob = self.model.predict_proba(processed_data)[:, 1]
            
            # 디버깅 정보 출력
            if debug:
                st.write(f"🔍 디버그: 예측 데이터 shape - {processed_data.shape}")
                st.write(f"🔍 디버그: 예측된 이탈 확률 - {churn_prob[0]:.2%}")
            
            return float(churn_prob[0])  # 단일 고객에 대한 예측이므로 첫 번째 값 반환
            
        except Exception as e:
            if debug:
                st.error(f"예측 중 오류 발생: {str(e)}")
                st.error(f"🔍 디버그: 처리된 데이터 컬럼 - {processed_data.columns.tolist()}")
            return None
    
    def analyze_customer(self, customer_id):
        """특정 고객의 데이터를 분석합니다."""
        if self.df is None:
            st.warning("데이터를 먼저 로드해주세요.")
            return {'customer_data': None, 'churn_prob': None}
        
        try:
            # 고객 데이터 조회
            customer_data = self.df[self.df['CustomerID'] == customer_id]
            if customer_data.empty:
                st.error(f"고객 ID {customer_id}를 찾을 수 없습니다.")
                return {'customer_data': None, 'churn_prob': None}
            
            # 데이터 전처리
            processed_data = self._preprocess_data(customer_data)
            
            # 이탈 예측
            churn_prob = self.predict(processed_data)
            
            return {
                'customer_data': customer_data,
                'churn_prob': churn_prob
            }
        except Exception as e:
            st.error(f"고객 분석 중 오류 발생: {str(e)}")
            return {'customer_data': None, 'churn_prob': None}
    
    def _preprocess_data(self, input_df):
        """입력 데이터를 전처리합니다."""
        try:
            # 입력 데이터 복사
            df = input_df.copy()
            
            # 결측치를 먼저 처리
            df = df.fillna(0)
            
            # CustomerID 제거 (예측에 사용되지 않음)
            columns_to_remove = ['CustomerID', 'customer_id', 'customerid', 'cust_id', 'id', 'Churn']
            for col in columns_to_remove:
                if col in df.columns:
                    df = df.drop(col, axis=1)
            
            # 수치형 컬럼들을 먼저 정수로 변환
            numeric_columns = [
                'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                'OrderAmountHikeFromlastYear', 'CouponUsed',
                'OrderCount', 'DaySinceLastOrder', 'CashbackAmount'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
            # Complain 불리언 변환
            if 'Complain' in df.columns:
                if df['Complain'].dtype == 'object':
                    df['Complain'] = df['Complain'].apply(lambda x: 1 if str(x).lower() in ['예', 'yes', '1', 'true'] else 0)
                df['Complain'] = pd.to_numeric(df['Complain'], errors='coerce').fillna(0).astype(int)
            
            # PreferredLoginDevice 원핫인코딩
            if 'PreferredLoginDevice' in df.columns:
                # 값 표준화
                df['PreferredLoginDevice'] = df['PreferredLoginDevice'].str.strip()
                df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({
                    'Mobile': 'Mobile Phone',
                    'Phone': 'Mobile Phone'
                })
                
                # 원핫인코딩
                df['PreferredLoginDevice_Mobile Phone'] = (df['PreferredLoginDevice'] == 'Mobile Phone').astype(int)
                df['PreferredLoginDevice_Phone'] = (df['PreferredLoginDevice'] == 'Phone').astype(int)
                df = df.drop('PreferredLoginDevice', axis=1)
            else:
                df['PreferredLoginDevice_Mobile Phone'] = 0
                df['PreferredLoginDevice_Phone'] = 0
            
            # PreferredPaymentMode 원핫인코딩
            payment_modes = ['COD', 'Cash on Delivery', 'Credit Card', 'Debit Card', 'E wallet', 'UPI']
            if 'PreferredPaymentMode' in df.columns:
                for mode in payment_modes:
                    df[f'PreferredPaymentMode_{mode}'] = (df['PreferredPaymentMode'] == mode).astype(int)
                df = df.drop('PreferredPaymentMode', axis=1)
            else:
                for mode in payment_modes:
                    df[f'PreferredPaymentMode_{mode}'] = 0
            
            # Gender 원핫인코딩
            if 'Gender' in df.columns:
                df['Gender_Male'] = (df['Gender'].isin(['M', 'Male'])).astype(int)
                df = df.drop('Gender', axis=1)
            else:
                df['Gender_Male'] = 0
            
            # PreferedOrderCat 원핫인코딩
            order_cats = ['Grocery', 'Laptop & Accessory', 'Mobile', 'Mobile Phone']
            if 'PreferedOrderCat' in df.columns:
                for cat in order_cats:
                    df[f'PreferedOrderCat_{cat}'] = (df['PreferedOrderCat'] == cat).astype(int)
                df = df.drop('PreferedOrderCat', axis=1)
            else:
                for cat in order_cats:
                    df[f'PreferedOrderCat_{cat}'] = 0
            
            # MaritalStatus 원핫인코딩
            marital_statuses = ['Married', 'Single']
            if 'MaritalStatus' in df.columns:
                for status in marital_statuses:
                    df[f'MaritalStatus_{status}'] = (df['MaritalStatus'] == status).astype(int)
                df = df.drop('MaritalStatus', axis=1)
            else:
                for status in marital_statuses:
                    df[f'MaritalStatus_{status}'] = 0
            
            # 필수 컬럼 확인 (모델이 사용하는 28개 특성)
            required_columns = [
                # 수치형 특성 (13개)
                'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                'OrderCount', 'DaySinceLastOrder', 'CashbackAmount',
                # 원핫인코딩된 특성 (15개)
                'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone',
                'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery',
                'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
                'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI',
                'Gender_Male',
                'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
                'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone',
                'MaritalStatus_Married', 'MaritalStatus_Single'
            ]
            
            # 누락된 컬럼 처리
            for col in required_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # 최종 결측치 처리
            df = df.fillna(0)
            
            # 필요한 컬럼만 선택하여 반환
            result_df = df[required_columns]
            
            # 모든 컬럼이 정수형인지 확인하고 변환
            for col in result_df.columns:
                result_df[col] = pd.to_numeric(result_df[col], errors='coerce').fillna(0).astype(int)
            
            return result_df
            
        except Exception as e:
            st.error(f"데이터 전처리 중 오류: {str(e)}")
            raise e
    
    def get_customer_list(self):
        """모든 고객 ID 목록을 반환합니다."""
        if self.df is None:
            if not self.load_data():
                return []
        return self.df['CustomerID'].tolist()
    
    def get_customer_ids(self):
        """get_customer_list의 별칭 메서드"""
        return self.get_customer_list()

    def get_feature_importance(self):
        """특성 중요도를 반환합니다."""
        if self.feature_importance_cache is None:
            self._compute_feature_importance()
        return self.feature_importance_cache

    def _compute_feature_importance(self):
        """특성 중요도를 계산합니다."""
        try:
            if self.model is None:
                st.error("모델이 로드되지 않았습니다.")
                return None
            
            if not hasattr(self.model, 'feature_importances_'):
                st.error("모델이 특성 중요도를 지원하지 않습니다.")
                return None
            
            # 특성 중요도 계산
            importance = self.model.feature_importances_
            
            # 특성 이름과 중요도 매핑
            feature_names = [
                # 수치형 특성 (13개)
                'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                'OrderCount', 'DaySinceLastOrder', 'CashbackAmount',
                # 원핫인코딩된 특성 (15개)
                'PreferredLoginDevice_Mobile Phone', 'PreferredLoginDevice_Phone',
                'PreferredPaymentMode_COD', 'PreferredPaymentMode_Cash on Delivery',
                'PreferredPaymentMode_Credit Card', 'PreferredPaymentMode_Debit Card',
                'PreferredPaymentMode_E wallet', 'PreferredPaymentMode_UPI',
                'Gender_Male',
                'PreferedOrderCat_Grocery', 'PreferedOrderCat_Laptop & Accessory',
                'PreferedOrderCat_Mobile', 'PreferedOrderCat_Mobile Phone',
                'MaritalStatus_Married', 'MaritalStatus_Single'
            ]
            
            self.feature_importance_cache = pd.Series(importance, index=feature_names)
            return self.feature_importance_cache
            
        except Exception as e:
            st.error(f"특성 중요도 계산 중 오류: {str(e)}")
            return None

    def get_top_issues(self, customer_id):
        """고객의 상위 3개 이탈 요인을 반환합니다."""
        try:
            # 고객 데이터 가져오기
            customer_data = self.df[self.df['CustomerID'] == customer_id].iloc[0]
            
            # 특성별 점수 계산
            scores = {}
            
            # 마지막 주문 경과일
            if customer_data['DaySinceLastOrder'] > 7:
                scores['장기간 주문 없음'] = customer_data['DaySinceLastOrder'] / 30
            
            # 만족도
            if customer_data['SatisfactionScore'] < 3:
                scores['낮은 만족도'] = (3 - customer_data['SatisfactionScore']) / 3
            
            # 불만 제기
            if customer_data['Complain'] == 1:
                scores['불만 제기 이력'] = 1.0
            
            # 주문 횟수
            avg_order_count = self.df['OrderCount'].mean()
            if customer_data['OrderCount'] < avg_order_count:
                scores['낮은 주문 빈도'] = (avg_order_count - customer_data['OrderCount']) / avg_order_count
            
            # 캐시백 사용
            avg_cashback = self.df['CashbackAmount'].mean()
            if customer_data['CashbackAmount'] < avg_cashback:
                scores['낮은 캐시백 사용'] = (avg_cashback - customer_data['CashbackAmount']) / avg_cashback
            
            # 앱 사용 시간
            avg_app_hours = self.df['HourSpendOnApp'].mean()
            if customer_data['HourSpendOnApp'] < avg_app_hours:
                scores['낮은 앱 사용 시간'] = (avg_app_hours - customer_data['HourSpendOnApp']) / avg_app_hours
            
            # 거래 기간
            avg_tenure = self.df['Tenure'].mean()
            if customer_data['Tenure'] < avg_tenure:
                scores['짧은 거래 기간'] = (avg_tenure - customer_data['Tenure']) / avg_tenure
            
            # 주문 금액 증가율
            if customer_data['OrderAmountHikeFromlastYear'] < 10:
                scores['낮은 주문 금액 증가율'] = (10 - customer_data['OrderAmountHikeFromlastYear']) / 10
            
            # 쿠폰 사용
            avg_coupon = self.df['CouponUsed'].mean()
            if customer_data['CouponUsed'] < avg_coupon:
                scores['낮은 쿠폰 사용'] = (avg_coupon - customer_data['CouponUsed']) / avg_coupon
            
            # 점수가 높은 순으로 정렬하여 상위 3개 선택
            sorted_issues = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # 이슈 이름만 반환
            return [issue[0] for issue in sorted_issues]
            
        except Exception as e:
            st.error(f"이탈 요인 분석 중 오류 발생: {str(e)}")
            return []

    def get_customer_insights(self, customer_id):
        """고객에 대한 인사이트를 반환합니다."""
        analysis = self.analyze_customer(customer_id)
        customer_data = analysis['customer_data']
        churn_prob = analysis['churn_prob']
        
        insights = {
            'churn_risk': '높음' if churn_prob >= 0.7 else ('중간' if churn_prob >= 0.3 else '낮음'),
            'key_factors': self._get_key_factors(customer_data),
            'recommendations': self._get_recommendations(customer_data, churn_prob)
        }
        
        return insights

    def _get_key_factors(self, customer_data):
        """주요 이탈 요인을 반환합니다."""
        factors = []
        if customer_data['DaySinceLastOrder'] > 7:
            factors.append('장기간 주문 없음')
        if customer_data['SatisfactionScore'] < 3:
            factors.append('낮은 만족도')
        if customer_data['Complain'] == 1:
            factors.append('불만 제기 이력')
        return factors

    def _get_recommendations(self, customer_data, churn_prob):
        """개선 방안을 반환합니다."""
        recommendations = []
        if churn_prob >= 0.7:
            recommendations.extend([
                '개인화된 할인 쿠폰 발송',
                '전담 상담원 배정',
                'VIP 혜택 제공'
            ])
        elif churn_prob >= 0.3:
            recommendations.extend([
                '관심 상품 재입고 알림',
                '맞춤형 추천 상품 제공',
                '로열티 포인트 추가 적립'
            ])
        else:
            recommendations.extend([
                '정기적인 만족도 조사',
                '신규 상품 소개',
                '기존 혜택 유지'
            ])
        return recommendations

    def analyze_last_order_days(self):
        """DaySinceLastOrder 컬럼의 통계 정보를 분석합니다."""
        try:
            if self.df is None:
                st.warning("데이터를 먼저 로드해주세요.")
                return
            
            # DaySinceLastOrder 컬럼의 통계 정보
            stats = self.df['DaySinceLastOrder'].describe()
            
            # 30일 이상인 데이터 개수
            over_30_days = len(self.df[self.df['DaySinceLastOrder'] >= 30])
            total_customers = len(self.df)
            percentage = (over_30_days / total_customers) * 100
            
            # 결과 출력
            st.write("### 마지막 주문 경과일 분석")
            st.write(f"- 최소: {stats['min']}일")
            st.write(f"- 최대: {stats['max']}일")
            st.write(f"- 평균: {stats['mean']:.2f}일")
            st.write(f"- 중앙값: {stats['50%']}일")
            st.write(f"- 30일 이상 고객 수: {over_30_days}명 ({percentage:.2f}%)")
            
            # 히스토그램 시각화
            fig = px.histogram(self.df, x='DaySinceLastOrder', 
                             title='마지막 주문 경과일 분포',
                             labels={'DaySinceLastOrder': '경과일', 'count': '고객 수'})
            fig.add_vline(x=30, line_dash="dash", line_color="red", 
                         annotation_text="30일 기준선", annotation_position="top right")
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"데이터 분석 중 오류 발생: {str(e)}") 