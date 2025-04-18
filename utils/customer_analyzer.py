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
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = Path(current_dir).parent / "models"
            model_path = models_dir / "xgboost_best_model.pkl"
            
            if model_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                except Exception:
                    try:
                        self.model = joblib.load(model_path)
                    except Exception:
                        self.model = None
        except Exception:
            self.model = None
    
    @staticmethod
    def generate_customer_ids(df):
        """데이터프레임에 CustomerID가 없는 경우 생성합니다.
        
        Args:
            df (pandas.DataFrame): CustomerID를 생성할 데이터프레임
            
        Returns:
            pandas.DataFrame: CustomerID가 추가된 데이터프레임
        """
        if 'CustomerID' not in df.columns:
            df['CustomerID'] = [f'CUST_{i:06d}' for i in range(1, len(df) + 1)]
        return df

    def load_data(self):
        """고객 데이터를 로드합니다."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = Path(current_dir).parent / "models"
            file_path = models_dir / "E Commerce Dataset2.xlsx"
            
            if not file_path.exists():
                return False
            
            # 데이터 로드
            df = pd.read_excel(file_path)
            
            # CustomerID 생성
            df = self.generate_customer_ids(df)
            
            self.df = df
            return True
            
        except Exception:
            return False
    
    def predict(self, input_data):
        """입력 데이터에 대한 이탈 확률을 예측합니다."""
        if self.model is None:
            return None
        
        try:
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
            
            # 특성 순서 맞추기
            input_data = input_data[required_features]
            
            # 예측 수행
            churn_prob = self.model.predict_proba(input_data)[:, 1]
            return float(churn_prob[0])
            
        except Exception:
            return None
    
    def analyze_customer(self, customer_id):
        """특정 고객의 데이터를 분석합니다."""
        if self.df is None:
            return {'customer_data': None, 'churn_prob': None}
        
        try:
            # 고객 데이터 조회
            customer_data = self.df[self.df['CustomerID'] == customer_id]
            if customer_data.empty:
                return {'customer_data': None, 'churn_prob': None}
            
            # 이탈 예측
            churn_prob = self.predict(customer_data)
            
            return {
                'customer_data': customer_data,
                'churn_prob': churn_prob
            }
        except Exception:
            return {'customer_data': None, 'churn_prob': None}
    
    def get_customer_list(self):
        """모든 고객 ID 목록을 반환합니다."""
        if self.df is None:
            if not self.load_data():
                return []
        return self.df['CustomerID'].tolist()
    
    def get_feature_importance(self):
        """특성 중요도를 반환합니다."""
        if self.feature_importance_cache is None:
            self._compute_feature_importance()
        return self.feature_importance_cache

    def _compute_feature_importance(self):
        """특성 중요도를 계산합니다."""
        try:
            if self.model is None:
                return None
            
            if not hasattr(self.model, 'feature_importances_'):
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
            
        except Exception:
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
            
        except Exception:
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
            
        except Exception:
            pass 