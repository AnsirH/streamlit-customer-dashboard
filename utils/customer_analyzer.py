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
    
    def _interpret_feature(self, feature_str):
        """
        영향 요인 문자열을 해석하여 직관적인 설명을 반환합니다.
        
        Args:
            feature_str (str): "특성명 (방향)" 형식의 문자열
            
        Returns:
            str: 해석된 설명
        """
        try:
            # 특성명과 방향 분리
            if ' (' not in feature_str:
                return feature_str
                
            feature_name = feature_str.split(' (')[0].strip()
            direction = feature_str.split(' (')[1].replace(')', '').strip()
            
            # 해석 규칙 정의
            interpretations = {
                '불만 제기': '높은 불만 제기',
                '만족도 점수': '낮은 만족도' if '부정' in direction else '높은 만족도',
                '마지막 주문 후 경과일': '장기간 주문 없음',
                '주문 횟수': '낮은 주문 빈도' if '부정' in direction else '높은 주문 빈도',
                '캐시백 금액': '낮은 캐시백 사용' if '부정' in direction else '높은 캐시백 사용',
                '앱 사용 시간': '낮은 앱 사용 시간' if '부정' in direction else '높은 앱 사용 시간',
                '거래 기간': '짧은 거래 기간' if '부정' in direction else '긴 거래 기간',
                '작년 대비 주문 증가율': '낮은 주문 증가율' if '부정' in direction else '높은 주문 증가율',
                '쿠폰 사용 횟수': '낮은 쿠폰 사용' if '부정' in direction else '높은 쿠폰 사용',
                '배송 거리': '먼 배송 거리' if '부정' in direction else '가까운 배송 거리',
                '선호 로그인 기기': '비선호 로그인 기기 사용' if '부정' in direction else '선호 로그인 기기 사용',
                '선호 결제 수단': '비선호 결제 수단 사용' if '부정' in direction else '선호 결제 수단 사용',
                '성별': '성별 관련 이슈',
                '선호 주문 카테고리': '비선호 카테고리 주문' if '부정' in direction else '선호 카테고리 주문',
                '결혼 여부': '결혼 상태 관련 이슈',
                '도시 등급': '도시 등급 관련 이슈',
                '주소 개수': '적은 주소 등록' if '부정' in direction else '많은 주소 등록',
                '등록된 기기 수': '적은 기기 등록' if '부정' in direction else '많은 기기 등록'
            }
            
            # 해석된 설명 반환
            return interpretations.get(feature_name, feature_name)
            
        except Exception as e:
            print(f"특성 해석 중 오류 발생: {str(e)}")
            return feature_str

    def get_top_issues(self, customer_id):
        """
        고객의 상위 3개 이탈 요인을 반환합니다.
        
        Args:
            customer_id (int): 고객 ID
            
        Returns:
            list: 상위 3개 이탈 요인 리스트
        """
        try:
            # analyze_customers() 결과 가져오기
            from models.customer_analyzer import analyze_customers
            analysis_results = analyze_customers()
            
            # 해당 고객의 데이터 찾기
            customer_data = analysis_results[analysis_results['CustomerID'] == customer_id]
            
            if customer_data.empty:
                return ["데이터를 찾을 수 없습니다."]
            
            # 상위 3개 영향 요인 추출
            top_issues = []
            for i in range(1, 4):
                feature_col = f'Top Feature {i}'
                
                if feature_col in customer_data.columns:
                    feature = customer_data.iloc[0][feature_col]
                    
                    if pd.notna(feature):  # None이나 NaN이 아닌 경우에만 추가
                        top_issues.append(feature)
            
            return top_issues if top_issues else ["이탈 요인이 없습니다."]
            
        except Exception as e:
            print(f"이탈 요인 분석 중 오류 발생: {str(e)}")
            return ["분석 중 오류가 발생했습니다."]

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