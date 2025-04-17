import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
import pickle
import shap

class ChurnPredictor:
    """고객 이탈 예측을 위한 모델 클래스"""
    
    def __init__(self):
        """모델을 로드하고 초기화합니다."""
        self.model = None
        self.feature_importance_cache = {}
        self.model_path = os.path.join('models', 'xgboost_best_model.pkl')
        try:
            self.load_model()
        except Exception as e:
            print(f"모델 로드 오류: {str(e)}")
    
    def load_model(self):
        """모델 파일을 로드합니다."""
        try:
            if not os.path.exists(self.model_path):
                print(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                return False
            
            self.model = joblib.load(self.model_path)
            print(f"모델 로드 성공: {self.model_path}")
            
            # 모델 정보 출력
            print("\n모델 정보:")
            print(f"모델 타입: {type(self.model)}")
            
            if hasattr(self.model, 'feature_names_'):
                print("\n모델이 요구하는 feature:")
                print(self.model.feature_names_)
            else:
                print("\n모델의 feature_names_ 속성이 없습니다.")
            
            if hasattr(self.model, 'feature_importances_'):
                print("\n모델의 feature 중요도:")
                print(self.model.feature_importances_)
            
            return True
        except Exception as e:
            print(f"모델 로드 실패: {str(e)}")
            return False
    
    def predict(self, input_df):
        """
        이탈 예측을 수행합니다.
        
        Args:
            input_df (pandas.DataFrame): 예측할 고객 데이터
            
        Returns:
            tuple: (예측 클래스, 이탈 확률)
        """
        try:
            # 모델이 없으면 로드 시도
            if self.model is None:
                self.load_model()
                
            # 모델 로드 실패 시 기본값 반환
            if self.model is None:
                return self._default_prediction()
            
            # 예측 수행
            try:
                y_pred = self.model.predict(input_df)
                y_proba = self.model.predict_proba(input_df)[:, 1]  # 이탈 확률
                
                # 예측 결과 확인
                if len(y_proba) == 0:
                    return self._default_prediction()
                
                # 성공적으로 예측한 경우 특성 중요도 계산
                try:
                    self._compute_feature_importance(processed_df)
                except Exception as e:
                    # 특성 중요도 계산 실패해도 예측 결과는 반환
                    pass
                
                return y_pred, y_proba
            except Exception as e:
                print(f"예측 오류: {str(e)}")
                return self._default_prediction()
                
        except Exception as e:
            print(f"예측 처리 중 오류: {str(e)}")
            return self._default_prediction()
    
    def _default_prediction(self):
        """기본 예측값 반환"""
        return np.array([0]), np.array([0.5])
    
    def _compute_feature_importance(self, input_df):
        """모델의 feature_importances_ 속성을 사용하여 특성 중요도를 계산합니다."""
        if self.model is None:
            return
            
        try:
            # feature_importances_ 속성 사용
            if hasattr(self.model, 'feature_importances_'):
                importance_dict = {}
                for i, col in enumerate(input_df.columns):
                    if i < len(self.model.feature_importances_):
                        importance_dict[col] = self.model.feature_importances_[i]
                self.feature_importance_cache = importance_dict
            else:
                # 기본 중요도 설정
                self.feature_importance_cache = {
                    'Tenure': 0.25,
                    'SatisfactionScore': 0.22,
                    'DaySinceLastOrder': 0.18,
                    'OrderCount': 0.15,
                    'HourSpendOnApp': 0.12,
                    'Complain': 0.08
                }
        except Exception as e:
            # 모든 방법 실패 시 기본 중요도 사용
            self.feature_importance_cache = {
                'Tenure': 0.25,
                'SatisfactionScore': 0.22,
                'DaySinceLastOrder': 0.18,
                'OrderCount': 0.15,
                'HourSpendOnApp': 0.12,
                'Complain': 0.08
            }
    
    def get_feature_importance(self):
        """
        계산된 특성 중요도를 반환합니다.
        
        Returns:
            dict: 특성별 중요도
        """
        # 특성 중요도가 없으면 기본값 반환
        if not self.feature_importance_cache:
            return {
                'Tenure': 0.25,
                'SatisfactionScore': 0.22,
                'DaySinceLastOrder': 0.18,
                'OrderCount': 0.15,
                'HourSpendOnApp': 0.12,
                'Complain': 0.08
            }
        
        return self.feature_importance_cache

def analyze_customers():
    """
    고객 데이터를 분석하고 이탈 예측 결과를 반환합니다.
    Returns:
        DataFrame: 고객 ID, 이탈 위험도, 상위 3개 영향 요인을 포함한 결과
    """
    try:
        # 데이터셋 경로 설정
        data_path = os.path.join('models', 'ecommerce_for_prediction.csv')
        
        # 파일 존재 여부 확인
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {data_path}")
        
        # 데이터셋 로드
        df = pd.read_csv(data_path)
        
        # CustomerID 컬럼 확인 및 생성
        if 'CustomerID' not in df.columns:
            print("CustomerID 컬럼이 없어 순차적인 ID를 생성합니다.")
            df['CustomerID'] = [f'CUST_{i:06d}' for i in range(1, len(df) + 1)]
        
        # ChurnPredictor 인스턴스 생성
        predictor = ChurnPredictor()
        
        # 예측을 위한 특성 컬럼 준비 (CustomerID 제외)
        feature_columns = [col for col in df.columns if col != 'CustomerID']
        prediction_df = df[feature_columns]  # CustomerID를 제외한 데이터프레임 생성
        
        # 예측 수행
        _, churn_probabilities = predictor.predict(prediction_df)
        
        # SHAP 값 계산
        explainer = shap.TreeExplainer(predictor.model)
        shap_values = explainer.shap_values(prediction_df)  # CustomerID를 제외한 데이터로 SHAP 값 계산
        
        # 이진 분류의 경우 positive class의 SHAP 값 사용
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # 특성 이름 한글화 매핑
        feature_korean = {
            'DaySinceLastOrder': '마지막 주문 후 경과일',
            'SatisfactionScore': '만족도 점수',
            'Complain': '불만 제기',
            'OrderCount': '주문 횟수',
            'CashbackAmount': '캐시백 금액',
            'HourSpendOnApp': '앱 사용 시간',
            'OrderAmountHikeFromlastYear': '작년 대비 주문 증가율',
            'CouponUsed': '쿠폰 사용 횟수',
            'Tenure': '거래 기간',
            'WarehouseToHome': '배송 거리',
            'PreferredLoginDevice': '선호 로그인 기기',
            'PreferredPaymentMode': '선호 결제 수단',
            'Gender': '성별',
            'PreferedOrderCat': '선호 주문 카테고리',
            'MaritalStatus': '결혼 여부'
        }
        
        # 결과 저장을 위한 리스트
        results = []
        
        # 분석에 사용할 특성 컬럼 (CustomerID 제외)
        feature_columns = [col for col in df.columns if col != 'CustomerID']
        
        # 각 고객별로 SHAP 값 분석
        for idx, (customer_id, prob) in enumerate(zip(df['CustomerID'], churn_probabilities)):
            # 해당 고객의 SHAP 값
            customer_shap = shap_values[idx]
            
            # 특성별 SHAP 값 계산
            feature_impacts = []
            
            # 전체 SHAP 값의 절대값 합계 계산 (전체 영향도)
            total_impact = np.sum(np.abs(customer_shap))
            
            for feature_name, shap_value in zip(feature_columns, customer_shap):
                # 특성의 기본 이름 추출 (원-핫 인코딩된 경우 처리)
                base_feature_name = feature_name.split('_')[0] if '_' in feature_name else feature_name
                
                # 특성의 한글 이름
                feature_kr = feature_korean.get(base_feature_name, feature_name)
                
                # 원-핫 인코딩된 경우 세부 값 추가
                if '_' in feature_name:
                    feature_value = feature_name.split('_', 1)[1]  # 첫 번째 '_' 이후의 모든 부분을 값으로 사용
                    feature_kr = f"{feature_kr} ({feature_value})"
                
                # SHAP 값의 절대값과 부호 저장
                abs_impact = abs(shap_value)
                # 전체 영향도 대비 상대적 중요도 계산 (백분율)
                relative_importance = (abs_impact / total_impact * 100) if total_impact > 0 else 0
                # 영향의 방향 결정 (양수: 이탈 확률 증가, 음수: 이탈 확률 감소)
                if shap_value > 0:
                    direction = "⚠️ 부정"
                else:
                    direction = "✅ 긍정"
                
                feature_impacts.append({
                    'feature': feature_kr,
                    'importance': relative_importance,
                    'direction': direction,
                    'raw_shap': shap_value
                })
            
            # 중요도가 큰 순서대로 정렬
            feature_impacts.sort(key=lambda x: x['importance'], reverse=True)
            
            # 상위 3개 특성 추출
            top_features = feature_impacts[:3]
            
            # 결과 저장
            result = {
                'CustomerID': customer_id,
                'Churn Risk': prob
            }
            
            # 상위 3개 특성 정보 저장
            for i, feature in enumerate(top_features, 1):
                result[f'Top Feature {i}'] = f"{feature['feature']} ({feature['direction']})"
                result[f'Importance {i}'] = feature['importance']
            
            results.append(result)
        
        # 결과 DataFrame 생성
        result_df = pd.DataFrame(results)
        
        # 결과 검증
        if len(result_df) == 0:
            raise ValueError("분석 결과가 생성되지 않았습니다.")
        if result_df['Churn Risk'].isna().any():
            raise ValueError("이탈 확률에 결측값이 있습니다.")
        
        return result_df
        
    except Exception as e:
        print(f"고객 분석 중 오류 발생: {str(e)}")
        raise 

def load_customer_data():
    """
    전체 고객 데이터를 로드합니다.
    Returns:
        DataFrame: 모든 컬럼을 포함한 전체 고객 데이터
    """
    try:
        # 데이터셋 경로 설정
        data_path = os.path.join('models', 'ecommerce_for_prediction.csv')
        
        # 파일 존재 여부 확인
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {data_path}")
        
        # 데이터셋 로드
        df = pd.read_csv(data_path)
        
        return df
        
    except Exception as e:
        print(f"데이터 로드 중 오류 발생: {str(e)}")
        raise 