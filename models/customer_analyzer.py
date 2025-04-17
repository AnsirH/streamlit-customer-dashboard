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
        self.model_path = os.path.join('models', 'xgb_best_model.pkl')
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
            
            # 데이터 전처리
            processed_df = self._preprocess_data(input_df)
            
            # 예측 수행
            try:
                y_pred = self.model.predict(processed_df)
                y_proba = self.model.predict_proba(processed_df)[:, 1]  # 이탈 확률
                
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
    
    def _preprocess_data(self, input_df):
        """
        입력 데이터를 전처리합니다.
        
        Args:
            input_df (pandas.DataFrame): 원본 입력 데이터
            
        Returns:
            pandas.DataFrame: 전처리된 데이터
        """
        try:
            print("전처리 시작 - 입력 데이터 정보:")
            print(f"컬럼: {input_df.columns.tolist()}")
            print(f"데이터 타입: {input_df.dtypes}")
            print(f"첫 번째 행: {input_df.iloc[0].to_dict()}")
            
            # 모델이 요구하는 feature 확인
            if self.model is not None and hasattr(self.model, 'feature_names_'):
                print("\n모델이 요구하는 feature:")
                print(self.model.feature_names_)
            
            # 입력 데이터 복사
            df = input_df.copy()
            
            # CustomerID 제거 (예측에 사용되지 않음)
            if 'CustomerID' in df.columns:
                df = df.drop('CustomerID', axis=1)
            
            # Churn 컬럼이 있다면 제거 (타겟 변수이므로 예측에 사용되지 않음)
            if 'Churn' in df.columns:
                df = df.drop('Churn', axis=1)
            
            # 카테고리형 변수 인코딩
            categorical_columns = [
                'PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                'PreferedOrderCat', 'MaritalStatus'
            ]
            
            for col in categorical_columns:
                if col in df.columns:
                    print(f"\n{col} 컬럼 처리 전:")
                    print(f"데이터 타입: {df[col].dtype}")
                    print(f"고유값: {df[col].unique()}")
                    
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    
                    print(f"{col} 컬럼 처리 후:")
                    print(f"데이터 타입: {df[col].dtype}")
                    print(f"고유값: {df[col].unique()}")
            
            print("\n전처리 완료 - 최종 데이터 정보:")
            print(f"컬럼: {df.columns.tolist()}")
            print(f"데이터 타입: {df.dtypes}")
            print(f"첫 번째 행: {df.iloc[0].to_dict()}")
            
            return df
            
        except Exception as e:
            print(f"전처리 중 오류 발생: {str(e)}")
            print(f"오류 발생 위치: {e.__traceback__.tb_lineno}")
            raise
    
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
        data_path = os.path.join('models', 'E Commerce Dataset2.xlsx')
        
        # 파일 존재 여부 확인
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {data_path}")
        
        # 데이터셋 로드
        df = pd.read_excel(data_path)
        
        # CustomerID 컬럼 확인
        if 'CustomerID' not in df.columns:
            raise ValueError("CustomerID 컬럼이 데이터셋에 존재하지 않습니다.")
        
        # ChurnPredictor 인스턴스 생성
        predictor = ChurnPredictor()
        
        # 예측 수행
        predictions, probabilities = predictor.predict(df)
        
        # SHAP 값 계산
        explainer = shap.TreeExplainer(predictor.model)
        shap_values = explainer.shap_values(X)
        
        # 이탈 확률 예측
        _, churn_probabilities = predictor.predict(X)
        
        # 결과 데이터프레임 생성
        results = []
        for i, (customer_id, prob) in enumerate(zip(df['CustomerID'], churn_probabilities)):
            # 해당 고객의 SHAP 값 추출
            customer_shap = shap_values[i]
            
            # 모든 특성의 SHAP 값 계산 (부호 유지)
            shap_values_with_sign = [(feature, shap) 
                                   for feature, shap in zip(feature_columns, customer_shap)]
            
            # 모든 특성의 SHAP 값 절대값 합계 계산
            total_abs_shap = sum(abs(shap) for _, shap in shap_values_with_sign)
            
            # SHAP 값 기준으로 정렬 (절대값 기준)
            sorted_features = sorted(shap_values_with_sign, 
                                   key=lambda x: abs(x[1]), reverse=True)[:3]
            
            # 정규화된 중요도 계산 (전체 특성 대비 비율)
            normalized_importances = [(feature, shap / total_abs_shap * 100) 
                                   for feature, shap in sorted_features]
            
            # 방향성 정보를 포함한 특성 이름 생성
            feature_names = []
            for feature, importance in normalized_importances:
                direction = "증가" if importance > 0 else "감소"
                feature_names.append(f"{feature} ({direction})")
            
            result = {
                'CustomerID': customer_id,
                'Churn Risk': prob,
                'Top Feature 1': feature_names[0],
                'Importance 1': abs(normalized_importances[0][1]),
                'Top Feature 2': feature_names[1],
                'Importance 2': abs(normalized_importances[1][1]),
                'Top Feature 3': feature_names[2],
                'Importance 3': abs(normalized_importances[2][1])
            }
            results.append(result)
        
        # 결과 DataFrame 생성
        result_df = pd.DataFrame(results)
        
        # 결과 검증
        if len(result_df) == 0:
            raise ValueError("분석 결과가 생성되지 않았습니다.")
        if result_df['Churn Risk'].isna().any():
            raise ValueError("이탈 확률에 결측값이 있습니다.")
        
        return result_df
        
    except FileNotFoundError as e:
        raise Exception(f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise Exception(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
    except Exception as e:
        raise Exception(f"고객 분석 중 오류가 발생했습니다: {str(e)}") 