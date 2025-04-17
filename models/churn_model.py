import pandas as pd
import numpy as np
from pathlib import Path
from utils.cache import load_model
from utils.logger import setup_logger
from config import PATHS, MODEL_CONFIG
import joblib
import pickle
import os
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

logger = setup_logger(__name__)

########## 함수업데이트작업 ##########

class ChurnPredictor:
    """고객 이탈 예측을 위한 모델 클래스"""
    
    def __init__(self, model_path=None):
        """모델을 로드하고 초기화합니다."""
        self.model = None
        if model_path is None:
            self.model_path = Path(__file__).parent / "xgboost_best_model.pkl"
        else:
            self.model_path = model_path
        self.feature_importance_cache = None  # 특성 중요도 캐시 추가
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"모델 로드 오류: {str(e)}")
            st.error(f"모델 로드 중 오류가 발생했습니다.")
    
    def load_model(self):
        """모델 파일을 로드합니다."""
        try:
            if not self.model_path.exists():
                logger.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                return False
            
            self.model = joblib.load(self.model_path)
            logger.info(f"모델 로드 성공: {self.model_path}")
            # 디버그 출력 추가
            st.write(f"🔍 디버그: 모델 로드 성공 - {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            # 디버그 출력 추가
            st.error(f"🔍 디버그: 모델 로드 실패 - {str(e)}")
            return False
    
    def predict(self, input_df):
        """
        이탈 예측을 수행합니다.
        
        Args:
            input_df (pandas.DataFrame): 예측할 고객 데이터
            
        Returns:
            tuple: (예측 클래스, 이탈 확률) 또는 오류 발생 시 (None, None)
        """
        try:
            # 1. 입력 데이터 검증
            if input_df is None or len(input_df) == 0:
                st.error("⚠️ 예측할 데이터가 없습니다.")
                return None, None
                
            # 2. 모델 검증
            if self.model is None:
                st.error("⚠️ 모델이 로드되지 않았습니다.")
                if self.load_model() is False:
                    return None, None
            
            # 3. 입력 데이터의 범주형 변수 값이 모델과 호환되는지 미리 검증
            safe_prediction, unknown_values = self._validate_categorical_values(input_df)
            
            # 4. 모델이 인식하지 못하는 범주값이 있는 경우
            if not safe_prediction:
                # 4.1 엄격한 모드: 예측 중단
                if self.strict_mode:
                    st.error("⚠️ 모델이 인식하지 못하는 범주값이 있어 예측을 중단합니다.")
                    for cat, values in unknown_values.items():
                        st.error(f"- '{cat}' 변수에 알 수 없는 값: {values}")
                    return None, None
                # 4.2 유연한 모드: 경고 표시 후 계속 진행 (자동 대체)
                else:
                    st.warning("⚠️ 모델이 인식하지 못하는 범주값이 있습니다. 자동 대체 사용.")
                    for cat, values in unknown_values.items():
                        st.warning(f"- '{cat}' 변수에 알 수 없는 값: {values}")
            
            # 5. 데이터 전처리
            processed_df = self._preprocess_data(input_df)
            
            # 6. 예측 수행
            try:
                y_pred = self.model.predict(processed_df)
                y_proba = self.model.predict_proba(processed_df)[:, 1]  # 이탈 확률
                
                # 7. 불확실성 정보 추가 (모델이 인식하지 못하는 범주값이 있는 경우)
                if not safe_prediction and not self.strict_mode:
                    # 7.1 예측 불확실성 경고
                    st.warning("⚠️ 모델이 인식하지 못하는 범주값이 있어 예측 결과의 신뢰도가 낮을 수 있습니다.")
                    # 7.2 불확실성을 반영한 확률 조정 (선택 사항)
                    # y_proba = self._adjust_probability_for_uncertainty(y_proba, unknown_values)
                
                # 8. 예측 성공 시 특성 중요도 계산 (선택 사항)
                try:
                    self._compute_feature_importance(processed_df)
                except Exception as e:
                    # 특성 중요도 계산 실패해도 예측 결과는 반환
                    pass
                
                return y_pred, y_proba
                
            except Exception as e:
                st.error(f"⚠️ 예측 수행 중 오류: {str(e)}")
                return None, None
        
        except Exception as e:
            st.error(f"⚠️ 예측 처리 중 오류: {str(e)}")
            import traceback
            st.write(f"🔍 [예측 오류]: {traceback.format_exc()}")
            return None, None
    
    def _validate_categorical_values(self, input_df):
        """
        입력 데이터의 범주형 변수 값이 모델과 호환되는지 검증합니다.
        
        Args:
            input_df (pd.DataFrame): 검증할 입력 데이터
            
        Returns:
            tuple: (안전한 예측 여부, 알 수 없는 값 목록)
        """
        # 1. 기본값 설정
        is_safe = True
        unknown_values = {}
        
        # 2. 모델이 없는 경우 검증 불가
        if self.model is None or not hasattr(self.model, 'feature_names_in_'):
            return True, {}  # 검증 불가하므로 안전하다고 가정
            
        # 3. 원핫인코딩된 특성 구조 가져오기
        encoded_features = self.get_onehot_encoded_features()
        
        # 4. 각 범주형 변수별 가능한 값 목록 추출
        category_values = {}
        for prefix, features in encoded_features.items():
            category_values[prefix] = [f.split('_', 1)[1] for f in features]
        
        # 5. 입력 데이터의 각 범주형 변수 검증
        for prefix in encoded_features.keys():
            if prefix in input_df.columns:
                # 5.1 입력 데이터의 해당 컬럼 고유값 가져오기
                input_values = input_df[prefix].astype(str).str.strip().unique()
                
                # 5.2 모델이 인식하지 못하는 값 찾기
                possible_values = [str(v).lower() for v in category_values.get(prefix, [])]
                unknown = [val for val in input_values 
                          if str(val).lower() not in possible_values]
                
                # 5.3 알 수 없는 값이 있으면 기록
                if unknown:
                    is_safe = False
                    unknown_values[prefix] = unknown
        
        return is_safe, unknown_values

    def _preprocess_data(self, data):
        """
        모델 예측을 위해 입력 데이터를 전처리합니다.
        모델이 요구하는 정확한 특성 구조와 일치하는 데이터를 생성합니다.
        
        Args:
            data (pd.DataFrame): 전처리할 입력 데이터
            
        Returns:
            pd.DataFrame: 모델 입력에 맞게 전처리된 데이터
        """
        try:
            # 0. 모델 설정 확인
            self.strict_mode = False  # 기본적으로 유연한 모드 사용 (예측 계속 진행)
            
            # 1. 원본 데이터 로깅
            st.write(f"🔍 [전처리]: 원본 데이터 크기: {data.shape}")
            
            # 2. 모델 확인
            if self.model is None:
                st.error("⚠️ 모델이 로드되지 않았습니다.")
                return data.copy()
                
            if not hasattr(self.model, 'feature_names_in_'):
                st.error("⚠️ 모델에 'feature_names_in_' 속성이 없습니다.")
                return data.copy()
            
            # 3. 모델 특성 가져오기
            model_features = list(self.model.feature_names_in_)
            st.write(f"🔍 [전처리]: 모델 특성 수: {len(model_features)}")
            
            # 4. 원핫인코딩 구조 분석
            encoded_features_dict = self.get_onehot_encoded_features()
            
            # 5. 범주형 변수 식별 (원핫인코딩된 특성에서 추출)
            categorical_cols = list(encoded_features_dict.keys())
            st.write(f"🔍 [전처리]: 원핫인코딩 대상 범주형 변수: {categorical_cols}")
            
            # 6. 각 범주형 변수별 가능한 값 목록 추출
            category_values = {}
            for prefix, features in encoded_features_dict.items():
                category_values[prefix] = [f.split('_', 1)[1] for f in features]
            
            # 7. 새 결과 데이터프레임 생성 (모델이 필요로 하는 모든 특성을 포함하게 됨)
            result_df = pd.DataFrame(index=data.index)
            
            # 8. 미리 모든 원핫인코딩 컬럼을 0으로 초기화
            for feature in model_features:
                result_df[feature] = 0
            
            # 9. 범주형 변수 처리 및 임의 값 매핑 기록
            fallback_mappings = {}  # 매핑 정보 저장
            
            # 10. 각 범주형 변수 처리
            for prefix in categorical_cols:
                if prefix in data.columns:
                    # 10.1 입력 데이터에서 해당 컬럼 값 가져오기
                    for idx, row in data.iterrows():
                        input_value = str(row[prefix]).strip()
                        
                        # 10.2 해당 값에 대한 원핫인코딩 컬럼 이름 생성
                        expected_col = f"{prefix}_{input_value}"
                        
                        # 10.3 해당 컬럼이 모델 특성에 존재하는지 확인
                        col_exists = expected_col in model_features
                        
                        if col_exists:
                            # 10.4 존재하는 경우 해당 컬럼을 1로 설정
                            result_df.loc[idx, expected_col] = 1
                        else:
                            # 10.5 존재하지 않는 경우 대체 전략 적용
                            st.warning(f"⚠️ '{prefix}'의 값 '{input_value}'에 대한 원핫인코딩 컬럼이 모델에 없습니다.")
                            
                            if prefix in category_values and category_values[prefix]:
                                # 10.6 유사도 기반 매핑 시도
                                possible_values = category_values[prefix]
                                
                                # 10.6.1 방법 1: 정확히 대소문자만 다른 경우
                                case_insensitive_match = next((val for val in possible_values 
                                                       if val.lower() == input_value.lower()), None)
                                
                                if case_insensitive_match:
                                    # 대소문자 차이만 있는 매칭 사용
                                    fallback_col = f"{prefix}_{case_insensitive_match}"
                                    result_df.loc[idx, fallback_col] = 1
                                    fallback_mappings[input_value] = case_insensitive_match
                                    st.write(f"✓ 대소문자 무시 매핑: '{input_value}' → '{case_insensitive_match}'")
                                else:
                                    # 10.6.2 방법 2: 첫 번째 값 사용 (기본값)
                                    fallback_value = possible_values[0]  # 첫 번째 값을 기본값으로 사용
                                    fallback_col = f"{prefix}_{fallback_value}"
                                    result_df.loc[idx, fallback_col] = 1
                                    fallback_mappings[input_value] = fallback_value
                                    st.write(f"✓ 기본값 매핑: '{input_value}' → '{fallback_value}' (첫 번째 값)")
                            else:
                                st.error(f"⚠️ '{prefix}'에 대한 가능한 값 목록을 찾을 수 없습니다!")
            
            # 11. 범주형이 아닌 일반 특성 처리
            for feature in model_features:
                # 원핫인코딩된 특성이 아닌 경우만 처리 (이미 위에서 처리되지 않은 경우)
                if '_' not in feature and feature in data.columns:
                    result_df[feature] = data[feature]
            
            # 12. 디버깅: 대체 매핑 정보 출력
            if fallback_mappings:
                st.write("### ⚠️ 원핫인코딩 값 대체 정보")
                for original, replacement in fallback_mappings.items():
                    st.write(f"- '{original}' → '{replacement}'")
            
            # 13. 결과 데이터 검증
            if set(result_df.columns) != set(model_features):
                st.error("⚠️ 생성된 특성이 모델 특성과 일치하지 않습니다!")
                missing = set(model_features) - set(result_df.columns)
                extra = set(result_df.columns) - set(model_features)
                
                if missing:
                    st.write(f"- 누락된 특성: {list(missing)}")
                if extra:
                    st.write(f"- 추가된 특성: {list(extra)}")
                    result_df = result_df.drop(columns=list(extra))
                
                # 누락된 특성 추가
                for feat in missing:
                    result_df[feat] = 0
            
            # 14. 모델 특성 순서에 맞게 재정렬
            result_df = result_df[model_features]
            
            # 15. 각 범주형 변수별 활성화된 컬럼 확인
            if len(data) == 1:  # 단일 레코드인 경우
                for prefix in categorical_cols:
                    if prefix in data.columns:
                        # 현재 값 확인
                        input_value = data[prefix].iloc[0]
                        
                        # 이 범주형 변수에 해당하는 모든 원핫인코딩 컬럼
                        ohe_cols = [col for col in result_df.columns if col.startswith(f"{prefix}_")]
                        
                        # 활성화된 컬럼 (값이 1인 컬럼)
                        active_cols = [col for col in ohe_cols if result_df[col].iloc[0] == 1]
                        
                        if active_cols:
                            # 활성화된 컬럼이 있는 경우
                            active_values = [col.split('_', 1)[1] for col in active_cols]
                            st.write(f"✓ '{prefix}': 입력값 '{input_value}' → 활성화된 값: {active_values}")
                        else:
                            # 활성화된 컬럼이 없는 경우 (중요한 오류!)
                            st.error(f"⚠️ '{prefix}'에 대해 활성화된 원핫인코딩 컬럼이 없습니다!")
                            st.write(f"  - 원본 값: '{input_value}'")
                            st.write(f"  - 가능한 값: {category_values.get(prefix, [])}")
            
            # 16. 전처리 완료
            st.write(f"🔍 [전처리]: 데이터 처리 완료, 최종 크기: {result_df.shape}")
            
            return result_df
            
        except Exception as e:
            st.error(f"⚠️ 데이터 전처리 중 오류: {str(e)}")
            import traceback
            st.write(f"🔍 [전처리 오류]: {traceback.format_exc()}")
            return data.copy()
    
    def get_onehot_encoded_features(self):
        """
        모델에서 사용하는 원핫인코딩된 특성 목록을 추출합니다.
        
        Returns:
            dict: 범주형 변수별 원핫인코딩된 특성 목록 (예: {'Gender': ['Gender_Male', 'Gender_Female']})
        """
        # 1. 기본 검증
        if self.model is None:
            st.warning("⚠️ 모델이 없어 원핫인코딩 특성을 분석할 수 없습니다.")
            return {}
            
        if not hasattr(self.model, 'feature_names_in_'):
            st.warning("⚠️ 모델에 feature_names_in_ 속성이 없습니다.")
            return {}
        
        # 2. 모델의 모든 특성 가져오기
        all_features = list(self.model.feature_names_in_)
        st.write(f"🔍 디버그 [원핫인코딩 분석]: 모델 전체 특성 수: {len(all_features)}")
        
        # 3. 모든 특성에서 접두사-값 패턴 찾기
        pattern_features = []
        for feature in all_features:
            if '_' in feature:
                pattern_features.append(feature)
        
        st.write(f"🔍 디버그 [원핫인코딩 분석]: '_'가 포함된 특성 수: {len(pattern_features)}")
        
        if not pattern_features:
            st.warning("⚠️ 원핫인코딩 패턴을 가진 특성이 없습니다. 모델이 원핫인코딩을 사용하지 않을 수 있습니다.")
            # 모델 특성 샘플 출력
            st.write(f"🔍 디버그 [원핫인코딩 분석]: 모델 특성 샘플: {all_features[:10]}")
            return {}
        
        # 4. 모든 접두사 추출 및 개수 카운트
        prefix_counter = {}
        for feature in pattern_features:
            prefix = feature.split('_')[0]
            if prefix not in prefix_counter:
                prefix_counter[prefix] = []
            prefix_counter[prefix].append(feature)
        
        # 5. 접두사별 통계 출력 (디버깅)
        st.write(f"🔍 디버그 [원핫인코딩 분석]: 발견된 접두사: {list(prefix_counter.keys())}")
        
        # 6. 원핫인코딩으로 간주할 최소 변수 수 (적어도 2개 이상이어야 함)
        min_variants = 2
        
        # 7. 원핫인코딩 변수 그룹화
        encoded_features = {}
        for prefix, features in prefix_counter.items():
            if len(features) >= min_variants:
                # 접두사가 일반적인 범주형 변수명인지 확인
                is_valid_prefix = any(common in prefix.lower() for common in [
                    'gender', 'sex', 'marital', 'city', 'status', 'login', 'device', 
                    'payment', 'order', 'cat', 'category', 'type', 'satisfaction', 
                    'education', 'income', 'occupation', 'complain'
                ])
                
                if is_valid_prefix or len(features) >= 3:  # 잘 알려진 범주형 변수명이거나 3개 이상의 변형이 있는 경우
                    encoded_features[prefix] = features
                    category_values = [f.split('_', 1)[1] for f in features]
                    
                    st.write(f"✅ 디버그 [원핫인코딩 분석]: 범주형 변수 '{prefix}' 확인됨")
                    st.write(f"  - 컬럼 수: {len(features)}")
                    st.write(f"  - 범주값: {category_values}")
        
        # 8. 원핫인코딩 변수가 발견되지 않은 경우 
        if not encoded_features:
            st.warning("⚠️ 원핫인코딩 변수가 발견되지 않았습니다!")
            
            # 발견된 모든 접두사 정보 출력
            for prefix, features in prefix_counter.items():
                st.write(f"🔍 디버그 [원핫인코딩 분석]: 접두사 '{prefix}': {len(features)}개 컬럼")
                if len(features) <= 5:  # 컬럼 수가 적은 경우 모두 표시
                    st.write(f"  - 컬럼: {features}")
                else:
                    st.write(f"  - 컬럼 샘플: {features[:5]}...")
            
            # 특성명 구조 분석
            uppercase_count = sum(1 for f in all_features if any(c.isupper() for c in f))
            digit_count = sum(1 for f in all_features if any(c.isdigit() for c in f))
            
            st.write(f"🔍 디버그 [원핫인코딩 분석]: 대문자 포함 특성 수: {uppercase_count}")
            st.write(f"🔍 디버그 [원핫인코딩 분석]: 숫자 포함 특성 수: {digit_count}")
            
            # 모델 내장 중요도 정보가 있다면 상위 특성 출력
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[-5:]  # 상위 5개
                st.write("🔍 디버그 [원핫인코딩 분석]: 가장 중요한 특성 Top 5:")
                for i in reversed(indices):
                    st.write(f"  - {all_features[i]}: {importances[i]:.4f}")
            
            # 수동으로 몇 가지 범주형 변수 추가
            st.write("🔍 디버그 [원핫인코딩 분석]: 일반적인 범주형 변수명 검색 중...")
            common_categories = ['Gender', 'MaritalStatus', 'LoginDevice', 'PaymentMode', 'OrderCat']
            
            for category in common_categories:
                matching = [f for f in all_features if category.lower() in f.lower()]
                if matching:
                    st.write(f"🔍 디버그 [원핫인코딩 분석]: '{category}' 관련 특성 발견: {len(matching)}개")
                    st.write(f"  - 특성: {matching}")
                    # 관련 특성이 발견되면 encoded_features에 추가
                    encoded_features[category] = matching
            
            # 그래도 발견되지 않으면 언더스코어로 구분된 모든 특성 추가
            if not encoded_features and pattern_features:
                st.write("🔍 디버그 [원핫인코딩 분석]: 최후의 방법으로 모든 '_' 포함 특성을 사용합니다")
                for prefix, features in prefix_counter.items():
                    if len(features) >= min_variants:
                        encoded_features[prefix] = features
        
        return encoded_features
    
    def _compute_feature_importance(self, input_data):
        """Calculate feature importance for a prediction."""
        try:
            # 캐시된 특성 중요도가 있으면 사용
            if self.feature_importance_cache is not None:
                return self.feature_importance_cache
                
            # 이하 기존 로직
            if self.model is None:
                self.load_model()
                
            if self.model is None:  # 여전히 None이면 기본값 반환
                return self._default_feature_importance()
                
            # SHAP 사용 시도
            try:
                import shap
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(input_data)
                
                # 분류 모델인 경우 클래스 1(이탈)에 대한 SHAP 값 선택
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                    
                # 절대값 취해 중요도 계산
                importance = np.abs(shap_values).mean(axis=0)
                
                # 캐시에 저장
                self.feature_importance_cache = importance
                return importance
                
            except Exception as e:
                print(f"SHAP을 사용한 특성 중요도 계산 실패: {str(e)}")
                
                # 모델에 feature_importances_ 속성이 있으면 사용
                if hasattr(self.model, 'feature_importances_'):
                    importance = self.model.feature_importances_
                    self.feature_importance_cache = importance
                    return importance
                    
                return self._default_feature_importance()
                
        except Exception as e:
            print(f"특성 중요도 계산 중 오류 발생: {str(e)}")
            return self._default_feature_importance()
    
    def _default_feature_importance(self):
        """특성 중요도 계산이 실패할 경우 기본값을 반환합니다."""
        # 기본 특성 중요도 값 설정
        default_importance = np.array([0.25, 0.22, 0.18, 0.15, 0.12, 0.08])
        self.feature_importance_cache = default_importance
        return default_importance
    
    def get_feature_importance(self):
        """캐시된 특성 중요도를 반환합니다."""
        if self.feature_importance_cache is None:
            # 특성 중요도가 계산되지 않았다면 기본값 반환
            return self._default_feature_importance()
            
        # 특성 중요도가 배열 형태인 경우 사전 형태로 변환
        if isinstance(self.feature_importance_cache, np.ndarray):
            # 특성 이름이 없는 경우 기본 이름 사용
            features = {}
            for i, val in enumerate(self.feature_importance_cache):
                features[f'feature_{i+1}'] = float(val)
            return features
            
        return self.feature_importance_cache

    def get_model_features(self):
        """
        모델이 요구하는 특성(컬럼) 목록을 반환합니다.
        
        Returns:
            dict: 모델 특성에 대한 상세 정보
        """
        if self.model is None:
            st.error("⚠️ 모델이 로드되지 않았습니다.")
            return {"error": "모델이 로드되지 않았습니다"}
            
        if not hasattr(self.model, 'feature_names_in_'):
            st.error("⚠️ 모델에 feature_names_in_ 속성이 없습니다.")
            return {"error": "모델에 feature_names_in_ 속성이 없습니다"}
            
        # 모델의 모든 특성
        all_features = list(self.model.feature_names_in_)
        
        # 결과 정보 구성
        result = {
            "total_features": len(all_features),
            "features": all_features,
            "sample_features": all_features[:10] if len(all_features) > 10 else all_features
        }
        
        # 원핫인코딩 특성 그룹화
        encoded_features = self.get_onehot_encoded_features()
        result["onehot_groups"] = len(encoded_features)
        result["onehot_features"] = encoded_features
        
        # 원핫인코딩 되지 않은 특성
        all_encoded = []
        for group in encoded_features.values():
            all_encoded.extend(group)
        
        non_encoded = [f for f in all_features if f not in all_encoded]
        result["non_encoded_features"] = non_encoded
        
        # 특성 타입별 분류
        has_underscore = [f for f in all_features if '_' in f]
        has_digit = [f for f in all_features if any(c.isdigit() for c in f)]
        has_uppercase = [f for f in all_features if any(c.isupper() for c in f)]
        
        result["has_underscore_count"] = len(has_underscore)
        result["has_digit_count"] = len(has_digit)
        result["has_uppercase_count"] = len(has_uppercase)
        
        # 중요도 정보가 있는 경우 
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[-10:]  # 상위 10개
            top_features = [(all_features[i], float(importances[i])) for i in reversed(top_indices)]
            result["top_features"] = top_features
            
        return result
    
    def print_model_features(self):
        """
        모델이 요구하는 특성 정보를 화면에 출력합니다.
        """
        features_info = self.get_model_features()
        
        if "error" in features_info:
            st.error(features_info["error"])
            return
            
        st.write("## 모델 특성 정보")
        st.write(f"- 총 특성 수: {features_info['total_features']}")
        
        # 특성 샘플 표시
        st.write("### 특성 샘플")
        st.write(features_info["sample_features"])
        
        # 원핫인코딩 그룹 표시
        st.write(f"### 원핫인코딩 그룹 ({features_info['onehot_groups']}개)")
        
        for group, features in features_info["onehot_features"].items():
            values = [f.split('_', 1)[1] for f in features]
            st.write(f"- **{group}**: {len(features)}개 값")
            st.write(f"  - 값 목록: {values}")
            
        # 원핫인코딩 되지 않은 특성
        if features_info["non_encoded_features"]:
            st.write("### 원핫인코딩 되지 않은 특성")
            st.write(features_info["non_encoded_features"])
            
        # 중요도 정보가 있는 경우
        if "top_features" in features_info:
            st.write("### 상위 10개 중요 특성")
            for feature, importance in features_info["top_features"]:
                st.write(f"- {feature}: {importance:.4f}")
                
        # 특성 형식 정보
        st.write("### 특성 형식 정보")
        st.write(f"- 언더스코어(_) 포함: {features_info['has_underscore_count']}개")
        st.write(f"- 숫자 포함: {features_info['has_digit_count']}개")
        st.write(f"- 대문자 포함: {features_info['has_uppercase_count']}개")


########## 함수영역역 ##########

# 모델 로드 함수
# MODEL_PATH = Path("models/xgb_best_model.pkl")  # GitHub 프로젝트 내 상대 경로 사용 권장

# ===============================
# ✅ 모델 로드 및 예측 함수
# ===============================
MODEL_PATH = Path(__file__).parent / "xgboost_best_model.pkl"

def load_churn_model(model_path: str = None):
    """
    Load the trained churn prediction model.
    
    Args:
        model_path: Path to the model file. Default is models/xgb_best_model.pkl
        
    Returns:
        Trained model
    """
    if model_path is None:
        model_path = Path(__file__).parent / "xgboost_best_model.pkl"
    else:
        model_path = Path(model_path)
        
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    return joblib.load(model_path)

def predict_churn(model, input_df: pd.DataFrame) -> np.ndarray:
    """
    Predict churn probabilities for input data.
    
    Args:
        model: Trained model
        input_df: Input DataFrame for prediction
        
    Returns:
        np.ndarray: Churn probabilities
    """
    return model.predict_proba(input_df)[:, 1]  # Return churn probabilities

# ===============================
# 토큰 시각화 함수
# ===============================

# 1. 이탈 비율 시각화
def plot_churn_ratio(df: pd.DataFrame, target_col="Churn"):
    churn_counts = df[target_col].value_counts()
    plt.figure(figsize=(5, 4))
    sns.barplot(x=churn_counts.index, y=churn_counts.values, palette="Set2")
    plt.title("이탈 여부 비율")
    plt.ylabel("고객 수")
    return plt.gcf()

# 2. 계약 유형별 이탈 비율
def plot_churn_by_contract(df: pd.DataFrame, contract_col="Contract", target_col="Churn"):
    plt.figure(figsize=(7, 4))
    sns.countplot(data=df, x=contract_col, hue=target_col, palette="pastel")
    plt.title("계약 유형별 이탈 여부")
    plt.ylabel("고객 수")
    plt.xticks(rotation=15)
    return plt.gcf()

# 3. 예측된 이탈확률 분포
def plot_churn_probability_distribution(df: pd.DataFrame, proba_col="이탈확률"):
    plt.figure(figsize=(7, 4))
    sns.histplot(df[proba_col], bins=20, kde=True, color="coral")
    plt.title("예측된 이탈확률 분포")
    plt.xlabel("이탈확률 (%)")
    return plt.gcf()

# 4. 수치형 변수별 이탈자 분포 (ex: 나이, 이용개월수)
def plot_churn_by_numeric_feature(df: pd.DataFrame, feature_col="Tenure", target_col="Churn"):
    plt.figure(figsize=(7, 4))
    sns.kdeplot(data=df, x=feature_col, hue=target_col, fill=True, common_norm=False, alpha=0.5)
    plt.title(f"{feature_col}에 따른 이탈 분포")
    return plt.gcf()

# 5. 여러 수치형 변수에 대한 Boxplot 비교
def plot_feature_comparison(df: pd.DataFrame, feature_list, target_col="Churn"):
    n = len(feature_list)
    fig, axes = plt.subplots(nrows=1, ncols=n, figsize=(5*n, 4))
    for i, col in enumerate(feature_list):
        sns.boxplot(x=target_col, y=col, data=df, ax=axes[i], palette="Set2")
        axes[i].set_title(f"{col} vs {target_col}")
    plt.tight_layout()
    return fig

# 6. 단일 고객 정보 bar chart
def plot_single_customer(df: pd.DataFrame, idx: int):
    row = df.iloc[idx]
    features = row.drop(['예측결과', '이탈확률'], errors='ignore')
    plt.figure(figsize=(10, 4))
    features.plot(kind='barh', color='skyblue')
    plt.title(f"고객 {idx}번 특성 요약")
    plt.tight_layout()
    return plt.gcf()

# 7. SHAP 해석
def explain_shap(model, X_sample: pd.DataFrame):
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    shap.plots.beeswarm(shap_values)

# 8. Feature Importance (모델 기준)
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=fi_df, palette="viridis")
    plt.title("모델 Feature 중요도")
    plt.tight_layout()
    return plt.gcf()

# 9. 이탈 고객 대응 전략 추천
def recommend_solution(row):
    strategies = []
    if 'Contract' in row and row['Contract'] == 'Month-to-month':
        strategies.append("2년 계약 유도")
    if 'TechSupport' in row and row['TechSupport'] == 'No':
        strategies.append("기술 지원 제공")
    if 'OnlineSecurity' in row and row['OnlineSecurity'] == 'No':
        strategies.append("보안 서비스 추가")
    return strategies

# ===============================
# 추가가 업데이트트
# ===============================

# 12. 전체 SHAP 평균 기준 상위 feature 반환
def get_top_shap_features(shap_values, X, n=5):
    # 전체 SHAP 값에서 평균 영향력이 큰 상위 n개 feature 반환
    ...

# 13. 개별 고객 SHAP Waterfall 시각화
def plot_waterfall_for_customer(explainer, shap_values, X, idx):
    # 개별 고객의 SHAP 값을 Waterfall plot으로 시각화
    ...

# 14. 개별 고객 상위 영향 feature 반환
def get_customer_top_features(shap_values, X, idx, n=5):
    # 특정 고객의 예측에 가장 큰 영향을 준 feature 상위 n개 반환
    ...

# ===============================
# 고난이도 함수 업데이트
# ===============================


##########################
# 1. 데이터입력
def get_customer_input():
    st.subheader("고객 데이터 입력")

    cols = st.columns(3)

    with cols[0]:
        tenure = st.number_input("거래기간 (개월)", min_value=0, value=12)
        gender = st.selectbox("성별", ["Male", "Female"])
        marital_status = st.selectbox("결혼 상태", ["Single", "Married"])
        num_orders = st.number_input("주문 횟수", min_value=0, value=10)
        city_tier = st.number_input("도시 등급", min_value=1, max_value=3, value=1)
        registered_devices = st.number_input("등록된 기기 수", min_value=1, value=2)

    with cols[1]:
        preferred_login_device = st.selectbox("선호 로그인 기기", ["Mobile", "Computer"])
        app_usage = st.number_input("앱 사용 시간 (시간)", min_value=0.0, value=3.0)
        address_count = st.number_input("주소 개수", min_value=0, value=2)
        last_order_days = st.number_input("마지막 주문 후 경과일", min_value=0, value=15)
        warehouse_to_home = st.number_input("창고-집 거리 (km)", min_value=0.0, value=20.0)
        satisfaction_score = st.slider("만족도 점수 (1-5)", min_value=1, max_value=5, value=3)

    with cols[2]:
        preferred_payment = st.selectbox("선호 결제 방식", ["Credit Card", "Debit Card", "Cash on Delivery"])
        preferred_category = st.selectbox("선호 주문 카테고리", ["Electronics", "Clothing", "Groceries"])
        complaints = st.selectbox("불만 제기 여부", ["예", "아니오"])
        order_amount_diff = st.number_input("작년 대비 주문 금액 증가율 (%)", value=15.0)
        coupon_used = st.number_input("쿠폰 사용 횟수", value=3)
        cashback_amount = st.number_input("캐시백 금액 (원)", value=150.0)

    input_data = {
        "tenure": tenure,
        "preferred_login_device": preferred_login_device,
        "city_tier": city_tier,
        "warehouse_to_home": warehouse_to_home,
        "preferred_payment_method": preferred_payment,
        "gender": gender,
        "app_usage": app_usage,
        "registered_devices": registered_devices,
        "preferred_order_category": preferred_category,
        "satisfaction_score": satisfaction_score,
        "marital_status": marital_status,
        "address_count": address_count,
        "complaint_status": complaints,
        "order_amount_diff": order_amount_diff,
        "coupon_used": coupon_used,
        "num_orders": num_orders,
        "last_order_days": last_order_days,
        "cashback_amount": cashback_amount
    }

    return input_data

##########################
# 2. 위험표 예측
def show_churn_risk_dashboard(probability: float):
    """
    이탈 확률을 시각화하고 위험도 및 대응 조치를 출력
    :param probability: 예측된 이탈 확률 (0~1 또는 0~100)
    """

    # 1. 확률 정규화
    if probability <= 1.0:
        probability *= 100
    prob = round(probability, 2)

    # 2. 위험도 등급 판정
    if prob < 30:
        level = "낮음"
        color = "green"
        recommendation = "안정적인 상태입니다. 지속적인 관리만 유지하면 됩니다."
    elif prob < 70:
        level = "중간"
        color = "orange"
        recommendation = "일정 수준의 리스크가 있습니다. 고객 만족도 점검이 필요합니다."
    else:
        level = "높음"
        color = "red"
        recommendation = "즉각적인 고객 응대와 특별 혜택 제공이 필요합니다."

    # 3. 게이지 차트 생성
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={"suffix": "%"},
        title={"text": "이탈 가능성 (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30], "color": "green"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"},
            ],
        }
    ))

    # 4. Streamlit 출력
    st.subheader("📈 예측 결과")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 예측 결과 요약")
    st.markdown(f"""
    - **이탈 확률**: **{prob:.2f}%**  
    - **위험도**: <span style='color:{color}; font-weight:bold'>{level}</span>
    """, unsafe_allow_html=True)

    st.subheader("🛠 권장 조치")
    st.markdown(f"{recommendation}")


####################################################
# 이탈 예측 함수 모음
####################################################


# 1. 📄 모델 & 데이터 로딩
@st.cache_resource
def load_model_and_data(model_path, data_path):
    model = joblib.load(model_path)
    df = pd.read_pickle(data_path)
    return model, df

# 2. 📋 컬럼별 고객 정보 출력
def show_customer_info(customer_row):
    st.subheader("📋 고객 입력 데이터")
    for col, val in customer_row.items():
        st.write(f"**{col}**: {val}")

# 3. 🎯 위험도 게이지 표시
def show_churn_gauge(prob):
    if prob <= 1: prob *= 100
    risk = round(prob, 2)
    level = "높음" if risk >= 70 else ("중간" if risk >= 30 else "낮음")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        number={"suffix": "%"},
        title={"text": "이탈 가능성 (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 30], "color": "green"},
                {"range": [30, 70], "color": "yellow"},
                {"range": [70, 100], "color": "red"}
            ]
        }
    ))
    st.subheader("📈 예측 결과")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"**예측 확률**: {risk:.2f}%  |  **위험도**: :red[{level}]" if level == "높음" else f"**예측 확률**: {risk:.2f}%  |  **위험도**: :orange[{level}]" if level == "중간" else f"**예측 확률**: {risk:.2f}%  |  **위험도**: :green[{level}]")

# 4. 🔍 SHAP 상위 3개 영향 변수 시각화
def show_top_influencers(model, X_input):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_input)
    shap_df = pd.DataFrame(shap_values[1], columns=X_input.columns)
    shap_df_mean = shap_df.abs().mean().sort_values(ascending=False).head(3)
    fig = px.bar(x=shap_df_mean.index, y=shap_df_mean.values,
                 labels={'x': 'Feature', 'y': 'SHAP 평균 영향도'}, title='📌 주요 영향 요인 Top 3')
    st.plotly_chart(fig, use_container_width=True)





##########################
import joblib
from pathlib import Path

def load_xgboost_model2():
    """
    /models/xgboost_best_model.pkl 파일을 로드합니다.
    
    Returns:
        model: 학습된 XGBoost 모델
    Raises:
        FileNotFoundError: 모델 파일이 존재하지 않을 경우
        Exception: 모델 로드 중 오류 발생 시
    """
    model_path = Path(__file__).resolve().parent / "xgboost_best_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"[❌ 모델 파일 없음] {model_path}")

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"[❌ 모델 로드 실패] {e}")


####################################

# 이탈탭 값 리소스 초기화 포함 코드
class ChurnPredictor2:
    """고객 이탈 예측을 위한 개선된 모델 클래스"""

    def __init__(self, model_path=None, external_model=None):
        """모델 로드 또는 외부 주입"""
        self.model = external_model
        self.model_path = model_path or (Path(__file__).parent / "xgboost_best_model.pkl")
        self.feature_importance_cache = None

        if self.model is None:
            try:
                self.load_model()
            except Exception as e:
                logger.error(f"❌ 모델 로드 실패: {str(e)}")
                st.error("❌ 모델 로드 실패")

    def load_model(self):
        """로컬 모델 파일 로드"""
        if not self.model_path.exists():
            logger.error(f"❌ 모델 파일 없음: {self.model_path}")
            st.error(f"❌ 모델 파일이 존재하지 않습니다.\n{self.model_path}")
            return False
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"✅ 모델 로드 성공: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"❌ 모델 로딩 중 오류: {str(e)}")
            st.error(f"❌ 모델 로딩 중 오류\n{str(e)}")
            return False

    def predict(self, input_df: pd.DataFrame):
        if self.model is None:
            self.load_model()
        if self.model is None:
            st.warning("❗ [DEBUG] 모델이 존재하지 않아 기본값 반환됨")
            return self._default_prediction()

        try:
            processed_df = self._preprocess_data(input_df)

            # ✅ 디버깅용 출력
            st.write("✅ [DEBUG] 전처리 후 데이터:", processed_df)
            st.write("✅ [DEBUG] 전처리 컬럼 수:", processed_df.shape[1])

            y_pred = self.model.predict(processed_df)
            y_proba = self.model.predict_proba(processed_df)[:, 1]

            st.write("✅ [DEBUG] 예측 확률:", y_proba)

            if len(y_proba) == 0:
                st.warning("❗ [DEBUG] 예측 확률이 비어 있음, 기본값 반환")
                return self._default_prediction()

            self.feature_importance_cache = None
            self._compute_feature_importance(processed_df)

            return y_pred, y_proba
        except Exception as e:
            logger.error(f"❌ 예측 오류: {str(e)}")
            st.error(f"❌ [DEBUG] 예측 중 오류: {e}")
            return self._default_prediction()


    def _default_prediction(self):
        """모델 실패 시 기본 반환"""
        return np.array([0]), np.array([0.5])

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """입력 전처리"""
        df = df.copy()
        for col in ['CustomerID', 'customer_id', 'cust_id', 'id']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        if 'Complain' in df.columns and isinstance(df['Complain'].iloc[0], str):
            df['Complain'] = df['Complain'].apply(lambda x: 1 if x == '예' else 0)

        return df

    def _compute_feature_importance(self, input_df: pd.DataFrame):
        """SHAP 기반 특성 중요도 계산"""
        try:
            if self.model is None:
                return self._default_feature_importance()

            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(input_df)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # 클래스 1 기준

            importance = np.abs(shap_values).mean(axis=0)
            self.feature_importance_cache = importance
            return importance
        except Exception as e:
            logger.warning(f"⚠️ SHAP 계산 실패, 기본값 사용: {str(e)}")
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importance_cache = self.model.feature_importances_
                return self.feature_importance_cache
            return self._default_feature_importance()

    def _default_feature_importance(self):
        """중요도 실패 시 기본값 반환"""
        default_importance = np.array([0.25, 0.22, 0.18, 0.15, 0.12, 0.08])
        self.feature_importance_cache = default_importance
        return default_importance

    def get_feature_importance(self):
        """캐시된 중요도 반환"""
        if self.feature_importance_cache is None:
            return self._default_feature_importance()

        if isinstance(self.feature_importance_cache, np.ndarray):
            return {
                f'feature_{i+1}': float(val)
                for i, val in enumerate(self.feature_importance_cache)
            }

        return self.feature_importance_cache


########## 함수업데이트작업 ##########



