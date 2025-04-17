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
            tuple: (예측 클래스, 이탈 확률)
        """
        try:
            # 데이터 전처리
            processed_df = self._preprocess_data(input_df)
            
            # 예측 수행
            y_pred = self.model.predict(processed_df)
            y_proba = self.model.predict_proba(processed_df)[:, 1]  # 이탈 확률
            
            # 성공적으로 예측한 경우 특성 중요도 계산
            try:
                self._compute_feature_importance(processed_df)
            except Exception as e:
                # 특성 중요도 계산 실패해도 예측 결과는 반환
                pass
            
            return y_pred, y_proba
        except Exception as e:
            logger.error(f"예측 오류: {str(e)}")
            # 예측 실패 시 빈 배열이 아닌 None 반환
            return None, None
    
    def _default_prediction(self):
        """기본 예측값 반환"""
        return np.array([0]), np.array([0.5])
    
    def _preprocess_data(self, data):
        """
        모델 예측을 위해 입력 데이터를 전처리합니다.
        
        Args:
            data (pd.DataFrame): 전처리할 입력 데이터
            
        Returns:
            pd.DataFrame: 모델 입력에 맞게 전처리된 데이터
        """
        try:
            # 1. 원본 데이터 로깅
            st.write(f"🔍 [전처리]: 원본 데이터 크기: {data.shape}")
            st.write(f"🔍 [전처리]: 원본 컬럼: {list(data.columns)}")
            
            # 2. 데이터 복사 (원본 데이터 변경 방지)
            processed_data = data.copy()
            
            # 3. 모델 확인
            if self.model is None:
                st.error("⚠️ 모델이 로드되지 않았습니다. 전처리를 수행할 수 없습니다.")
                return processed_data
                
            if not hasattr(self.model, 'feature_names_in_'):
                st.error("⚠️ 모델에 'feature_names_in_' 속성이 없습니다.")
                return processed_data
            
            # 4. 모델 특성 확인
            model_features = list(self.model.feature_names_in_)
            st.write(f"🔍 [전처리]: 모델 특성 수: {len(model_features)}")
            st.write(f"🔍 [전처리]: 모델 특성 샘플: {model_features[:5] if len(model_features) >= 5 else model_features}")
            
            # 5. 컬럼명 표준화 (대소문자 및 공백 처리)
            # 이 부분은 실제 데이터와 모델의 컬럼명 불일치 문제가 있을 경우 주석 해제
            # processed_data.columns = [col.replace(' ', '').strip() for col in processed_data.columns]
            
            # 6. 원핫인코딩된 특성 구조 가져오기
            encoded_features_dict = self.get_onehot_encoded_features()
            
            # 7. 범주형 특성 식별
            categorical_cols = []
            
            # 7.1 먼저 원핫인코딩된 특성에서 범주형 변수명 추출
            for prefix in encoded_features_dict.keys():
                if prefix in processed_data.columns:
                    categorical_cols.append(prefix)
            
            # 7.2 데이터 타입 기반 범주형 변수 추가 식별
            if not categorical_cols:  # 범주형 변수가 아직 식별되지 않았다면
                st.warning("⚠️ 원핫인코딩 구조에서 범주형 변수를 식별하지 못했습니다. 데이터 타입 기반으로 식별합니다.")
                # 문자열 타입 컬럼
                categorical_cols = processed_data.select_dtypes(include=['object', 'category']).columns.tolist()
                # 고유값이 적은 숫자형 컬럼도 범주형으로 처리
                for col in processed_data.select_dtypes(include=['int64', 'float64']).columns:
                    if processed_data[col].nunique() < 10:
                        categorical_cols.append(col)
            
            # 8. 범주형 변수 정보 출력
            st.write(f"🔍 [전처리]: 범주형 변수: {categorical_cols}")
            
            # 9. 각 범주형 변수의 고유값 확인 (디버깅)
            for col in categorical_cols:
                if col in processed_data.columns:
                    unique_vals = processed_data[col].unique()
                    st.write(f"🔍 [전처리]: '{col}' 고유값: {list(unique_vals)}, 개수: {len(unique_vals)}")
            
            # 10. 비표준 값 처리 (예: 대소문자 불일치, 공백 등)
            for col in categorical_cols:
                if col in processed_data.columns:
                    # 문자열인 경우에만 처리
                    if processed_data[col].dtype == 'object':
                        # 공백 제거 및 대소문자 표준화
                        processed_data[col] = processed_data[col].astype(str).str.strip().str.title()
            
            # 11. 원핫인코딩 수행
            X_encoded = pd.get_dummies(processed_data, columns=categorical_cols, drop_first=False)
            
            # 12. 원핫인코딩 결과 확인
            ohe_cols = [col for col in X_encoded.columns if '_' in col and any(col.startswith(f"{c}_") for c in categorical_cols)]
            st.write(f"🔍 [전처리]: 생성된 원핫인코딩 컬럼 수: {len(ohe_cols)}")
            st.write(f"🔍 [전처리]: 원핫인코딩 컬럼 샘플: {ohe_cols[:5] if ohe_cols else '없음'}")
            
            # 13. 범주형 변수별 생성된 컬럼 수 확인
            for col in categorical_cols:
                gen_cols = [c for c in X_encoded.columns if c.startswith(f"{col}_")]
                if gen_cols:
                    st.write(f"🔍 [전처리]: '{col}'에서 생성된 원핫인코딩 컬럼 수: {len(gen_cols)}")
                    st.write(f"🔍 [전처리]: '{col}' 원핫인코딩 컬럼: {gen_cols}")
                    if len(gen_cols) <= 1:
                        st.warning(f"⚠️ '{col}'에 대해 하나 이하의 원핫인코딩 컬럼만 생성되었습니다!")
            
            # 14. 모델에 필요한 특성 확인 및 처리
            missing_features = set(model_features) - set(X_encoded.columns)
            extra_features = set(X_encoded.columns) - set(model_features)
            
            # 15. 누락된 특성 처리
            if missing_features:
                st.warning(f"⚠️ 모델에 필요한 특성 {len(missing_features)}개가 누락되었습니다.")
                for feature in missing_features:
                    X_encoded[feature] = 0
                    # 누락된 특성이 원핫인코딩 컬럼인지 확인
                    if '_' in feature:
                        prefix = feature.split('_')[0]
                        cat_val = feature.split('_', 1)[1]
                        st.write(f"🔍 [전처리]: 누락된 원핫인코딩 컬럼 '{feature}' 추가 (변수: {prefix}, 값: {cat_val})")
            
            # 16. 추가 특성 제거
            if extra_features:
                st.warning(f"⚠️ 모델에 없는 추가 특성 {len(extra_features)}개가 있습니다.")
                X_encoded = X_encoded.drop(columns=list(extra_features))
            
            # 17. 모델 특성 순서에 맞게 재정렬
            # 이 과정에서 특성 불일치 문제가 발생하면 아래 코드를 주석 해제하고 문제를 확인
            try:
                final_data = X_encoded[model_features]
            except KeyError as e:
                st.error(f"⚠️ 특성 불일치 오류: {str(e)}")
                
                # 문제가 있는 특성 확인
                missing_after_align = set(model_features) - set(X_encoded.columns)
                if missing_after_align:
                    st.write(f"🔍 [전처리]: 누락된 특성: {list(missing_after_align)}")
                
                # 불일치 문제가 발생할 경우 원본 데이터 반환
                return processed_data
            
            # 18. 최종 결과 확인
            st.write(f"🔍 [전처리]: 최종 처리된 데이터 크기: {final_data.shape}")
            
            return final_data
            
        except Exception as e:
            st.error(f"⚠️ 데이터 전처리 중 오류: {str(e)}")
            import traceback
            st.write(f"🔍 [전처리 오류]: {traceback.format_exc()}")
            # 원본 데이터 복사본 반환
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



