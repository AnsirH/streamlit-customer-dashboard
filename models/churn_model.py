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
            # 원본 데이터 로깅
            st.write(f"🔍 디버그 [전처리]: 원본 데이터 크기: {data.shape}")
            st.write(f"🔍 디버그 [전처리]: 원본 컬럼: {list(data.columns)}")
            
            # 모델 확인
            if self.model is None:
                st.error("⚠️ 모델이 로드되지 않았습니다. 전처리를 수행할 수 없습니다.")
                return data
                
            if not hasattr(self.model, 'feature_names_in_'):
                st.error("⚠️ 모델에 'feature_names_in_' 속성이 없습니다. 올바른 형식의 모델인지 확인하세요.")
                return data
            
            # 모델 특성 확인
            model_features = list(self.model.feature_names_in_)
            st.write(f"🔍 디버그 [전처리]: 모델 특성 수: {len(model_features)}")
            st.write(f"🔍 디버그 [전처리]: 모델 특성 샘플: {model_features[:5]}")
            
            # 범주형 특성 식별
            if hasattr(self, 'categorical_columns') and self.categorical_columns:
                categorical_cols = [col for col in self.categorical_columns if col in data.columns]
            else:
                # 데이터 유형에 기반하여 범주형 특성 자동 감지
                categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
                # 숫자형이지만 고유값이 적은 경우도 범주형으로 간주
                for col in data.select_dtypes(include=['int64', 'float64']).columns:
                    if col in data.columns and data[col].nunique() < 10:
                        categorical_cols.append(col)
            
            st.write(f"🔍 디버그 [전처리]: 범주형 특성: {categorical_cols}")
            
            # 원핫인코딩 적용 전 데이터 체크
            st.write(f"🔍 디버그 [전처리]: 원핫인코딩 전 데이터 크기: {data.shape}")
            
            # 원핫인코딩된 특성 사전 가져오기 
            encoded_features_dict = self.get_onehot_encoded_features()
            st.write(f"🔍 디버그 [전처리]: 원핫인코딩 그룹 수: {len(encoded_features_dict)}")
            
            # 모든 원핫인코딩 특성 목록
            all_encoded_features = []
            for feature_list in encoded_features_dict.values():
                all_encoded_features.extend(feature_list)
            
            st.write(f"🔍 디버그 [전처리]: 총 원핫인코딩 특성 수: {len(all_encoded_features)}")
            
            # 원핫인코딩 적용
            if categorical_cols:
                X_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=False)
                st.write(f"🔍 디버그 [전처리]: 원핫인코딩 후 데이터 크기: {X_encoded.shape}")
                st.write(f"🔍 디버그 [전처리]: 원핫인코딩 후 컬럼: {list(X_encoded.columns)[:10]}...")
            else:
                X_encoded = data.copy()
                st.write("🔍 디버그 [전처리]: 범주형 변수가 없어 원핫인코딩을 수행하지 않습니다.")
            
            # 누락된 특성 확인 및 처리
            missing_features = set(model_features) - set(X_encoded.columns)
            extra_features = set(X_encoded.columns) - set(model_features)
            
            st.write(f"🔍 디버그 [전처리]: 누락된 특성 수: {len(missing_features)}")
            if missing_features:
                st.write(f"🔍 디버그 [전처리]: 누락된 특성 목록: {list(missing_features)[:5]}...")
                # 누락된 특성에 0 채우기
                for feature in missing_features:
                    X_encoded[feature] = 0
            
            st.write(f"🔍 디버그 [전처리]: 추가 특성 수: {len(extra_features)}")
            if extra_features:
                st.write(f"🔍 디버그 [전처리]: 추가 특성 목록: {list(extra_features)[:5]}...")
                # 예상하지 않은 특성 제거
                X_encoded = X_encoded.drop(columns=extra_features)
            
            # 원핫인코딩 디버깅: 원본 특성과 인코딩된 특성 간의 매핑 확인
            if categorical_cols:
                st.write("🔍 디버그 [전처리]: 범주형 특성별 원핫인코딩 결과:")
                for cat_col in categorical_cols:
                    # 원본 데이터의 고유값
                    unique_values = data[cat_col].unique()
                    # 해당 특성으로 시작하는 원핫인코딩 컬럼
                    ohe_cols = [col for col in X_encoded.columns if col.startswith(f"{cat_col}_")]
                    st.write(f"  - {cat_col}: 원본값 {list(unique_values)} → 인코딩 컬럼 {ohe_cols}")
            
            # 모델 특성 순서에 맞게 재정렬
            processed_data = X_encoded[model_features]
            
            st.write(f"🔍 디버그 [전처리]: 최종 처리된 데이터 크기: {processed_data.shape}")
            
            return processed_data
            
        except Exception as e:
            st.error(f"⚠️ 데이터 전처리 중 오류 발생: {str(e)}")
            import traceback
            st.write(f"🔍 디버그 [전처리]: 상세 오류: {traceback.format_exc()}")
            # 원본 데이터 반환
            return data
    
    def get_onehot_encoded_features(self):
        """
        모델에서 사용하는 원핫인코딩된 특성 목록을 추출합니다.
        
        Returns:
            dict: 범주형 변수별 원핫인코딩된 특성 목록 (예: {'Gender': ['Gender_Male', 'Gender_Female']})
        """
        # 모델 체크
        if self.model is None:
            st.warning("⚠️ 모델이 로드되지 않아 원핫인코딩 특성을 분석할 수 없습니다.")
            return {}
            
        if not hasattr(self.model, 'feature_names_in_'):
            st.warning("⚠️ 모델에 feature_names_in_ 속성이 없습니다.")
            return {}
        
        # 모델의 모든 특성 목록 가져오기
        all_features = list(self.model.feature_names_in_)
        total_features = len(all_features)
        st.write(f"🔍 디버그 [원핫인코딩]: 모델의 총 특성 수: {total_features}")
        
        # 원핫인코딩된 특성을 찾기 위한 분석
        underscore_features = [f for f in all_features if '_' in f]
        st.write(f"🔍 디버그 [원핫인코딩]: 언더스코어(_)가 포함된 특성 수: {len(underscore_features)}")
        
        if len(underscore_features) == 0:
            st.warning("⚠️ 언더스코어가 포함된 특성이 없습니다. 원핫인코딩 특성이 아닐 수 있습니다.")
            # 특성 구조 더 자세히 분석
            has_digits = sum(1 for f in all_features if any(c.isdigit() for c in f))
            has_uppercase = sum(1 for f in all_features if any(c.isupper() for c in f))
            st.write(f"🔍 디버그 [원핫인코딩]: 숫자가 포함된 특성 수: {has_digits}")
            st.write(f"🔍 디버그 [원핫인코딩]: 대문자가 포함된 특성 수: {has_uppercase}")
            st.write(f"🔍 디버그 [원핫인코딩]: 특성 샘플: {all_features[:10]}")
            # 특성 값 분포 확인
            if hasattr(self.model, 'feature_importances_'):
                top_indices = np.argsort(self.model.feature_importances_)[-5:]
                top_features = [all_features[i] for i in top_indices]
                top_importances = [self.model.feature_importances_[i] for i in top_indices]
                st.write("🔍 디버그 [원핫인코딩]: 상위 5개 중요 특성:")
                for f, imp in zip(top_features, top_importances):
                    st.write(f"  - {f}: {imp:.4f}")
            return {}
        
        # 가능한 범주형 변수 접두사
        possible_prefixes = [
            'Gender', 'Marital', 'City', 'Complain', 'CityTier', 
            'PreferredLogin', 'Login', 'PreferredPayment', 'Payment', 
            'PreferedOrder', 'OrderCat', 'Status', 'Occupation'
        ]
        
        # 각 접두사별로 특성 탐색
        encoded_features = {}
        found_any = False
        
        for prefix in possible_prefixes:
            # 접두사로 시작하는 특성 찾기
            prefix_features = [f for f in all_features if f.startswith(f"{prefix}_")]
            if prefix_features:
                found_any = True
                encoded_features[prefix] = prefix_features
                st.write(f"🔍 디버그 [원핫인코딩]: '{prefix}'에 대한 원핫인코딩 특성 발견: {len(prefix_features)}개")
                st.write(f"🔍 디버그 [원핫인코딩]: '{prefix}' 범주값: {[f.split('_', 1)[1] for f in prefix_features]}")
        
        # 접두사 다시 확인 (대소문자 무시)
        if not found_any:
            st.warning("⚠️ 일반적인 접두사로 원핫인코딩 특성이 발견되지 않았습니다. 추가 분석 중...")
            
            # 언더스코어로 구분된 모든 접두사 분석
            all_prefixes = {}
            for feature in underscore_features:
                prefix = feature.split('_')[0]
                if prefix not in all_prefixes:
                    all_prefixes[prefix] = []
                all_prefixes[prefix].append(feature)
            
            # 접두사별 특성 수 분석
            st.write(f"🔍 디버그 [원핫인코딩]: 발견된 모든 접두사: {list(all_prefixes.keys())}")
            for prefix, features in all_prefixes.items():
                if len(features) >= 2:  # 최소 2개 이상의 특성이 있으면 원핫인코딩 가능성 있음
                    encoded_features[prefix] = features
                    st.write(f"🔍 디버그 [원핫인코딩]: 잠재적인 원핫인코딩 접두사 '{prefix}': {len(features)}개 특성")
                    
            # 언더스코어로 시작하는 특성 분석
            starts_with_underscore = [f for f in all_features if f.startswith('_')]
            if starts_with_underscore:
                st.write(f"🔍 디버그 [원핫인코딩]: 언더스코어로 시작하는 특성: {len(starts_with_underscore)}개")
        
        if not encoded_features:
            st.warning("⚠️ 어떤 방식으로도 원핫인코딩 특성을 찾을 수 없습니다.")
            # 전체 특성 목록의 처음 10개 출력
            st.write(f"🔍 디버그 [원핫인코딩]: 특성 목록 샘플: {all_features[:10]}")
        
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



