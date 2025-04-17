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
            self.model_path = Path(__file__).parent / "xgb_best_model.pkl"
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
                logger.error(f"예측 오류: {str(e)}")
                return self._default_prediction()
                
        except Exception as e:
            logger.error(f"예측 처리 중 오류: {str(e)}")
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
        # 입력 데이터 복사
        df = input_df.copy()
        
        # CustomerID 제거 (예측에 사용되지 않음)
        columns_to_remove = ['CustomerID', 'customer_id', 'cust_id', 'id']
        for col in columns_to_remove:
            if col in df.columns:
                df = df.drop(col, axis=1)
        
        # Complain 불리언 변환 (예/아니오 -> 0/1)
        if 'Complain' in df.columns and isinstance(df['Complain'].iloc[0], str):
            df['Complain'] = df['Complain'].apply(lambda x: 1 if x == '예' else 0)
        
        return df
    
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
MODEL_PATH = Path(__file__).parent / "xgb_best_model.pkl"

def load_churn_model(model_path: str = None):
    """
    Load the trained churn prediction model.
    
    Args:
        model_path: Path to the model file. Default is models/xgb_best_model.pkl
        
    Returns:
        Trained model
    """
    if model_path is None:
        model_path = Path(__file__).parent / "xgb_best_model.pkl"
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



