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
    
    def __init__(self):
        """모델을 로드하고 초기화합니다."""
        self.model = None
        self.feature_importance_cache = {}
        self.model_path = Path(__file__).parent / "xgb_best_model.pkl"
        st.write(f"DEBUG: 모델 경로: {self.model_path}")
        st.write(f"DEBUG: 모델 파일 존재: {self.model_path.exists()}")
        try:
            self.load_model()
            st.write("DEBUG: 모델 로드 성공")
        except Exception as e:
            logger.error(f"모델 로드 오류: {str(e)}")
            st.error(f"모델 로드 중 오류가 발생했습니다: {str(e)}")
    
    def load_model(self):
        """모델 파일을 로드합니다."""
        try:
            if not self.model_path.exists():
                logger.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                st.error(f"모델 파일을 찾을 수 없습니다: {self.model_path}")
                return False
            
            st.write(f"DEBUG: 모델 파일 크기: {os.path.getsize(self.model_path)} 바이트")
            self.model = joblib.load(self.model_path)
            
            # 모델 정보 출력
            st.write(f"DEBUG: 모델 타입: {type(self.model)}")
            st.write(f"DEBUG: 모델 사용 가능 메서드: {dir(self.model)[:5]}...")
            
            if hasattr(self.model, 'feature_importances_'):
                st.write(f"DEBUG: 특성 중요도 수: {len(self.model.feature_importances_)}")
            
            logger.info(f"모델 로드 성공: {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            st.error(f"모델 로드 실패: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False
    
    def _default_prediction(self):
        """기본 예측값 반환"""
        return np.array([0]), np.array([0.5])
    
    def _test_prediction(self, input_df):
        """테스트용 예측 함수 - 실제 모델 없이 입력값에 따라 다른 예측 반환"""
        # 간단한 규칙 기반 예측 (실제 모델 사용하지 않음)
        if 'Tenure' in input_df.columns and 'SatisfactionScore' in input_df.columns:
            tenure = input_df['Tenure'].iloc[0]
            satisfaction = input_df['SatisfactionScore'].iloc[0]
            
            # 간단한 규칙: 거래기간 길고 만족도 높으면 이탈 확률 낮음
            base_prob = 0.5
            tenure_effect = min(0.3, tenure / 100.0)  # 최대 0.3 감소
            satisfaction_effect = min(0.2, satisfaction / 25.0)  # 최대 0.2 감소
            
            churn_prob = base_prob - tenure_effect - satisfaction_effect
            # 확률 범위 제한
            churn_prob = max(0.1, min(0.9, churn_prob))
            
            st.write(f"DEBUG: 테스트 예측 - 거래기간: {tenure}, 만족도: {satisfaction}, 확률: {churn_prob}")
            
            return np.array([1 if churn_prob > 0.5 else 0]), np.array([churn_prob])
        
        # 기본값 반환
        return np.array([0]), np.array([0.5])
    
    def predict(self, input_df):
        """
        이탈 예측을 수행합니다.
        
        Args:
            input_df (pandas.DataFrame): 예측할 고객 데이터
            
        Returns:
            tuple: (예측 클래스, 이탈 확률)
        """
        try:
            st.write("DEBUG: 입력 데이터 컬럼 -", input_df.columns.tolist())
            
            # 테스트 모드 사용 - 실제 모델 대신 테스트 예측 사용
            st.write("DEBUG: 테스트 모드로 예측 수행")
            return self._test_prediction(input_df)
            
            # 아래 코드는 모델 디버깅 완료 후 주석 해제
            """
            # 모델이 없으면 로드 시도
            if self.model is None:
                st.write("DEBUG: 모델이 없어 로드 시도")
                self.load_model()
                
            # 모델 로드 실패 시 기본값 반환
            if self.model is None:
                st.write("DEBUG: 모델 로드 실패, 기본값 반환")
                return self._default_prediction()
            
            # 데이터 전처리
            processed_df = self._preprocess_data(input_df)
            
            # 예측 수행
            try:
                st.write("DEBUG: 예측 시작")
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
            """
                
        except Exception as e:
            logger.error(f"예측 처리 중 오류: {str(e)}")
            st.error(f"예측 처리 중 오류: {str(e)}")
            return self._default_prediction()
    
    def _preprocess_data(self, input_df):
        """
        입력 데이터를 전처리합니다.
        
        Args:
            input_df (pandas.DataFrame): 원본 입력 데이터
            
        Returns:
            pandas.DataFrame: 전처리된 데이터
        """
        # CustomerID 제거 (예측에 사용되지 않음)
        df = input_df.copy()
        if 'CustomerID' in df.columns:
            df = df.drop('CustomerID', axis=1)
        
        return df
    
    def _compute_feature_importance(self, input_df):
        """모델의 feature_importances_ 속성을 사용하여 특성 중요도를 계산합니다."""
        if self.model is None:
            st.write("DEBUG: 모델이 없어 특성 중요도 계산 불가")
            return
            
        try:
            # SHAP 사용하지 않고 직접 feature_importances_ 사용
            st.write("DEBUG: feature_importances_ 속성 확인")
            if hasattr(self.model, 'feature_importances_'):
                st.write("DEBUG: feature_importances_ 속성 있음")
                importance_dict = {}
                for i, col in enumerate(input_df.columns):
                    if i < len(self.model.feature_importances_):
                        importance_dict[col] = self.model.feature_importances_[i]
                self.feature_importance_cache = importance_dict
                st.write(f"DEBUG: 특성 중요도 계산 결과: {importance_dict}")
            else:
                st.write("DEBUG: feature_importances_ 속성 없음, 기본값 사용")
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
            st.write(f"DEBUG: 특성 중요도 계산 중 오류: {str(e)}")
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


########## 함수영역역 ##########

# 모델 로드 함수
# MODEL_PATH = Path("models/xgb_best_model.pkl")  # GitHub 프로젝트 내 상대 경로 사용 권장

# ===============================
# ✅ 모델 로드 및 예측 함수
# ===============================
MODEL_PATH = Path(__file__).parent / "xgb_best_model.pkl"

def load_churn_model(model_path=MODEL_PATH):
    if not model_path.exists():
        raise FileNotFoundError(f"\u274c 모델 파일을 찾을 수 없습니다: {model_path}")
    model = joblib.load(model_path)
    return model

def predict_churn(model, input_df: pd.DataFrame):
    y_pred = model.predict(input_df)
    y_proba = model.predict_proba(input_df)[:, 1]  # 이탈 확률
    return y_pred, y_proba

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





########## 함수업데이트작업 ##########



