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
    """이탈 예측을 위한 클래스"""
    
    def __init__(self):
        self.model = None
        try:
            # 모델 파일 경로 설정
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = Path(current_dir) / "xgboost_best_model.pkl"
            
            # 모델 로드
            if model_path.exists():
                self.model = joblib.load(model_path)
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
    
    def predict(self, input_data):
        """입력 데이터에 대한 이탈 확률을 예측합니다."""
        if self.model is None:
            return None
        
        try:
            # 예측 수행
            churn_prob = self.model.predict_proba(input_data)[:, 1]
            return churn_prob[0]  # 단일 고객에 대한 예측이므로 첫 번째 값 반환
        except Exception as e:
            print(f"예측 중 오류 발생: {str(e)}")
            return None


########## 함수영역역 ##########

# 모델 로드 함수
# MODEL_PATH = Path("models/xgb_best_model.pkl")  # GitHub 프로젝트 내 상대 경로 사용 권장

# ===============================
# ✅ 모델 로드 및 예측 함수
# ===============================
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = Path(current_dir) / "xgboost_best_model.pkl"

def load_churn_model(model_path: str = None):
    """
    Load the trained churn prediction model.
    
    Args:
        model_path: Path to the model file. Default is models/xgboost_best_model.pkl
        
    Returns:
        Trained model
    """
    if model_path is None:
        model_path = Path(current_dir) / "xgboost_best_model.pkl"
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





########## 함수업데이트작업 ##########



