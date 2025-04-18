# Streamlit Customer Dashboard

## 프로젝트 필요성

# 효율적인 리텐션 전략 수립
- 고객 유지는 신규 고객 확보보다 최대 5배 저렴하다는 점에서,
이탈 가능 고객을 사전에 식별하고 대응하는 것은 E-Commerce 기업의 매출 안정성과 성장에 핵심적인 전략입니다.
이 프로젝트는 머신러닝 기반 예측 모델을 통해 이탈 위험군을 조기에 선별하고 효율적인 리텐션 전략 수립을 가능하게 합니다.



## 프로젝트 목표



## 📦 요구사항

- Python 3.9 이상, 3.12 미만
- pip 24.0 이상
    ```
    - Python 3.12 호환 버전
    numpy>=1.26.0
    pandas>=2.1.0
    streamlit>=1.31.0
    plotly>=5.18.0
    matplotlib>=3.8.0
    
    - 데이터 처리 및 분석
    openpyxl>=3.1.2
    
    - 머신러닝 및 모델링
    xgboost>=2.0.0
    lightgbm>=4.1.0
    imbalanced-learn>=0.11.0
    joblib>=1.3.0
    shap>=0.42.0
    scikit-learn>=1.3.0   # 의존성 충돌 방지용 권장 추가
    ```
## 💾 설치 방법

1. 가상환경 생성 및 활성화:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```

2. 패키지 설치:    
    2-1. 일괄 설치
    
    ```bash
    pip install -r requirements.txt
    ```
    
    2-2. 개별설치
    
    ```bash
    # 기본 환경
    pip install numpy>=1.26.0 pandas>=2.1.0 streamlit>=1.31.0 plotly>=5.18.0 matplotlib>=3.8.0 
    ```
    ```bash
    # 데이터 처리 및 분석
    pip install openpyxl>=3.1.2
    ```
    ```bash
    # 머신러닝 및 모델링
    pip install xgboost>=2.0.0 lightgbm>=4.1.0 imbalanced-learn>=0.11.0 joblib>=1.3.0 shap>=0.42.0 scikit-learn>=1.3.0 
    ```

3. 앱 실행:
    ```bash
    streamlit run app.py
    ```

## 📋 주요 기능
    ```
    - 고객 이탈 예측
    - 신규 고객 이탈 위험성 예측
    - 데이터 시각화
    - 통계 분석
    ```


## 🎯 프로젝트 개요
    ```
    # 원본 데이터
    - **목적**: 고객의 이탈 가능성을 사전에 예측하여 유지 전략 수립에 활용
    - **데이터 출처**: Kaggle 공개 데이터셋
    - **샘플 수**: 5,630명
    - **피처 수**: 20개 (ID 포함)
    - **타깃**: `Churn` (0: 유지, 1: 이탈)
    
    # 원핫 인코딩 이후 데이터
    - **샘플 수**: 5,630명
    - **피처 수**: 28개 (ID 포함)
    
    
    ```

## ⚙️ 데이터 전처리
    ```
    1. 수치형 변수 결측치 평균 대체
    2. 수치형 변수 이상치 IQR 필터링
    3. 범주형 변수 One-hot인코딩
    4. SMOTE OverSampling으로 클래스 불균형 처리
    5. StandardScaler 적용
    ```

## 🧠 모델링
    ```
    - 사용 모델: XGBoostClassifier (최종 선택)
    - 비교 모델: LogisticRegression, KNN, SVC, NaiveBayes
    - 교차검증: 5-Fold
    - 하이퍼파라미터 튜닝: GridSearchCV
    
    ```
