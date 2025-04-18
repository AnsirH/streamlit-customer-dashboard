# Streamlit Customer Dashboard

## 👨‍💻 프로젝트 필요성

### 효율적인 리텐션 전략 수립
- 고객 유지는 신규 고객 확보보다 최대 5배 저렴하다는 점에서,
이탈 가능 고객을 사전에 식별하고 대응하는 것은 E-Commerce 기업의 매출 안정성과 성장에 핵심적인 전략입니다.
이 프로젝트는 머신러닝 기반 예측 모델을 통해 이탈 위험군을 조기에 선별하고 효율적인 리텐션 전략 수립을 가능하게 합니다.

### 이탈 방지 전략의 자동화 가능성
- E-Commerce 산업은 고객 획득 비용(CAC)이 점점 증가하고 있으며,
그에 비해 충성 고객의 유지가 더욱 중요해지고 있습니다.
고객 이탈은 매출 감소로 직결되며, 데이터 기반의 이탈 예측은 마케팅 및 CS 부서의 정밀 타겟팅과 리소스 최적화에 큰 기여를 할 수 있습니다.



## 🎯 프로젝트 목표

### 팀으로서의 목표
- 본 프로젝트는 고객 이탈을 예측하기 위한 XGBoost 기반 모델을 구축하고, 전처리, 불균형 보정(SMOTE), 하이퍼파라미터 튜닝(GridSearchCV) 등의 과정을 통해 정확도 높은 이탈 예측 시스템을 완성하는 것을 목표로 합니다.

### 프로그램의 목표
- 본 프로젝트의 목표는 E-Commerce 환경에서 **머신러닝 모델(XGBoost)**을 통해 고객의 이탈 가능성을 예측하고, 이를 바탕으로 리텐션 전략 수립 및 마케팅 타겟팅의 효율성 향상에 기여하는 것입니다.
- 


## 기술 스택



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


## 📁 파일 구조
```
streamlit-customer-dashboard/
│
├── app.py                     # 메인 애플리케이션 진입점
├── config.py                  # 전역 설정 파일
├── requirements.txt           # 의존성 라이브러리
│
├── components/                # UI 컴포넌트
│   ├── __pycache__/
│   ├── animations.py          # 페이지 전환 애니메이션
│   └── header.py              # 헤더 컴포넌트
│
├── data/                      # 데이터 저장 디렉토리
│   ├── raw/                   # 원본 데이터
│   └── processed/             # 전처리된 데이터
│
├── logs/                      # 로그 파일 디렉토리
│
├── models/                    # 머신러닝 모델
│   ├── __pycache__/
│   ├── churn_model.py         # 이탈 예측 모델 클래스
│   └── xgboost_best_model.pkl     # 훈련된 XGBoost 모델
│
├── pages/                     # 대시보드 페이지
│   ├── __pycache__/
│   ├── all_data.py            # 전체 데이터 페이지 (78줄)
│   ├── customer_analysis.py   # 고객 분석 페이지 (109줄)
│   ├── customer_dashboard.py  # 고객 대시보드 페이지 (256줄)
│   └── prediction.py          # 현재 사용중인 예측 페이지 (705줄)
│
└── utils/                     # 유틸리티 함수
    ├── __pycache__/
    ├── cache.py               # 캐싱 기능 (35줄)
    ├── data_generator.py      # 샘플 데이터 생성 (63줄)
    ├── data_processor.py      # 데이터 전처리 (29줄)
    ├── logger.py              # 로깅 설정 (27줄)
    ├── model_predictor.py     # 모델 예측 기능 (233줄)
    └── visualizer.py          # 데이터 시각화 도구 (408줄)

```



## 📊 데이터셋 소개

본 프로젝트는 Kaggle에서 제공된 **E-Commerce 고객 데이터셋**을 기반으로, 고객 이탈 여부를 예측하기 위해 활용하였습니다.

- 총 샘플 수: 5,630명
- 총 컬럼 수: 20개
- 타깃 변수: `Churn` (0: 유지, 1: 이탈)

| 컬럼명                        | 설명                                       |
|-----------------------------|--------------------------------------------|
| CustomerID                  | 고객 고유 식별자                          |
| Churn                       | 이탈 여부 (0: 유지, 1: 이탈)               |
| Tenure                      | 고객 이용 기간 (개월)                      |
| PreferredLoginDevice        | 선호 로그인 기기 (Phone, Mobile Phone 등)  |
| CityTier                    | 고객 거주 도시 등급 (1~3)                  |
| WarehouseToHome             | 창고-집 거리 (단위 미상)                   |
| PreferredPaymentMode        | 선호 결제 방식                             |
| Gender                      | 성별                                       |
| HourSpendOnApp              | 하루 평균 앱 사용 시간                    |
| NumberOfDeviceRegistered    | 등록된 디바이스 수                         |
| PreferedOrderCat            | 선호 주문 카테고리                         |
| SatisfactionScore           | 고객 만족도 (1~5)                          |
| MaritalStatus               | 결혼 여부 (Single, Married)               |
| NumberOfAddress             | 등록된 배송지 수                           |
| Complain                    | 불만 제기 여부 (0 또는 1)                  |
| OrderAmountHikeFromlastYear | 전년도 대비 주문 금액 상승률              |
| CouponUsed                  | 쿠폰 사용 여부                             |
| OrderCount                  | 총 주문 횟수                               |
| DaySinceLastOrder           | 마지막 주문 이후 경과일                    |
| CashbackAmount              | 누적 캐시백 금액                           |

> ⚠ 일부 변수(`Tenure`, `WarehouseToHome`, `HourSpendOnApp` 등)는 결측값을 포함하고 있으며, 전처리 과정에서 평균 대체 또는 제거됨.



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


## ✅ 프로젝트 결론

본 프로젝트에서는 E-Commerce 고객 데이터를 기반으로 XGBoost 모델을 활용한 이탈 예측 시스템을 구축하였습니다.  
전처리 및 모델 튜닝을 통해 다음과 같은 우수한 성능을 달성하였습니다:

- **Accuracy**: 0.97 (Train) / 0.93 (Test)  
- **F1 Score**: 0.96 (Train) / 0.91 (Test)  
- **AUC Score**: 0.99 (Train) / 0.94 (Test)

과적합 여부는 F1 Score 기준으로 GAP이 **0.05로 수용 가능한 수준**이며,  
이는 모델이 학습 데이터에 과하게 의존하지 않고 **일반화 성능이 우수함**을 의미합니다.

결과적으로, 해당 모델은 **이탈 가능성이 높은 고객을 조기에 탐지**하여  
E-Commerce 비즈니스에서 **효율적인 리텐션 전략 수립과 마케팅 자원 최적화**에 기여할 수 있는 실질적인 도구로 활용 가능합니다.
