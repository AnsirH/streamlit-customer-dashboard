# Streamlit Customer Dashboard

## 🧑‍💻 Team 소개

본 프로젝트는 5인 팀으로 구성된 전자상거래( E-Commerce ) 고객 이탈 예측 프로젝트입니다.

| 팀원 사진 | 이름 | 역할 |
|-----------|------|------|
| <img src="images\김도윤_이미지.png" width="120"/> | **김도윤** | 예측 모델 생성 및 튜닝, 솔루션 탭 제작 |
| <img src="images\윤권_이미지.png" width="120"/> | **윤권** | 예측 모델 제작, 예측 탭 제작 |
| <img src="images\이정민_이미지.png" width="120"/> | **이정민** | 스토리 보드, 고객 분석 탭, 발표 자료 제작 |
| <img src="images\이준석_이미지.png" width="120"/> | **이준석** | 전체 데이터 분석 탭 제작 및 모델링 과정 분석 |
| <img src="images\허한결_이미지.png" width="120"/> | **허한결** | 프로젝트 설계 및 고객 분석 탭 제작 |


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
- 고객 이탈을 예측하기 위한 XGBoost 기반 모델을 구축하고, 전처리, 불균형 보정(SMOTE), 하이퍼파라미터 튜닝(GridSearchCV) 등의 과정을 통해 정확도 높은 이탈 예측 시스템을 완성하는 것을 목표로 합니다.

### 프로젝트의 목표
- E-Commerce 환경에서 **머신러닝 모델(XGBoost)**을 통해 고객의 이탈 가능성을 예측하고, 이를 바탕으로 리텐션 전략 수립 및 마케팅 타겟팅의 효율성 향상에 기여합니다.


## 🛠 Tech Stack

<p align="left">

  <!-- 언어 -->
  <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  </a>

  <!-- 데이터 처리 -->
  <a href="https://numpy.org/" target="_blank">
    <img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  </a>
  <a href="https://pandas.pydata.org/" target="_blank">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  </a>
  <a href="https://openpyxl.readthedocs.io/" target="_blank">
    <img src="https://img.shields.io/badge/OpenPyXL-Data%20Handling-yellow?style=for-the-badge"/>
  </a>

  <!-- 시각화 -->
  <a href="https://matplotlib.org/" target="_blank">
    <img src="https://img.shields.io/badge/Matplotlib-007ACC?style=for-the-badge&logo=matplotlib&logoColor=white"/>
  </a>
  <a href="https://plotly.com/" target="_blank">
    <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  </a>
  <a href="https://streamlit.io/" target="_blank">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  </a>

  <!-- 머신러닝 -->
  <a href="https://xgboost.readthedocs.io/" target="_blank">
    <img src="https://img.shields.io/badge/XGBoost-EC1C24?style=for-the-badge"/>
  </a>
  <a href="https://lightgbm.readthedocs.io/" target="_blank">
    <img src="https://img.shields.io/badge/LightGBM-9ACD32?style=for-the-badge"/>
  </a>
  <a href="https://scikit-learn.org/" target="_blank">
    <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  </a>
  <a href="https://imbalanced-learn.org/" target="_blank">
    <img src="https://img.shields.io/badge/Imbalanced--Learn-008B8B?style=for-the-badge"/>
  </a>
  <a href="https://joblib.readthedocs.io/" target="_blank">
    <img src="https://img.shields.io/badge/Joblib-C0C0C0?style=for-the-badge"/>
  </a>
  <a href="https://shap.readthedocs.io/" target="_blank">
    <img src="https://img.shields.io/badge/SHAP-Model%20Explainability-blueviolet?style=for-the-badge"/>
  </a>

</p>




## 화면구성


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
│   ├── all_data.py            # 전체 데이터 페이지
│   ├── customer_analysis.py   # 고객 분석 페이지
│   ├── customer_dashboard.py  # 고객 대시보드 페이지
│   └── prediction.py          # 현재 사용중인 예측 페이지
│
└── utils/                     # 유틸리티 함수
    ├── __pycache__/
    ├── cache.py               # 캐싱 기능
    ├── data_generator.py      # 샘플 데이터 생성
    ├── data_processor.py      # 데이터 전처리
    ├── logger.py              # 로깅 설정
    ├── model_predictor.py     # 모델 예측 기능
    └── visualizer.py          # 데이터 시각화 도구

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
    - 12개 모델 테스트 후 모델 선택   
    - 사용 모델:  
                🔹 선형/기초 모델
                Logistic Regression             
                K-Nearest Neighbors (KNN)                
                Support Vector Classifier (SVC) 
                Gaussian Naive Bayes (GaussianNB)
                
                🔹 트리 기반 모델
                Decision Tree       
                Random Forest
                Extra Trees
                
                🔹 앙상블 모델
                Gradient Boosting
                AdaBoost
                Bagging
                XGBoost
                LightGBM
                
    - 최종 모델: XGBoostClassifier (최종 선택)
    - 교차검증: 5-Fold
    - 비교 모델: LogisticRegression, KNN, SVC, NaiveBayes
    - 하이퍼파라미터 튜닝: GridSearchCV
    ```


## ✅ 프로젝트 결과

본 프로젝트에서는 E-Commerce 고객 데이터를 기반으로 XGBoost 모델을 활용한 이탈 예측 시스템을 구축하였습니다.  
전처리 및 모델 튜닝을 통해 다음과 같은 우수한 성능을 달성하였습니다:

- **Accuracy**: 0.97 (Train) / 0.93 (Test)  
- **F1 Score**: 0.96 (Train) / 0.91 (Test)  
- **AUC Score**: 0.99 (Train) / 0.94 (Test)

과적합 여부는 F1 Score 기준으로 GAP이 **0.05로 수용 가능한 수준**이며,  
이는 모델이 학습 데이터에 과하게 의존하지 않고 **일반화 성능이 우수함**을 의미합니다.

결과적으로, 해당 모델은 **이탈 가능성이 높은 고객을 조기에 탐지**하여  
E-Commerce 비즈니스에서 **효율적인 리텐션 전략 수립과 마케팅 자원 최적화**에 기여할 수 있는 실질적인 도구로 활용 가능합니다.

## 📌 프로젝트 결론

- 고객 이탈에는 여러 요인이 복합적으로 작용하며, 특히 다음과 같은 특성이 이탈과 강한 상관관계를 보였습니다:
  - **창고와 고객 간 거리(WarehouseToHome)**: 멀수록 이탈 가능성 증가
  - **고객 이용 기간(Tenure)**: 짧을수록 이탈 가능성 높음
  - **마지막 주문 이후 경과일(DaySinceLastOrder)**: 길수록 이탈 가능성 증가
  - **디바이스 수(NumberOfDeviceRegistered)**: 적을수록 충성도 낮음
  - **결혼 여부(MaritalStatus)**: 기혼 고객이 상대적으로 이탈 확률 낮음

- XGBoost 모델은 **높은 정확도와 재현율**을 기반으로 이탈 고객을 신뢰성 있게 분류함

- **모델의 일반화 성능이 우수(F1 GAP = 0.05)**하여 실제 서비스에 적용 가능성이 높음

## 🚀 기대 효과

- **고객 유지율 향상**: 이탈 가능성이 높은 고객을 조기 식별하고 리마케팅에 활용 가능
- **마케팅 비용 절감**: 타겟 마케팅 전략 수립으로 불필요한 비용 절감
- **LTV(Lifetime Value) 극대화**: 이탈 방지 → 고객 생애가치 향상
- **운영 최적화**: 고객 이탈 예측을 기반으로 CRM, CS 리소스 우선순위 조정 가능
- **의사결정 자동화**: 향후 실시간 이탈 예측 시스템 기반으로 자동 대응 가능성 확보


## 🧩 한계점 및 개선 가능성

### ❗ 한계점

- **정적(Static) 데이터 기반**: 모델은 특정 시점의 고정된 고객 정보만을 기반으로 학습됨.  
  → 고객 행동의 **시간 흐름에 따른 변화**(예: 최근 활동 패턴, 구매 빈도 추세 등)는 반영되지 않음.

- **텍스트/리뷰 데이터 미활용**: 고객 불만, 피드백 등 **비정형 데이터**가 제외되어 있음.  
  → 감성 분석이나 후기 내용 분석이 포함된다면 더 풍부한 해석 가능.

- **실제 운영 환경 반영 부족**: 실시간 API 연동, 예측 결과 기반 알림/조치 등은 구현되지 않음.  
  → 예측 결과를 **즉시 활용할 수 있는 시스템 구조로 발전 필요**.

- **모델 설명력(해석력) 부족**: XGBoost는 강력하지만 **비선형 모델로 해석력이 낮음**.  
  → SHAP 등의 도구로 변수 영향도 분석은 했지만, 비전문가 대상 설명은 어려울 수 있음.

---

### 🔧 개선 가능성

- **행동 로그 데이터 통합**: 클릭 수, 장바구니 추가, 최근 앱 방문 등 **고객 행동 기반 피처 추가**  
  → 시간 기반 특성이 들어가면 예측 성능 향상 가능

- **딥러닝 기반 시계열 모델 도입**: RNN, LSTM 등으로 고객 이탈 **시점 예측**까지 가능해짐

- **비정형 데이터 활용**: 리뷰, 설문, 콜센터 기록 등 **텍스트 데이터 분석** 병행

- **실시간 예측 시스템 연계**: Streamlit이나 FastAPI 기반으로 모델 서빙 및 즉시 대응 구조 구현

- **AutoML 기반 성능 향상 실험**: 다양한 앙상블/튜닝을 자동화하여 **모델 성능의 상한선 탐색**




## ✏️한줄회고
💬 [김도윤]
처음으로 GitHub Desktop을 통한 협업과 코드 버전 관리가 매우 인상 깊었습니다. 머신러닝 모델을 직접 학습시키고, 이를 Streamlit 앱으로 시각화하며 하나의 완성된 서비스를 만들어 보는 경험이 정말 뜻깊었습니다. 특히 모델 선정, 성능 개선, SHAP 기반 해석까지 전 과정을 직접 겪으며 실무 감각을 키울 수 있었고, 동료들과의 협업을 통해 부족한 부분도 많이 배울 수 있었습니다. 모두 정말 고생 많으셨습니다!

💬 [허한결]
이번 프로젝트를 하며 단순한 모델 구현을 넘어 실제 사용자에게 보여줄 수 있는 예측 결과를 만들어야 한다는 점에서 많이 고민하고 성장할 수 있었습니다. 개인적으로는 SHAP를 활용하여 모델의 예측 결과를 분석하고, 그에 맞는 솔루션을 도출하는 과정을 구현하면서 가장 성취감을 느꼈고, Streamlit을 활용해 시각화까지 해보니 초반 기획 단계에서 나왔던 아이디어들이 실제 눈 앞에 그려지니까 너무 재밌었습니다. 혼자서는 빠르게 갈 수 있지만 여럿이서는 더 멀리, 많이 갈 수 있다는 걸 다시 깨달으면서 같이 작업한 팀원분들 정말 고생하셨고 감사합니다!

💬 [이준석]
데이터 분석과 모델링도 중요했지만, 예측 결과를 직관적으로 전달하기 위한 UI/UX 설계가 정말 많은 생각을 하게 했던 프로젝트였습니다. 특히 Streamlit을 활용해 앱 형태로 구현하고, SHAP 분석을 시각화하여 사용자에게 의미 있는 설명을 주려는 시도가 인상 깊었습니다. GitHub으로 버전 관리하며 협업한 것도 실무에 가까운 경험이었고, 혼자였다면 절대 완성할 수 없었던 결과물이라고 생각합니다. 팀원들에게 정말 감사합니다!

💬 [윤권]
이번 프로젝트는 제가 학습한 머신러닝 기술이 실제 앱 개발에 어떻게 적용될 수 있는지를 보여준 귀중한 경험이었습니다. 모델 선정부터 하이퍼파라미터 튜닝, 앙상블 구성, 그리고 Streamlit UI 구현까지 전반적인 프로세스를 한 번에 경험할 수 있어 좋았습니다. 특히 GitHub Desktop을 사용한 협업에서 나의 부족한 점을 많이 깨달았고, 더 발전하고 싶다는 동기부여가 생겼습니다. 팀원들과 함께해서 정말 즐거운 시간이었습니다.

💬 [이정민]
처음엔 데이터가 복잡하고 모델이 너무 많아 어떻게 해야 할지 막막했지만, 팀원들과 역할을 나누고 함께 고민하면서 하나하나 풀어나갈 수 있었습니다. 저는 주로 Streamlit UI와 모델 연결 파트를 담당했는데, 사용자의 입장에서 결과를 어떻게 보여주는지가 얼마나 중요한지를 느낄 수 있었어요. GitHub Desktop을 통해 협업하며 서로의 코드를 리뷰하고 개선해나간 것도 큰 배움이었습니다. 모두 고생 많으셨고, 다음 프로젝트도 기대됩니다!
