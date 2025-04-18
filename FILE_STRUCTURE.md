# 고객 대시보드 파일 구조

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
│   ├── prediction.py          # 현재 사용중인 예측 페이지 (705줄)
│   ├── prediction1.py         # 예측 페이지 버전 1 (100줄)
│   └── prediction2.py         # 예측 페이지 버전 2 (1098줄)
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

## 주요 파일 설명

### 애플리케이션 핵심

- **app.py**: 메인 Streamlit 애플리케이션으로, 페이지 라우팅과 사이드바 메뉴를 관리합니다.
- **config.py**: 경로, 모델, 시각화 설정을 중앙 관리하는 설정 파일입니다.

### 페이지 구성

- **pages/customer_analysis.py**: 고객 데이터에 대한 분석과 인사이트를 제공합니다.
- **pages/prediction.py**: 고객 이탈 예측 결과를 시각화하고 설명합니다.
- **pages/all_data.py**: 전체 고객 데이터를 테이블 형태로 표시하고 필터링 기능을 제공합니다.

### 모델 및 유틸리티

- **models/churn_model.py**: 이탈 예측 모델의 로드, 예측, 설명을 담당하는 `ChurnPredictor` 클래스가 구현되어 있습니다.
- **utils/visualizer.py**: 다양한 시각화 함수를 제공하는 `Visualizer` 클래스가 구현되어 있습니다.
- **utils/data_generator.py**: 테스트용 샘플 데이터를 생성하는 함수를 제공합니다.

### UI 컴포넌트

- **components/header.py**: 모든 페이지에서 사용되는 공통 헤더 컴포넌트입니다.
- **components/animations.py**: 페이지 전환 시 애니메이션 효과를 제공합니다.

## 데이터 흐름

1. `app.py`에서 사용자 인터랙션에 따라 적절한 페이지로 라우팅
2. 각 페이지(`pages/*.py`)에서 데이터 로드, 처리, 시각화 수행
3. 예측 페이지는 `models/churn_model.py`의 `ChurnPredictor` 클래스를 통해 예측 수행
4. 시각화는 `utils/visualizer.py`의 `Visualizer` 클래스를 통해 구현
5. 설정은 `config.py`에서 중앙 관리되어 다른 모듈에서 참조 
