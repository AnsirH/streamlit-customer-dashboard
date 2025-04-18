# Streamlit Customer Dashboard

## 요구사항

- Python 3.9 이상, 3.12 미만
- pip 24.0 이상

## 설치 방법

1. 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 패키지 설치:
```bash
pip install -r requirements.txt
```

3. 앱 실행:
```bash
streamlit run app.py
```

## 주요 기능

- 고객 이탈 예측
- 데이터 시각화
- 통계 분석



## 🎯 프로젝트 개요
```
- **목적**: 고객의 이탈 가능성을 사전에 예측하여 유지 전략 수립에 활용
- **데이터 출처**: Kaggle 공개 데이터셋
- **샘플 수**: 5,630명
- **피처 수**: 20개 (ID 포함)
- **타깃**: `Churn` (0: 유지, 1: 이탈)

```

## 🧹 데이터 전처리
```
- 결측값 제거 및 평균 대체
- 수치형 변수 IQR 기반 이상치 정리
- One-hot 인코딩, StandardScaler 적용
- SMOTE 오버샘플링으로 클래스 불균형 처리

```

## 🧠 모델링
```
- 사용 모델: XGBoostClassifier (최종 선택)
- 비교 모델: LogisticRegression, KNN, SVC, NaiveBayes
- 교차검증: 5-Fold
- 하이퍼파라미터 튜닝: GridSearchCV

```
