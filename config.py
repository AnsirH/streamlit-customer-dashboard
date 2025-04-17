import os
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# 디렉토리 생성
for dir_path in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True)

# 파일 경로 설정
PATHS = {
    'raw_data': DATA_DIR / "raw" / "customer_data.csv",
    'processed_data': DATA_DIR / "processed" / "processed_data.csv",
    'model': MODEL_DIR / "churn_model.joblib",
    'log_file': LOG_DIR / "app.log"
}

# 모델 설정
MODEL_CONFIG = {
    'threshold': 0.5,  # 이탈 판단 임계값
    'feature_names': [
        'recency', 'frequency', 'monetary',
        'avg_order_value', 'days_since_last_order',
        'total_orders', 'total_spent'
    ]
}

# 시각화 설정
VIZ_CONFIG = {
    'colors': {
        'low_risk': '#4CAF50',  # 초록색
        'medium_risk': '#FFC107',  # 노란색
        'high_risk': '#F44336'  # 빨간색
    },
    'thresholds': {
        'low': 0.3,
        'medium': 0.7
    }
} 