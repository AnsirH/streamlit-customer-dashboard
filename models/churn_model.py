import pandas as pd
import numpy as np
from pathlib import Path
from utils.cache import load_model
from utils.logger import setup_logger
from config import PATHS, MODEL_CONFIG

logger = setup_logger(__name__)

class ChurnPredictor:
    def __init__(self):
        self.model = self._load_model()
        self.feature_names = MODEL_CONFIG['feature_names']
        self.threshold = MODEL_CONFIG['threshold']
        
    def _load_model(self):
        """모델 로드 (캐싱 적용)"""
        try:
            return load_model(PATHS['model'])
        except Exception as e:
            logger.error(f"모델 로딩 실패: {e}")
            return None
            
    def predict(self, data):
        """이탈 확률 예측"""
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다.")
            return None
            
        try:
            # 데이터 전처리
            processed_data = self._preprocess(data)
            # 예측
            probabilities = self.model.predict_proba(processed_data)
            return probabilities[:, 1]  # 이탈 확률 반환
        except Exception as e:
            logger.error(f"예측 실패: {e}")
            return None
            
    def get_feature_importance(self):
        """특성 중요도 반환"""
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(self.feature_names, self.model.feature_importances_))
        return None
        
    def _preprocess(self, data):
        """데이터 전처리"""
        try:
            # 필요한 전처리 로직 구현
            # 예: 결측치 처리, 스케일링 등
            processed_data = data.copy()
            
            # 특성 선택
            if isinstance(data, pd.DataFrame):
                processed_data = processed_data[self.feature_names]
            
            return processed_data
        except Exception as e:
            logger.error(f"데이터 전처리 실패: {e}")
            return None
            
    def get_risk_level(self, probability):
        """이탈 위험도 판단"""
        if probability < MODEL_CONFIG['threshold']:
            return "low"
        elif probability < 0.7:
            return "medium"
        else:
            return "high"
            
    def get_explanation(self, probability, feature_importance):
        """예측 결과 설명"""
        risk_level = self.get_risk_level(probability)
        top_features = dict(sorted(feature_importance.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:3])
        
        return {
            'risk_level': risk_level,
            'probability': probability,
            'top_features': top_features
        } 