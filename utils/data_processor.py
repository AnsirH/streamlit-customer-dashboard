import pandas as pd
import numpy as np

class DataProcessor:
    @staticmethod
    def load_data(file_path):
        """데이터 로드"""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"데이터 로딩 실패: {e}")
            return None
            
    @staticmethod
    def preprocess_data(df):
        """데이터 전처리"""
        # 필요한 전처리 로직 구현
        # 예: 결측치 처리, 인코딩, 스케일링 등
        return df
        
    @staticmethod
    def get_customer_data(df, customer_id):
        """특정 고객의 데이터 추출"""
        return df[df['customer_id'] == customer_id]
        
    @staticmethod
    def get_summary_statistics(df):
        """데이터 요약 통계"""
        return df.describe() 