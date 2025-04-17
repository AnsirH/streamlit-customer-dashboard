import pandas as pd
import numpy as np
from config import MODEL_CONFIG

def generate_sample_data(n_samples=1000):
    """임시 데이터 생성"""
    np.random.seed(42)
    
    # 숫자형 데이터 생성
    numeric_data = {
        'customer_id': [f'C{i:04d}' for i in range(1, n_samples+1)],
        'churn_risk': np.random.uniform(0, 1, n_samples).astype(float),
        'Tenure': np.random.randint(1, 60, n_samples).astype(int),
        'CityTier': np.random.randint(1, 4, n_samples).astype(int),
        'WarehouseToHome': np.random.randint(5, 50, n_samples).astype(int),
        'HourSpendOnApp': np.random.uniform(0, 10, n_samples).round(1).astype(float),
        'NumberOfDeviceRegistered': np.random.randint(1, 5, n_samples).astype(int),
        'SatisfactionScore': np.random.randint(1, 6, n_samples).astype(int),
        'NumberOfAddress': np.random.randint(1, 4, n_samples).astype(int),
        'Complain': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]).astype(int),
        'OrderAmountHikeFromlastYear': np.random.uniform(0, 30, n_samples).round(1).astype(float),
        'CouponUsed': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]).astype(int),
        'OrderCount': np.random.randint(1, 100, n_samples).astype(int),
        'DaySinceLastOrder': np.random.randint(1, 90, n_samples).astype(int),
        'CashbackAmount': np.random.uniform(0, 100, n_samples).round(2).astype(float)
    }
    
    # 범주형 데이터 생성
    categorical_data = {
        'PreferredLoginDevice': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples),
        'PreferredPaymentMode': np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Cash on Delivery'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'PreferedOrderCat': np.random.choice(['Electronics', 'Fashion', 'Grocery', 'Home'], n_samples),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples)
    }
    
    # 데이터프레임 생성
    df = pd.DataFrame({**numeric_data, **categorical_data})
    
    # 상위 3개 영향 요인과 중요도 추가
    features = ['Tenure', 'SatisfactionScore', 'HourSpendOnApp', 'OrderCount', 'DaySinceLastOrder']
    for i in range(1, 4):
        df[f'top_feature_{i}'] = np.random.choice(features, n_samples)
        df[f'importance_{i}'] = np.random.uniform(0.1, 0.3, n_samples)
    
    # 열 타입 명시적 지정
    numeric_columns = list(numeric_data.keys()) + [f'importance_{i}' for i in range(1, 4)]
    categorical_columns = list(categorical_data.keys()) + [f'top_feature_{i}' for i in range(1, 4)]
    
    for col in numeric_columns:
        if 'int' in str(df[col].dtype):
            df[col] = df[col].astype('Int64')
        elif 'float' in str(df[col].dtype):
            df[col] = df[col].astype('float64')
    
    for col in categorical_columns:
        df[col] = df[col].astype('string')
    
    return df 