import pandas as pd
import numpy as np
from config import MODEL_CONFIG

def generate_sample_data(n_samples=1000):
    """임시 데이터 생성"""
    np.random.seed(42)
    
    data = {
        'CustomerID': [f'C{i:04d}' for i in range(1, n_samples+1)],
        'Churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]).astype(int),
        'Tenure': np.random.randint(1, 60, n_samples).astype(int),
        'PreferredLoginDevice': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples),
        'CityTier': np.random.randint(1, 4, n_samples).astype(int),
        'WarehouseToHome': np.random.randint(5, 50, n_samples).astype(int),
        'PreferredPaymentMode': np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Cash on Delivery'], n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'HourSpendOnApp': np.random.uniform(0, 10, n_samples).round(1).astype(float),
        'NumberOfDeviceRegistered': np.random.randint(1, 5, n_samples).astype(int),
        'PreferedOrderCat': np.random.choice(['Electronics', 'Fashion', 'Grocery', 'Home'], n_samples),
        'SatisfactionScore': np.random.randint(1, 6, n_samples).astype(int),
        'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'NumberOfAddress': np.random.randint(1, 4, n_samples).astype(int),
        'Complain': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]).astype(int),
        'OrderAmountHikeFromlastYear': np.random.uniform(0, 30, n_samples).round(1).astype(float),
        'CouponUsed': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]).astype(int),
        'OrderCount': np.random.randint(1, 100, n_samples).astype(int),
        'DaySinceLastOrder': np.random.randint(1, 90, n_samples).astype(int),
        'CashbackAmount': np.random.uniform(0, 100, n_samples).round(2).astype(float)
    }
    
    df = pd.DataFrame(data)
    
    # 이탈 확률 생성 (실제 모델 대신 임시값)
    df['churn_probability'] = np.random.uniform(0, 1, n_samples).astype(float)
    
    return df 