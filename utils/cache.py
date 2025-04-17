import streamlit as st
import pandas as pd
import joblib
from functools import wraps
import time

def cache_data(ttl=3600):
    """데이터 캐싱 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # 캐시된 데이터가 있고 TTL이 만료되지 않았다면 반환
            if cache_key in st.session_state:
                cached_data, timestamp = st.session_state[cache_key]
                if time.time() - timestamp < ttl:
                    return cached_data
            
            # 캐시된 데이터가 없거나 만료되었다면 새로 계산
            result = func(*args, **kwargs)
            st.session_state[cache_key] = (result, time.time())
            return result
        return wrapper
    return decorator

@cache_data(ttl=3600)
def load_model(model_path):
    """모델 로드 (캐싱 적용)"""
    return joblib.load(model_path)

@cache_data(ttl=3600)
def load_data(file_path):
    """데이터 로드 (캐싱 적용)"""
    return pd.read_csv(file_path) 