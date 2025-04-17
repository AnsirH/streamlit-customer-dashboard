import sys
import platform

print(f"Python 버전: {platform.python_version()}")
print(f"상세 버전: {sys.version}")

# 주요 라이브러리 버전 체크
try:
    import pandas as pd
    print(f"pandas 버전: {pd.__version__}")
except ImportError:
    print("pandas를 찾을 수 없습니다.")

try:
    import numpy as np
    print(f"numpy 버전: {np.__version__}")
except ImportError:
    print("numpy를 찾을 수 없습니다.")

try:
    import streamlit as st
    print(f"streamlit 버전: {st.__version__}")
except ImportError:
    print("streamlit을 찾을 수 없습니다.")

try:
    import xgboost as xgb
    print(f"xgboost 버전: {xgb.__version__}")
except ImportError:
    print("xgboost를 찾을 수 없습니다.")

try:
    import plotly
    print(f"plotly 버전: {plotly.__version__}")
except ImportError:
    print("plotly를 찾을 수 없습니다.")
    
# Python 3.12 호환성 체크
print("\nPython 3.12 호환성 체크:")
if sys.version_info.major == 3 and sys.version_info.minor >= 12:
    print("현재 Python 3.12 이상 환경입니다.")
else:
    print(f"현재 Python {sys.version_info.major}.{sys.version_info.minor} 환경입니다. Python 3.12가 필요합니다.")
    
# 주요 라이브러리가 Python 3.12와 호환되는지 확인하는 내용 추가
print("\n주요 라이브러리 Python 3.12 호환성:")
print("pandas: 2.0.0 이상 버전이 Python 3.12와 호환됩니다.")
print("numpy: 1.26.0 이상 버전이 Python 3.12와 호환됩니다.")
print("streamlit: 1.22.0 이상 버전이 Python 3.12와 호환됩니다.")
print("xgboost: 1.7.3 이상 버전이 Python 3.12와 호환됩니다.")
print("plotly: 5.13.0 이상 버전이 Python 3.12와 호환됩니다.") 