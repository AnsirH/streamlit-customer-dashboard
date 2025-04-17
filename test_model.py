import sys
print("Python version:", sys.version)

try:
    import pandas as pd
    import numpy as np
    import joblib
    from pathlib import Path
    print("기본 라이브러리 import 성공")
    
    try:
        # 모듈 경로 설정
        import models.churn_model
        from models.churn_model import load_churn_model
        print("churn_model 모듈 import 성공")
        
        # 모델 파일 확인
        model_path = Path("models/xgb_best_model.pkl")
        if model_path.exists():
            print(f"모델 파일 존재: {model_path} (크기: {model_path.stat().st_size} 바이트)")
            
            # 모델 직접 로드 시도
            try:
                # 직접 joblib으로 로드
                direct_model = joblib.load(model_path)
                print(f"직접 joblib으로 모델 로드 성공: {type(direct_model)}")
            except Exception as e:
                print(f"직접 모델 로드 실패: {e}")
                
            # 함수를 통한 모델 로드 시도
            try:
                # load_churn_model 함수로 로드
                model = load_churn_model(model_path)
                print(f"load_churn_model 함수로 모델 로드 성공: {type(model)}")
            except Exception as e:
                print(f"load_churn_model 함수로 모델 로드 실패: {e}")
        else:
            print(f"모델 파일이 존재하지 않음: {model_path}")
            
    except Exception as e:
        print(f"churn_model 모듈 import 실패: {e}")
        
except Exception as e:
    print(f"기본 라이브러리 import 실패: {e}") 