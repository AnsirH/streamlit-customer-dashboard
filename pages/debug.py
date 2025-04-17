import streamlit as st
import pandas as pd
import numpy as np
from models.churn_model import ChurnPredictor
from pathlib import Path

def show():
    st.title("ChurnPredictor 디버그 페이지")
    st.write("이 페이지는 ChurnPredictor 클래스의 작동을 확인하기 위한 디버그 페이지입니다.")
    
    # 모델 경로 확인
    predictor = ChurnPredictor()
    model_path = predictor.model_path
    
    st.subheader("1. 모델 파일 정보")
    st.write(f"모델 경로: {model_path}")
    st.write(f"모델 파일 존재 여부: {model_path.exists()}")
    
    if model_path.exists():
        st.write(f"모델 파일 크기: {model_path.stat().st_size / 1024:.2f} KB")
    
    # 모델 로드 테스트
    st.subheader("2. 모델 로드 테스트")
    load_success = predictor.load_model()
    
    if load_success:
        st.success("✅ 모델 로드 성공!")
        st.write(f"모델 타입: {type(predictor.model)}")
        
        if hasattr(predictor.model, 'feature_importances_'):
            st.write(f"특성 중요도 개수: {len(predictor.model.feature_importances_)}")
            
        # 모델 속성 확인
        if hasattr(predictor.model, 'feature_names_in_'):
            st.write("모델이 사용한 특성 이름:")
            st.write(predictor.model.feature_names_in_)
    else:
        st.error("❌ 모델 로드 실패!")
    
    # 샘플 데이터 생성
    st.subheader("3. 샘플 데이터로 예측 테스트")
    
    # 간단한 샘플 데이터 생성
    sample_data = {
        'CustomerID': 'CUST-12345',
        'Tenure': 12,
        'PreferredLoginDevice': 'Mobile',
        'CityTier': 1,
        'WarehouseToHome': 20,
        'PreferredPaymentMode': 'Credit Card',
        'Gender': 'Male',
        'HourSpendOnApp': 3.0,
        'NumberOfDeviceRegistered': 2,
        'PreferedOrderCat': 'Electronics',
        'SatisfactionScore': 3,
        'MaritalStatus': 'Single',
        'NumberOfAddress': 2,
        'Complain': '아니오',
        'OrderAmountHikeFromlastYear': 15.0,
        'CouponUsed': 3,
        'OrderCount': 10,
        'DaySinceLastOrder': 15,
        'CashbackAmount': 150.0
    }
    
    df = pd.DataFrame([sample_data])
    st.write("샘플 입력 데이터:")
    st.dataframe(df)
    
    # 데이터 전처리 테스트
    st.subheader("4. 데이터 전처리 테스트")
    try:
        processed_df = predictor._preprocess_data(df)
        st.success("✅ 데이터 전처리 성공!")
        st.write("전처리 후 컬럼:")
        st.write(processed_df.columns.tolist())
        st.write("전처리 후 데이터:")
        st.dataframe(processed_df)
    except Exception as e:
        st.error(f"❌ 데이터 전처리 실패: {str(e)}")
    
    # 예측 테스트
    st.subheader("5. 예측 테스트")
    try:
        y_pred, y_proba = predictor.predict(df)
        st.success("✅ 예측 성공!")
        st.write(f"예측 클래스: {y_pred[0]}")
        st.write(f"이탈 확률: {y_proba[0]:.4f} ({y_proba[0]*100:.2f}%)")
        
        # 확률에 따른 위험도 표시
        if y_proba[0] < 0.3:
            st.write("위험도: 🟢 낮음")
        elif y_proba[0] < 0.7:
            st.write("위험도: 🟡 중간")
        else:
            st.write("위험도: 🔴 높음")
    except Exception as e:
        st.error(f"❌ 예측 실패: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    # 특성 중요도 테스트
    st.subheader("6. 특성 중요도 테스트")
    try:
        feature_importance = predictor.get_feature_importance()
        st.success("✅ 특성 중요도 계산 성공!")
        
        # 특성 중요도를 데이터프레임으로 변환
        fi_df = pd.DataFrame({
            "특성": feature_importance.keys(),
            "중요도": feature_importance.values()
        })
        fi_df = fi_df.sort_values(by="중요도", ascending=False)
        
        st.write("특성 중요도:")
        st.dataframe(fi_df)
        
        # 막대 그래프로 시각화
        st.bar_chart(fi_df.set_index("특성"))
    except Exception as e:
        st.error(f"❌ 특성 중요도 계산 실패: {str(e)}")

# 독립 실행 시 호출
if __name__ == "__main__":
    show() 