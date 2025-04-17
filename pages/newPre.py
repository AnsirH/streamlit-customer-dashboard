import pandas as pd
import os
import sys
from pathlib import Path

# 현재 스크립트의 경로를 기준으로 프로젝트 루트 경로 설정
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
project_root = current_dir
sys.path.append(str(project_root))

# ChurnPredictor 클래스 임포트
try:
    from models.churn_model import ChurnPredictor
except ImportError:
    print("모델을 불러오는 데 실패했습니다. 경로를 확인해주세요.")
    sys.exit(1)

def predict_churn_for_new_customer():
    """
    새로운 고객 데이터에 대한 이탈 확률을 예측합니다.
    """
    # 1. 예측기 인스턴스 생성
    predictor = ChurnPredictor()
    
    print("고객 이탈 예측 모델이 로드되었습니다.")
    
    # 2. 새로운 고객 데이터 준비 (원본 형태 - 원핫인코딩 전)
    new_customer_data = pd.DataFrame({
        'CustomerID': ['CUST-12345'],  # 고객 ID
        'Tenure': [12],                # 거래기간 (개월)
        'PreferredLoginDevice': ['Mobile'],  # 선호 로그인 기기
        'CityTier': [1],               # 도시 등급
        'WarehouseToHome': [20],       # 창고-집 거리 (km)
        'PreferredPaymentMode': ['Credit Card'],  # 선호 결제 방식
        'Gender': ['Male'],            # 성별
        'HourSpendOnApp': [3.0],       # 앱 사용 시간 (시간)
        'NumberOfDeviceRegistered': [2],  # 등록된 기기 수
        'PreferedOrderCat': ['Electronics'],  # 선호 주문 카테고리
        'SatisfactionScore': [3],      # 만족도 점수 (1-5)
        'MaritalStatus': ['Single'],   # 결혼 상태
        'NumberOfAddress': [2],        # 주소 개수
        'Complain': ['아니오'],        # 불만 제기 여부
        'OrderAmountHikeFromlastYear': [15.0],  # 작년 대비 주문 금액 증가율 (%)
        'CouponUsed': [3],             # 쿠폰 사용 횟수
        'OrderCount': [10],            # 주문 횟수
        'DaySinceLastOrder': [15],     # 마지막 주문 후 경과일
        'CashbackAmount': [150.0]      # 캐시백 금액 (원)
    })
    
    print("새로운 고객 데이터가 준비되었습니다:")
    print(new_customer_data)
    
    # 3. 예측 수행 (내부적으로 원핫인코딩 등 전처리가 이루어짐)
    try:
        y_pred, y_proba = predictor.predict(new_customer_data)
        
        # 4. 결과 출력
        print("\n===== 예측 결과 =====")
        print(f"이탈 예측: {'이탈' if y_pred[0] == 1 else '유지'}")
        print(f"이탈 확률: {y_proba[0] * 100:.2f}%")
        
        # 선택적: 이탈 위험도 판정
        risk_level = "높음" if y_proba[0] >= 0.7 else ("중간" if y_proba[0] >= 0.3 else "낮음")
        print(f"이탈 위험도: {risk_level}")
        
        # 권장 조치 제안
        if risk_level == "높음":
            print("\n[권장 조치]")
            print("- 즉각적인 고객 응대 필요")
            print("- 특별 할인 또는 혜택 제공 고려")
            print("- 맞춤형 제안으로 고객 관계 강화")
        elif risk_level == "중간":
            print("\n[권장 조치]")
            print("- 고객 만족도 점검")
            print("- 추가 서비스나 혜택 제안")
            print("- 정기적인 소통 강화")
        else:
            print("\n[권장 조치]")
            print("- 현재 관리 방식 유지")
            print("- 정기적인 프로모션 안내")
    except Exception as e:
        print(f"예측 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    predict_churn_for_new_customer() 