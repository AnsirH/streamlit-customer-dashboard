import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import LabelEncoder

def analyze_customers():
    """
    고객 데이터를 분석하고 이탈 예측 결과를 반환합니다.
    Returns:
        DataFrame: 고객 ID, 이탈 위험도, 상위 3개 영향 요인을 포함한 결과
    """
    try:
        # 모델과 데이터셋 경로 설정
        model_path = os.path.join('models', 'xgb_best_model.pkl')
        data_path = os.path.join('models', 'E Commerce Dataset2.xlsx')
        
        # 파일 존재 여부 확인
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터셋 파일을 찾을 수 없습니다: {data_path}")
        
        # 모델과 데이터셋 로드
        model = joblib.load(model_path)
        df = pd.read_excel(data_path)
        
        # CustomerID 컬럼 확인
        if 'CustomerID' not in df.columns:
            raise ValueError("CustomerID 컬럼이 데이터셋에 존재하지 않습니다.")
        
        # 원본 컬럼명으로 변경
        column_mapping = {
            '거래기간': 'Tenure',
            '도시_등급': 'CityTier',
            '배송_거리': 'WarehouseToHome',
            '앱_사용_시간': 'HourSpendOnApp',
            '등록_기기_수': 'NumberOfDeviceRegistered',
            '만족도_점수': 'SatisfactionScore',
            '주소_수': 'NumberOfAddress',
            '불만_제기_여부': 'Complain',
            '주문_금액_상승률': 'OrderAmountHikeFromlastYear',
            '쿠폰_사용_여부': 'CouponUsed',
            '주문_횟수': 'OrderCount',
            '마지막_주문_후_일수': 'DaySinceLastOrder',
            '캐쉬백_금액': 'CashbackAmount',
            '선호_로그인_기기': 'PreferredLoginDevice',
            '선호_결제_방식': 'PreferredPaymentMode',
            '성별': 'Gender',
            '선호_주문_카테고리': 'PreferedOrderCat',
            '결혼_상태': 'MaritalStatus'
        }
        
        # 컬럼명 변경
        df = df.rename(columns=column_mapping)
        
        # 범주형 변수 원-핫 인코딩
        categorical_columns = ['PreferredLoginDevice', 'PreferredPaymentMode', 
                             'Gender', 'PreferedOrderCat', 'MaritalStatus']
        
        for col in categorical_columns:
            if col in df.columns:
                # 원-핫 인코딩 수행
                dummies = pd.get_dummies(df[col], prefix=col)
                # 기존 컬럼 제거
                df = df.drop(col, axis=1)
                # 원-핫 인코딩된 컬럼 추가
                df = pd.concat([df, dummies], axis=1)
        
        # 예측에 사용할 특성 선택
        feature_columns = [col for col in df.columns if col not in ['CustomerID', 'Churn']]
        X = df[feature_columns]
        
        # 이탈 확률 예측
        churn_probabilities = model.predict_proba(X)[:, 1]
        
        # 특성 중요도 계산
        feature_importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)
        
        # 결과 데이터프레임 생성
        results = []
        for idx, row in df.iterrows():
            # 상위 3개 특성 선택
            top_features = feature_importance_df.head(3)
            
            result = {
                'CustomerID': row['CustomerID'],
                'Churn Risk': churn_probabilities[idx],
                'Top Feature 1': top_features.iloc[0]['Feature'],
                'Importance 1': top_features.iloc[0]['Importance'],
                'Top Feature 2': top_features.iloc[1]['Feature'],
                'Importance 2': top_features.iloc[1]['Importance'],
                'Top Feature 3': top_features.iloc[2]['Feature'],
                'Importance 3': top_features.iloc[2]['Importance']
            }
            results.append(result)
        
        # 결과 DataFrame 생성
        result_df = pd.DataFrame(results)
        
        # 결과 검증
        if len(result_df) == 0:
            raise ValueError("분석 결과가 생성되지 않았습니다.")
        if result_df['Churn Risk'].isna().any():
            raise ValueError("이탈 확률에 결측값이 있습니다.")
        
        return result_df
        
    except FileNotFoundError as e:
        raise Exception(f"파일을 찾을 수 없습니다: {str(e)}")
    except ValueError as e:
        raise Exception(f"데이터 처리 중 오류가 발생했습니다: {str(e)}")
    except Exception as e:
        raise Exception(f"고객 분석 중 오류가 발생했습니다: {str(e)}") 