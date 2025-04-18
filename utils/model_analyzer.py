import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import os
from utils.visualizer import Visualizer
from components.header import show_header
from components.animations import add_page_transition

class ModelPredictor:
    @staticmethod
    def load_data():
        """데이터셋을 로드하는 함수"""
        try:
            data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'E Commerce Dataset2.xlsx')
            df = pd.read_excel(data_path)
            return df
        except Exception as e:
            raise Exception(f"데이터셋 로드 중 오류 발생: {e}")

    @staticmethod
    def load_model():
        """모델을 로드하는 함수"""
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'xgb_best_model.pkl')
            return load(model_path)
        except Exception as e:
            raise Exception(f"모델 로드 중 오류 발생: {e}")

    @staticmethod
    def preprocess_data(df):
        """데이터 전처리 함수"""
        try:
            # 필요한 특성만 선택
            features = [
                'DaySinceLastOrder', 'SatisfactionScore', 'HourSpendOnApp',
                'OrderAmountHikeFromlastYear', 'CashbackAmount', 'Tenure',
                'OrderCount', 'CityTier', 'Complain', 'PreferredLoginDevice',
                'PreferredPaymentMode', 'Gender', 'PreferedOrderCat',
                'MaritalStatus', 'WarehouseToHome', 'NumberOfAddress',
                'NumberOfDeviceRegistered', 'CouponUsed'
            ]
            
            # 데이터 복사
            X = df[features].copy()
            
            # 결측치 처리
            X = X.fillna(0)
            
            # 문자열 특성들을 숫자형으로 변환
            device_mapping = {
                'Mobile': 0, 'Computer': 1, 'Phone': 2, 'Tablet': 3,
                'Desktop': 4, 'Laptop': 5, 'Smartphone': 6
            }
            payment_mapping = {
                'Debit Card': 0, 'UPI': 1, 'CC': 2, 'COD': 3, 'E wallet': 4,
                'Credit Card': 5, 'Cash on Delivery': 6
            }
            gender_mapping = {
                'M': 0, 'F': 1, 'Male': 0, 'Female': 1
            }
            category_mapping = {
                'Laptop & Accessory': 0, 'Mobile': 1, 'Mobile Phone': 2, 
                'Fashion': 3, 'Grocery': 4, 'Tablet': 5,
                'Electronics': 6, 'Clothing': 7, 'Food': 8
            }
            marital_mapping = {
                'Single': 0, 'Married': 1, 'Divorced': 2,
                'Unmarried': 0, 'Separated': 2
            }
            
            # 각 열에 대해 안전하게 변환
            X['PreferredLoginDevice'] = X['PreferredLoginDevice'].replace(device_mapping).fillna(0).astype('int64')
            X['PreferredPaymentMode'] = X['PreferredPaymentMode'].replace(payment_mapping).fillna(0).astype('int64')
            X['Gender'] = X['Gender'].replace(gender_mapping).fillna(0).astype('int64')
            X['PreferedOrderCat'] = X['PreferedOrderCat'].replace(category_mapping).fillna(0).astype('int64')
            X['MaritalStatus'] = X['MaritalStatus'].replace(marital_mapping).fillna(0).astype('int64')
            
            # 불리언 특성을 숫자형으로 변환
            X['Complain'] = X['Complain'].astype(int)
            
            return X, df
            
        except Exception as e:
            raise Exception(f"데이터 전처리 중 오류 발생: {e}")

    @staticmethod
    def predict_churn():
        """고객 이탈 확률을 예측하는 함수"""
        try:
            # 데이터와 모델 로드
            df = ModelPredictor.load_data()
            model = ModelPredictor.load_model()
            
            # 데이터 전처리
            X, df = ModelPredictor.preprocess_data(df)
            
            # 예측 확률 계산
            probabilities = model.predict_proba(X)[:, 1]
            df['churn_prob'] = probabilities
            
            # 상위 3개 영향 요인과 중요도 계산
            feature_importances = model.feature_importances_
            top_features = np.argsort(feature_importances)[-3:][::-1]
            
            for i in range(len(df)):
                df.loc[i, 'top_feature_1'] = X.columns[top_features[0]]
                df.loc[i, 'importance_1'] = feature_importances[top_features[0]]
                df.loc[i, 'top_feature_2'] = X.columns[top_features[1]]
                df.loc[i, 'importance_2'] = feature_importances[top_features[1]]
                df.loc[i, 'top_feature_3'] = X.columns[top_features[2]]
                df.loc[i, 'importance_3'] = feature_importances[top_features[2]]
            
            # 상위 3개 고객 표시
            top_customers = df.nlargest(3, 'churn_prob')
            for _, customer_data in top_customers.iterrows():
                st.markdown(f"### 고객 ID: {customer_data['CustomerID']}")
                Visualizer.create_churn_gauge(customer_data['churn_prob']),
            
            return df
            
        except Exception as e:
            raise Exception(f"예측 중 오류 발생: {e}")

    @staticmethod
    def display_customer_analysis(df):
        """고객 분석 결과를 표시하는 함수"""
        # 테이블 표시
        Visualizer().display_prediction_table(df)
        
        # 고객 선택
        customer_id = st.selectbox(
            "고객 ID 선택",
            df['CustomerID'].tolist()
        )
        
        # 선택된 고객 데이터
        customer_data = df[df['CustomerID'] == customer_id].iloc[0]
        
        # 메인 컨테이너
        main_container = st.container()
        
        # 레이아웃 설정 - 4:6 비율
        with main_container:
            left_col, right_col = st.columns([4, 6])
            
            # 오른쪽 열에 이탈 확률 게이지 먼저 표시
            with right_col:
                st.markdown("#### 이탈 확률")
                st.markdown('70% 이상의 이탈 확률을 가진 고객은 이탈 위험이 높습니다.')
                st.plotly_chart(
                    Visualizer.create_churn_gauge(customer_data['churn_prob']),
                    use_container_width=True
                )

                # 주요 이탈 요인과 개선 방안을 카드 형태로 표시
                st.markdown("##### 주요 이탈 요인")
                st.markdown("""
                <div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin-top: 10px; margin-bottom: 20px;'>
                    <p style='color: white; font-size: 15px; margin: 0;'>
                        {feature_1}: {importance_1:.1%}<br>
                        {feature_2}: {importance_2:.1%}<br>
                        {feature_3}: {importance_3:.1%}
                    </p>
                </div>
                """.format(
                    feature_1=customer_data['top_feature_1'],
                    importance_1=customer_data['importance_1'],
                    feature_2=customer_data['top_feature_2'],
                    importance_2=customer_data['importance_2'],
                    feature_3=customer_data['top_feature_3'],
                    importance_3=customer_data['importance_3']
                ), unsafe_allow_html=True)

                st.markdown("##### 개선 방안")
                st.markdown("""
                <div style='background-color: rgba(255, 255, 255, 0.1); padding: 15px; border-radius: 10px; margin-top: 10px;'>
                    <ul style='color: white; font-size: 13px; margin: 0; padding-left: 20px;'>
                        <li>개인화된 할인 쿠폰 발송</li>
                        <li>관심 상품 재입고 알림 서비스</li>
                        <li>최근 트렌드 상품 추천</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # 왼쪽 열에 고객 정보 표시
            with left_col:
                # 고객 번호 표시
                st.markdown(f"### 고객번호: {customer_id}")
                
                st.markdown("##### 고객 기본 정보")
                info_data = {
                    '거래기간': f"{customer_data['Tenure']}개월",
                    '선호 로그인 기기': customer_data['PreferredLoginDevice'],
                    '도시 등급': f"Tier {customer_data['CityTier']}",
                    '성별': customer_data['Gender']
                }
                st.write(pd.Series(info_data))

                # 주문정보와 만족도정보를 순차적으로 표시
                st.markdown("##### 주문 정보")
                order_data = {
                    '주문 횟수': customer_data['OrderCount'],
                    '마지막 주문': f"{customer_data['DaySinceLastOrder']}일 전",
                    '주문 증가율': f"{customer_data['OrderAmountHikeFromlastYear']}%",
                    '캐쉬백': f"${customer_data['CashbackAmount']:.2f}"
                }
                st.write(pd.Series(order_data))
                
                st.markdown("##### 만족도 정보")
                satisfaction_data = {
                    '만족도': f"{customer_data['SatisfactionScore']}/5",
                    '불만 제기': '있음' if customer_data['Complain'] else '없음',
                    '앱 사용': f"{customer_data['HourSpendOnApp']}시간"
                }
                st.write(pd.Series(satisfaction_data))
        
        # 페이지 구분선
        st.markdown("---")
        
        # 상관계수 분석
        st.subheader("각 칼럼 별 이탈 여부와의 상관관계")
        st.plotly_chart(
            Visualizer.create_correlation_bar(),
            use_container_width=True
        )

    @staticmethod
    def show():
        """고객 분석 페이지를 표시하는 함수"""
        add_page_transition()
        show_header()
        
        try:
            # predict_churn을 인자 없이 호출
            df = ModelPredictor.predict_churn()
            ModelPredictor.display_customer_analysis(df)
        except Exception as e:
            st.error(str(e)) 