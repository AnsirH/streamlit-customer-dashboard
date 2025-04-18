import pandas as pd
import numpy as np

# Excel 파일 읽기
df = pd.read_excel('models/E Commerce Dataset2.xlsx')

def analyze_feature(df, feature, threshold, direction='lower'):
    """특성 분석 및 기준값 제안"""
    print(f"\n=== {feature} 분석 ===")
    print(df[feature].describe())
    
    if direction == 'lower':
        target = df[df[feature] <= threshold]
    else:
        target = df[df[feature] >= threshold]
    
    print(f"\n{threshold} 기준 {'이하' if direction == 'lower' else '이상'} 고객 수: {len(target)}명")
    print(f"전체 고객 대비 비율: {len(target)/len(df)*100:.2f}%")
    print(f"이탈률: {target['Churn'].mean()*100:.2f}%")
    
    # 기준값 제안
    if direction == 'lower':
        q25 = df[feature].quantile(0.25)
        print(f"\n제안 기준값: {q25:.2f} (25% 백분위수)")
    else:
        q75 = df[feature].quantile(0.75)
        print(f"\n제안 기준값: {q75:.2f} (75% 백분위수)")

# 각 특성별 분석
analyze_feature(df, 'SatisfactionScore', 3, 'lower')  # 만족도가 낮을수록 이탈 위험
analyze_feature(df, 'OrderCount', 2, 'lower')  # 주문 횟수가 적을수록 이탈 위험
analyze_feature(df, 'CashbackAmount', 100, 'lower')  # 캐시백 사용이 적을수록 이탈 위험
analyze_feature(df, 'HourSpendOnApp', 1, 'lower')  # 앱 사용 시간이 적을수록 이탈 위험
analyze_feature(df, 'Tenure', 3, 'lower')  # 거래 기간이 짧을수록 이탈 위험
analyze_feature(df, 'OrderAmountHikeFromlastYear', 10, 'lower')  # 주문 금액 증가율이 낮을수록 이탈 위험
analyze_feature(df, 'CouponUsed', 1, 'lower')  # 쿠폰 사용이 적을수록 이탈 위험
analyze_feature(df, 'Complain', 1, 'higher')  # 불만 제기 이력이 있으면 이탈 위험

# DaySinceLastOrder 컬럼 분석
print("\nDaySinceLastOrder 통계:")
print(df['DaySinceLastOrder'].describe())

# 30일 이상인 데이터 개수 확인
over_30 = df[df['DaySinceLastOrder'] >= 30]
print(f"\n30일 이상 고객 수: {len(over_30)}명")
print(f"전체 고객 대비 비율: {len(over_30)/len(df)*100:.2f}%")

# 30일 이상인 고객들의 이탈률 확인
print(f"\n30일 이상 고객의 이탈률: {over_30['Churn'].mean()*100:.2f}%") 