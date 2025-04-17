import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from components.header import show_header
from components.animations import add_page_transition
from models.churn_model import ChurnPredictor

# --------------------
# 🔁 캐시된 모델 로딩
# --------------------
@st.cache_resource
def get_predictor():
    return ChurnPredictor()

# --------------------
# 🎯 게이지 차트 생성
# --------------------
def create_churn_gauge(probability: float) -> go.Figure:
    risk_percent = probability * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_percent,
        title={"text": "이탈 가능성 (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': 'green'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'red'}
            ],
            'bar': {'color': 'darkblue'},
            'threshold': {'line': {'color': 'red', 'width': 4}, 'value': risk_percent}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# --------------------------
# 📥 입력 필드 렌더링 함수
# --------------------------
def render_input_fields(form) -> dict:
    fields = {
        'CustomerID': form.text_input('고객 ID *', value=f"CUST-{np.random.randint(10000,99999)}"),
        'Tenure': form.number_input('거래기간 (개월) *', min_value=0, value=12),
        'SatisfactionScore': form.number_input('만족도 점수 (1-5) *', min_value=1, max_value=5, value=3),
        'OrderCount': form.number_input('주문 횟수 *', min_value=0, value=10),
        'HourSpendOnApp': form.number_input('앱 사용 시간 (시간) *', min_value=0.0, value=3.0),
        'DaySinceLastOrder': form.number_input('마지막 주문 후 경과일 *', min_value=0, value=15),
        'Gender': form.selectbox('성별', ['Male','Female']),
        'Complain': form.selectbox('불만 제기 여부', ['아니오','예'])
    }
    return fields

# ------------------------
# 📊 결과 출력 렌더링 함수
# ------------------------
def render_results(input_data: dict, prob_value: float, predictor: ChurnPredictor):
    st.subheader('예측 결과')
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(create_churn_gauge(prob_value), use_container_width=True)

    with col2:
        if prob_value < 0.3:
            risk_text, action = '낮음', '정기적인 마케팅 이메일을 보내세요.'
        elif prob_value < 0.7:
            risk_text, action = '중간', '개인화된 제안을 고려하세요.'
        else:
            risk_text, action = '높음', '즉각적인 고객 혜택이 필요합니다.'

        st.markdown(f"""
        - **이탈 확률**: {prob_value:.2%}  
        - **위험도**: **{risk_text}**

        **권장 조치**: {action}
        """)

    st.subheader('주요 영향 요인')
    fi = predictor.get_feature_importance()
    if fi:
        top_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:3]
        df_fi = pd.DataFrame(top_fi, columns=['요인','영향도'])
        st.bar_chart(df_fi.set_index('요인'))
    else:
        st.warning('특성 중요도를 불러올 수 없습니다.')

# --------------------------
# 🚀 Streamlit 메인 페이지
# --------------------------
def main():
    add_page_transition()
    show_header()
    st.title('고객 이탈 예측')
    st.write('고객 정보를 입력하고 제출 버튼을 눌러 예측하세요.')

    predictor = get_predictor()
    with st.form('predict_form') as form:
        input_data = render_input_fields(form)
        submitted = form.form_submit_button('예측하기')

    if submitted:
        try:
            input_df = pd.DataFrame([input_data])
            _, y_proba = predictor.predict(input_df)
            render_results(input_data, y_proba[0], predictor)
        except Exception as e:
            st.error(f'예측 중 오류 발생: {e}')

# --------------------------
# 앱 실행
# --------------------------
if __name__ == '__main__':
    main()