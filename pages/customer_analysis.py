import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from components.header import show_header
from components.animations import add_page_transition

def show():
    # 애니메이션 적용
    add_page_transition()

    show_header()

    st.subheader("선택한 고객 ID: **a500**")

    data = {
        "고객 ID": ["a500"],
        "위험률(%)": [70],
        "칼럼1": [12],
        "칼럼2": [1111],
        "칼럼3": [45],
        "칼럼4": ["aa"],
        "칼럼5": ["xx"],
        "칼럼6": [2.14],
        "칼럼7": [1],
        "칼럼8": [0],
        "칼럼9": ["ff"],
        "칼럼10": [1],
        "칼럼11": [1],
        "칼럼12": [1]
    }
    df = pd.DataFrame(data)
    st.dataframe(df.T, use_container_width=True)

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = 70,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "이탈확률"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "orange"},
            'shape': "angular"
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 예측 근거")
    st.markdown("""
    <div style="border: 2px dashed gray; padding: 10px;">
    <b style="background-color: yellow;">마지막 주문 기간 공백</b><br><br>
    위 칼럼에 대해 저장해둔 대안(글)
    </div>
    """, unsafe_allow_html=True) 