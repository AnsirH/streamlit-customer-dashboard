import streamlit as st
from components.header import show_header
from components.animations import add_page_transition

# 애니메이션 적용
add_page_transition()

show_header()

st.title("예측")
st.write("예측 관련 내용이 여기에 표시됩니다.") 