import streamlit as st
from components.header import show_header

# 페이지 설정
st.set_page_config(
    page_title="이탈 예측 대시보드",
    page_icon="🚀",
    layout="wide"
)

# 사이드바 설정
st.sidebar.title("메뉴")
st.sidebar.markdown("---")
st.sidebar.markdown("⬇️ **스크롤**")

# 메인 페이지 내용
show_header()
st.write("좌측 사이드바에서 원하는 페이지를 선택하세요.") 