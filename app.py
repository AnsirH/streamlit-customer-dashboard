import streamlit as st
from components.header import show_header
from components.animations import add_page_transition

# 페이지 설정
st.set_page_config(
    page_title="이탈 예측 대시보드",
    page_icon="🚀",
    layout="wide"
)

# 애니메이션 적용
add_page_transition()

# 사이드바 설정
st.sidebar.title("메뉴")

# 페이지 이동 버튼
st.sidebar.page_link("pages/1_고객분석.py", label="고객분석", icon="📊")
st.sidebar.page_link("pages/2_예측.py", label="예측", icon="🔮")
st.sidebar.page_link("pages/3_전체_데이터.py", label="전체 데이터", icon="📈")

st.sidebar.markdown("---")
st.sidebar.markdown("⬇️ **스크롤**")

# 메인 페이지 내용
show_header()
st.write("좌측 사이드바에서 원하는 페이지를 선택하세요.") 