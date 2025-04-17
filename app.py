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

# 세션 상태 초기화
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

# 사이드바 설정
st.sidebar.title("메뉴")

# 페이지 이동 버튼
if st.sidebar.button("📊 고객분석", use_container_width=True):
    st.session_state.current_page = 'customer_analysis'
    st.experimental_rerun()
if st.sidebar.button("🔮 예측", use_container_width=True):
    st.session_state.current_page = 'prediction'
    st.experimental_rerun()
if st.sidebar.button("📈 전체 데이터", use_container_width=True):
    st.session_state.current_page = 'all_data'
    st.experimental_rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("⬇️ **스크롤**")

# 현재 페이지에 따라 내용 표시
if st.session_state.current_page == 'main':
    show_header()
    st.write("좌측 사이드바에서 원하는 페이지를 선택하세요.")
elif st.session_state.current_page == 'customer_analysis':
    from pages.customer_analysis import show
    show()
elif st.session_state.current_page == 'prediction':
    from pages.prediction import show
    show()
elif st.session_state.current_page == 'all_data':
    from pages.all_data import show
    show() 