import streamlit as st
from components.header import show_header
from components.animations import add_page_transition

# 페이지 설정
# st.set_page_config(
#     page_title="이탈 예측 대시보드",
#     page_icon="🚀",
#     layout="wide"
# )

# 세션 상태 초기화
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'

# 애니메이션 적용
add_page_transition()
# st.title("🚀이탈 예측 대시보드")

st.markdown(
    "<h1 style='text-align: center; font-size: 150px; margin: 50px'>PAINT</h1>",
    unsafe_allow_html=True
)

# pages = {
#     "Main": [
#         st.Page("pages\customer_analysis.py")
#     ],
#     "Sub": [
#         st.Page("pages\prediction.py"),
#         st.Page(r"pages\all_data.py"),
#     ],
# }


st.navigation(["pages\customer_analysis.py", "pages\prediction.py", r"pages\all_data.py", "pages\prediction4.py"], position="hidden")

# 사이드바 설정
st.sidebar.title("Main")

# 페이지 이동 버튼
if st.sidebar.button("📊 고객분석", use_container_width=True):
    st.session_state.current_page = 'customer_analysis'
    st.rerun()


# 사이드바 설정
st.sidebar.title("Sub")
if st.sidebar.button("🔮 예측", use_container_width=True):
    st.session_state.current_page = 'prediction'
    st.rerun()
if st.sidebar.button("📈 전체 데이터", use_container_width=True):
    st.session_state.current_page = 'all_data'
    st.rerun()
if st.sidebar.button("🔮 예측4", use_container_width=True):
    st.session_state.current_page = 'prediction4'
    st.rerun()

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
elif st.session_state.current_page == 'prediction4':
    from pages.prediction4 import show
    show() 