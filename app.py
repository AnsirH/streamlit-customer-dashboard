import streamlit as st
from components.header import show_header
from components.animations import add_page_transition
import datetime
from pathlib import Path

# 디버깅 로그 함수
def debug_log(message):
    """디버깅용 로그 기록 함수"""
    log_path = Path(__file__).resolve().parent / "debug_log.txt"
    with open(log_path, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] [app.py] {message}\n")

# 시작 로그
debug_log("app.py 스크립트 시작")

# 페이지 설정
st.set_page_config(
    page_title="이탈 예측 대시보드",
    page_icon="🚀",
    layout="wide"
)
debug_log("페이지 설정 완료")

# 애니메이션 적용
add_page_transition()
debug_log("애니메이션 적용 완료")

# 세션 상태 초기화
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'main'
    debug_log("세션 상태 초기화: 'main' 페이지로 설정")
else:
    debug_log(f"현재 페이지: {st.session_state.current_page}")

# 사이드바 설정
st.sidebar.title("메뉴")
debug_log("사이드바 메뉴 설정 완료")

# 페이지 이동 버튼
if st.sidebar.button("📊 고객분석", use_container_width=True):
    debug_log("고객분석 버튼 클릭됨")
    st.session_state.current_page = 'customer_analysis'
    debug_log("페이지 변경: customer_analysis")
    st.rerun()
if st.sidebar.button("🔮 예측", use_container_width=True):
    debug_log("예측 버튼 클릭됨")
    st.session_state.current_page = 'prediction'
    debug_log("페이지 변경: prediction")
    st.rerun()
if st.sidebar.button("📈 전체 데이터", use_container_width=True):
    debug_log("전체 데이터 버튼 클릭됨")
    st.session_state.current_page = 'all_data'
    debug_log("페이지 변경: all_data")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("⬇️ **스크롤**")

# 현재 페이지에 따라 내용 표시
debug_log(f"페이지 로딩 시작: {st.session_state.current_page}")
try:
    if st.session_state.current_page == 'main':
        debug_log("메인 페이지 표시 시작")
        show_header()
        st.write("좌측 사이드바에서 원하는 페이지를 선택하세요.")
        debug_log("메인 페이지 표시 완료")
    elif st.session_state.current_page == 'customer_analysis':
        debug_log("고객분석 페이지 로드 시작")
        from pages.customer_analysis import show
        debug_log("customer_analysis 모듈 임포트 완료, show() 함수 호출 시작")
        show()
        debug_log("customer_analysis show() 함수 호출 완료")
    elif st.session_state.current_page == 'prediction':
        debug_log("예측 페이지 로드 시작")
        from pages.prediction import show
        debug_log("prediction 모듈 임포트 완료, show() 함수 호출 시작")
        show()
        debug_log("prediction show() 함수 호출 완료")
    elif st.session_state.current_page == 'all_data':
        debug_log("전체 데이터 페이지 로드 시작")
        from pages.all_data import show
        debug_log("all_data 모듈 임포트 완료, show() 함수 호출 시작")
        show()
        debug_log("all_data show() 함수 호출 완료")
except Exception as e:
    debug_log(f"페이지 로드 중 오류 발생: {str(e)}")
    import traceback
    debug_log(traceback.format_exc())
    st.error(f"페이지 로딩 중 오류가 발생했습니다: {str(e)}")

debug_log("app.py 스크립트 종료") 