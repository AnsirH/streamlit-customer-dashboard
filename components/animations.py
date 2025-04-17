import streamlit as st

def add_page_transition():
    st.markdown("""
    <style>
    /* 페이지 전환 애니메이션 */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .stApp {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    .stMarkdown, .stDataFrame, .stPlotlyChart {
        animation: slideIn 0.5s ease-out;
    }
    
    /* 버튼 호버 효과 */
    .stButton > button {
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True) 