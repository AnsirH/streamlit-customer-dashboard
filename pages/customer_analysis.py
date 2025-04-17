import streamlit as st
import pandas as pd
from components.header import show_header
from components.animations import add_page_transition
from utils.visualizer import Visualizer
from utils.data_generator import generate_sample_data

def show():
    # 애니메이션 적용
    add_page_transition()

    show_header()
    # 데이터 생성
    df = generate_sample_data(n_samples=100)

    # Visualizer 인스턴스 생성
    viz = Visualizer()

    # 테이블 표시
    viz.display_prediction_table(df)