import streamlit as st
from components.header import show_header
from components.animations import add_page_transition
from utils.visualizer import Visualizer
from utils.data_generator import generate_sample_data
import pandas as pd

def show():
    # 애니메이션 적용
    add_page_transition()

###############################################

###############################################