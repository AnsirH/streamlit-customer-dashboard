import streamlit as st
from components.header import show_header
from components.animations import add_page_transition
from utils.visualizer import Visualizer
from utils.data_generator import generate_sample_data
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os


# CSV 경로 설정 (models 폴더 안에 있는 파일이라면)
csv_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'correlation_coefficient.csv')

# 데이터 불러오기 함수
@st.cache_data
def load_data():
    return pd.read_csv(csv_path)

# 데이터 불러오기
try:
    df = load_data()
except FileNotFoundError:
    st.error(f"파일을 찾을 수 없습니다: {csv_path}")
except Exception as e:
    st.error(f"오류 발생: {e}")




def show():
    try:
        add_page_transition()
        show_header()

        st.title("전체 고객 데이터 보기")
        st.subheader("이탈 여부와의 상관관계를 확인할 기준 칼럼들을 선택하세요")

        # 수치형 컬럼 목록
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        # 다중 선택 UI
        selected_cols = st.multiselect("기준 컬럼 선택", options=numeric_cols)

        if st.button("적용하기") and selected_cols:
            for selected_col in selected_cols:
                st.markdown(f"---")
                st.subheader(f"기준: **{selected_col}**")

                # 상관계수 계산
                other_cols = [col for col in numeric_cols if col != selected_col]
                corr_values = df[[selected_col] + other_cols].corr()[selected_col].drop(selected_col)
                corr_df = corr_values.reset_index()
                corr_df.columns = ['변수', '상관계수']

                # 시각화
                fig = px.bar(
                    corr_df,
                    x='상관계수',
                    y='변수',
                    orientation='h',
                    text='상관계수',
                    color='상관계수',
                    color_continuous_scale='Teal'
                )
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                fig.update_layout(title=f"{selected_col}과 다른 변수들과의 상관관계", height=500)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("기준 컬럼들을 다중 선택한 후 [적용하기] 버튼을 눌러주세요.")

    except Exception as e:
        st.error(f"오류 발생: {e}")



    