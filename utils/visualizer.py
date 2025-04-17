import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from config import VIZ_CONFIG
import numpy as np
import streamlit as st
from typing import List, Dict, Tuple

class Visualizer:
    def __init__(self):
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD']
    
    def create_chart(self, data: pd.DataFrame, chart_type: str, **kwargs) -> go.Figure:
        """Create various types of charts based on the input parameters"""
        if chart_type == 'bar':
            return self._create_bar_chart(data, **kwargs)
        elif chart_type == 'pie':
            return self._create_pie_chart(data, **kwargs)
        elif chart_type == 'scatter':
            return self._create_scatter_plot(data, **kwargs)
        elif chart_type == 'histogram':
            return self._create_histogram(data, **kwargs)
        elif chart_type == 'box':
            return self._create_box_plot(data, **kwargs)
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")

    def _create_bar_chart(self, data: pd.DataFrame, x: str, y: str, title: str, color: str = None) -> go.Figure:
        fig = px.bar(data, x=x, y=y, title=title, color=color)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        return fig

    def _create_pie_chart(self, data: pd.DataFrame, names: str, values: str, title: str) -> go.Figure:
        fig = px.pie(data, names=names, values=values, title=title, color_discrete_sequence=self.colors)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        return fig

    def _create_scatter_plot(self, data: pd.DataFrame, x: str, y: str, color: str, title: str) -> go.Figure:
        fig = px.scatter(data, x=x, y=y, color=color, title=title)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        return fig

    def _create_histogram(self, data: pd.DataFrame, x: str, title: str) -> go.Figure:
        fig = px.histogram(data, x=x, title=title, color_discrete_sequence=self.colors)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        return fig

    def _create_box_plot(self, data: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
        fig = px.box(data, x=x, y=y, title=title, color_discrete_sequence=self.colors)
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(size=12)
        )
        return fig

    def display_prediction_table(self, df: pd.DataFrame) -> None:
        """
        예측 결과 테이블을 Streamlit으로 표시합니다.
        
        Args:
            df (pd.DataFrame): 예측 결과를 포함하는 DataFrame
                - customer_id: 고객 ID
                - churn_risk: 이탈 위험도
                - top_feature_1, importance_1: 첫 번째 영향 요인과 중요도
                - top_feature_2, importance_2: 두 번째 영향 요인과 중요도
                - top_feature_3, importance_3: 세 번째 영향 요인과 중요도
        """
        # 필요한 컬럼만 선택
        display_df = df[['customer_id', 'churn_risk', 
                        'top_feature_1', 'importance_1',
                        'top_feature_2', 'importance_2',
                        'top_feature_3', 'importance_3']].copy()
        
        # 컬럼명 변경
        display_df.columns = ['고객 ID', '이탈 위험도',
                            '영향 요인 1', '중요도 1',
                            '영향 요인 2', '중요도 2',
                            '영향 요인 3', '중요도 3']
        
        # 이탈 위험도와 중요도를 퍼센트로 변환
        display_df['이탈 위험도'] = display_df['이탈 위험도'].apply(lambda x: f"{x:.1%}")
        display_df['중요도 1'] = display_df['중요도 1'].apply(lambda x: f"{x:.1%}")
        display_df['중요도 2'] = display_df['중요도 2'].apply(lambda x: f"{x:.1%}")
        display_df['중요도 3'] = display_df['중요도 3'].apply(lambda x: f"{x:.1%}")
        
        # 테이블 스타일 설정
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "고객 ID": st.column_config.TextColumn("고객 ID", width="medium"),
                "이탈 위험도": st.column_config.TextColumn("이탈 위험도", width="small"),
                "영향 요인 1": st.column_config.TextColumn("영향 요인 1", width="medium"),
                "중요도 1": st.column_config.TextColumn("중요도 1", width="small"),
                "영향 요인 2": st.column_config.TextColumn("영향 요인 2", width="medium"),
                "중요도 2": st.column_config.TextColumn("중요도 2", width="small"),
                "영향 요인 3": st.column_config.TextColumn("영향 요인 3", width="medium"),
                "중요도 3": st.column_config.TextColumn("중요도 3", width="small")
            }
        )

    @staticmethod
    def create_churn_gauge(probability):
        """이탈 확률 게이지 차트 (Indicator)"""
        risk_level = "high" if probability > VIZ_CONFIG['thresholds']['medium'] else \
                    "medium" if probability > VIZ_CONFIG['thresholds']['low'] else "low"
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "이탈확률"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': VIZ_CONFIG['colors'][f'{risk_level}_risk']},
                'steps': [
                    {'range': [0, 30], 'color': VIZ_CONFIG['colors']['low_risk']},
                    {'range': [30, 70], 'color': VIZ_CONFIG['colors']['medium_risk']},
                    {'range': [70, 100], 'color': VIZ_CONFIG['colors']['high_risk']}
                ]
            }
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_bar_chart(data, x, y, title="", orientation='v'):
        """기본 막대 그래프 (plotly.express)"""
        fig = px.bar(
            data,
            x=x,
            y=y,
            title=title,
            orientation=orientation,
            color_discrete_sequence=[VIZ_CONFIG['colors']['medium_risk']]
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_custom_bar_chart(data, x, y, title="", positive_color=None, negative_color=None):
        """커스텀 막대 그래프 (plotly.graph_objects)"""
        positive_color = positive_color or VIZ_CONFIG['colors']['low_risk']
        negative_color = negative_color or VIZ_CONFIG['colors']['high_risk']
        
        fig = go.Figure()
        
        # 양수 값
        fig.add_trace(go.Bar(
            x=x,
            y=[val if val > 0 else 0 for val in y],
            name='Positive',
            marker_color=positive_color
        ))
        
        # 음수 값
        fig.add_trace(go.Bar(
            x=x,
            y=[val if val < 0 else 0 for val in y],
            name='Negative',
            marker_color=negative_color
        ))
        
        fig.update_layout(
            title=title,
            barmode='relative',
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig

    @staticmethod
    def create_feature_importance(feature_importance, feature_names):
        """특성 중요도 바 차트"""
        fig = px.bar(
            x=feature_names,
            y=feature_importance,
            title="특성 중요도",
            labels={'x': '특성', 'y': '중요도'},
            color_discrete_sequence=[VIZ_CONFIG['colors']['medium_risk']]
        )
        return fig

    @staticmethod
    def create_customer_timeline(data):
        """고객 타임라인 시각화"""
        fig = go.Figure()
        
        # 타임라인 시각화 로직 구현
        # 예: 주문 이력, 이탈 위험도 변화 등
        
        return fig

    @staticmethod
    def create_risk_distribution(data, column='churn_probability'):
        """이탈 위험도 분포 시각화"""
        fig = px.histogram(data, x=column, nbins=20, title='이탈 위험도 분포')
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        return fig

    @staticmethod
    def create_correlation_heatmap(correlation_matrix):
        """상관관계 히트맵을 생성합니다."""
        # 상관관계 행렬이 DataFrame인 경우 numpy 배열로 변환
        if isinstance(correlation_matrix, pd.DataFrame):
            z = correlation_matrix.values
            x = correlation_matrix.columns.tolist()
            y = correlation_matrix.index.tolist()
        else:
            z = correlation_matrix
            x = list(range(correlation_matrix.shape[1]))
            y = list(range(correlation_matrix.shape[0]))
        
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale='RdBu',
            zmid=0,
            text=np.round(z, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='상관관계 히트맵',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0)
        )
        
        return fig 