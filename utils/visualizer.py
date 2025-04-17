import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from config import VIZ_CONFIG
import numpy as np

class Visualizer:
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
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix, 2),
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