import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from config import VIZ_CONFIG
import numpy as np
import streamlit as st

from typing import List, Dict, Tuple
import shap
import matplotlib.pyplot as plt


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
            use_container_width=True
        )
        
        # 테이블이 표시된 후 페이지 새로고침
        st.experimental_rerun()

    @staticmethod
    def create_churn_gauge(probability):
        """이탈 확률 게이지 차트 (Indicator)"""
        try:
            # probability가 None이거나 숫자가 아닌 경우 0으로 처리
            if probability is None or not isinstance(probability, (int, float)):
                probability = 0
                st.warning("이탈 확률을 계산할 수 없습니다.")

            # 확률값을 0-1 범위로 제한
            probability = max(0, min(1, float(probability)))
            
            # 위험도 수준 결정 (0-100 스케일)
            prob_percentage = probability * 100
            
            # 게이지 색상 결정
            if prob_percentage >= 70:
                gauge_color = "#FF4B4B"
            elif prob_percentage >= 30:
                gauge_color = "#FFA500"
            else:
                gauge_color = "#32CD32"
            
            # Plotly Figure 객체 생성
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_percentage,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={
                    'suffix': "%",
                    'font': {'size': 36, 'color': 'white'}  # 확률값 크기를 36으로 증가
                },
                gauge={
                    'shape': "angular",
                    'axis': {
                        'range': [0, 100],
                        'tickwidth': 1,
                        'tickcolor': "darkblue",
                        'tickvals': [0, 30, 50, 70, 100],
                        'ticktext': ["0%", "30%", "50%", "70%", "100%"]
                    },
                    'bar': {'color': gauge_color, 'thickness': 0.8},  # 게이지 바 색상을 검정색으로 변경
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': prob_percentage
                    }
                }
            ))

            # 레이아웃 설정
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
                margin=dict(l=30, r=30, t=100, b=30)
            )
            
            return fig
            
        except Exception as e:
            st.error(f"게이지 차트 생성 중 오류 발생: {str(e)}")
            return go.Figure()  # 빈 Figure 반환

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
    def create_risk_distribution(data, column='churn_prob'):
        """이탈 위험 분포를 시각화합니다."""
        fig = px.histogram(data, x=column, nbins=50,
                          title='이탈 위험 분포',
                          labels={column: '이탈 확률', 'count': '고객 수'})
        fig.update_layout(bargap=0.1)
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

    @staticmethod
    def create_shap_waterfall(explainer, shap_values, idx, feature_names):
        """SHAP Waterfall Plot 생성"""
        fig = plt.figure()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value[1],
            shap_values[1][idx],
            feature_names=feature_names,
            show=False
        )
        return fig

    @staticmethod
    def create_shap_summary(shap_values, feature_names):
        """SHAP Summary Plot 생성"""
        fig = plt.figure()
        shap.summary_plot(shap_values[1], feature_names=feature_names, show=False)
        return fig

    @staticmethod
    def create_top_features_impact(shap_values, feature_names, idx, top_n=5):
        """상위 영향력 feature들을 DataFrame으로 반환"""
        impact_df = pd.DataFrame({
            'feature': feature_names,
            'shap_value': shap_values[1][idx]
        })
        impact_df['abs_impact'] = impact_df['shap_value'].abs()
        top_features = impact_df.sort_values(by='abs_impact', ascending=False).head(top_n)
        return top_features[['feature', 'shap_value']].set_index('feature')

    @staticmethod
    def create_feature_importance_bar(shap_values, feature_names, top_n=10):
        """특성 중요도 바 차트 (SHAP 기반)"""
        mean_shap = np.abs(shap_values[1]).mean(axis=0)
        sorted_idx = np.argsort(mean_shap)[-top_n:]
        
        fig = px.bar(
            x=mean_shap[sorted_idx],
            y=[feature_names[i] for i in sorted_idx],
            orientation='h',
            title="SHAP 기반 특성 중요도",
            labels={'x': '평균 |SHAP 값|', 'y': '특성'},
            color_discrete_sequence=[VIZ_CONFIG['colors']['medium_risk']]
        )
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig 

    @staticmethod
    def create_correlation_bar(df, customer_id=None):
        """선택된 고객의 특성별 이탈 영향도를 막대 그래프로 시각화합니다."""
        try:
            # 특성 이름과 한글 매핑
            features = {
                'Tenure': '거래 기간',
                'CityTier': '도시 등급',
                'WarehouseToHome': '배송 거리',
                'HourSpendOnApp': '앱 사용 시간',
                'NumberOfDeviceRegistered': '등록된 기기 수',
                'SatisfactionScore': '만족도',
                'NumberOfAddress': '배송지 수',
                'Complain': '불만 제기',
                'OrderAmountHikeFromlastYear': '주문 금액 증가율',
                'CouponUsed': '쿠폰 사용 횟수',
                'OrderCount': '주문 횟수',
                'DaySinceLastOrder': '마지막 주문 경과일',
                'CashbackAmount': '캐시백 금액'
            }
            
            # 수치형 컬럼만 선택
            numeric_cols = list(features.keys())
            
            if customer_id is not None and customer_id in df.index:
                # 선택된 고객의 데이터
                customer_data = df.loc[customer_id]
                
                # 각 특성의 영향도 계산
                feature_impacts = []
                feature_names = []
                
                for col in numeric_cols:
                    if col in df.columns:
                        # 해당 특성의 평균과 표준편차 계산
                        mean_val = df[col].mean()
                        std_val = df[col].std()
                        
                        if std_val != 0:
                            # Z-score 계산으로 영향도 산출
                            z_score = (customer_data[col] - mean_val) / std_val
                            # 이탈 확률과의 상관관계 방향 고려
                            corr = df[col].corr(df['churn_prob'])
                            impact = z_score * corr
                            
                            feature_impacts.append(impact)
                            feature_names.append(features[col])
                
                # 데이터프레임 생성
                impact_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Impact': feature_impacts
                })
                
                # 절대값 기준으로 정렬
                impact_df['abs_impact'] = impact_df['Impact'].abs()
                impact_df = impact_df.sort_values('abs_impact', ascending=True)
                
                # 색상 설정 (빨간색: 이탈 위험 증가, 파란색: 이탈 위험 감소)
                colors = ['#FF6B6B' if x > 0 else '#4ECDC4' for x in impact_df['Impact']]
                
                # 막대 그래프 생성
                fig = go.Figure()
                
                # 양수 값 (이탈 위험 증가)
                positive_mask = impact_df['Impact'] > 0
                if positive_mask.any():
                    fig.add_trace(go.Bar(
                        y=impact_df[positive_mask]['Feature'],
                        x=impact_df[positive_mask]['Impact'],
                        orientation='h',
                        name='이탈 위험 증가',
                        marker_color='#FF6B6B',
                        text=[f'+{x:.2f}' for x in impact_df[positive_mask]['Impact']],
                        textposition='auto',
                    ))
                
                # 음수 값 (이탈 위험 감소)
                negative_mask = impact_df['Impact'] <= 0
                if negative_mask.any():
                    fig.add_trace(go.Bar(
                        y=impact_df[negative_mask]['Feature'],
                        x=impact_df[negative_mask]['Impact'],
                        orientation='h',
                        name='이탈 위험 감소',
                        marker_color='#4ECDC4',
                        text=[f'{x:.2f}' for x in impact_df[negative_mask]['Impact']],
                        textposition='auto',
                    ))
                
                # 레이아웃 설정
                fig.update_layout(
                    title=f'고객 {customer_id}의 특성별 이탈 영향도',
                    xaxis_title='이탈 영향도 (음수: 위험 감소, 양수: 위험 증가)',
                    yaxis_title='특성',
                    showlegend=True,
                    legend=dict(
                        orientation='h',
                        yanchor='bottom',
                        y=1.02,
                        xanchor='right',
                        x=1
                    ),
                    height=500,
                    xaxis=dict(
                        gridcolor='rgba(255,255,255,0.1)',
                        zerolinecolor='rgba(255,255,255,0.2)',
                    ),
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                return fig
            
            return None
            
        except Exception as e:
            st.error(f"이탈 영향도 그래프 생성 중 오류 발생: {str(e)}")
            return None 