"""
UI/UX Utilities Module
Provides enhanced user interface components and visualizations for Streamlit
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any


class UIComponents:
    """Provides reusable UI components for Streamlit."""
    
    @staticmethod
    def render_page_header(title: str, subtitle: str = "", icon: str = "🩺"):
        """
        Render a professional page header.
        
        Args:
            title (str): Main title
            subtitle (str): Optional subtitle
            icon (str): Header icon
        """
        col1, col2 = st.columns([0.1, 0.9])
        with col1:
            st.write(icon)
        with col2:
            st.title(title)
            if subtitle:
                st.markdown(f"*{subtitle}*")
        st.divider()
    
    @staticmethod
    def render_info_cards(metrics: Dict[str, Tuple[str, str]]):
        """
        Render metric cards in a row.
        
        Args:
            metrics (Dict): Dictionary of {label: (value, color)}
        """
        cols = st.columns(len(metrics))
        for col, (label, (value, color)) in zip(cols, metrics.items()):
            with col:
                st.metric(label, value)
    
    @staticmethod
    def render_risk_alert(
        probability: float,
        risk_level: str,
        recommendation: str,
        icon: str = "🚨"
    ):
        """
        Render risk alert with color coding.
        
        Args:
            probability (float): Risk probability
            risk_level (str): Risk level (Low/Moderate/High)
            recommendation (str): Recommendation text
            icon (str): Alert icon
        """
        if probability > 50:
            st.warning(
                f"{icon} **High Risk Alert**\n\n"
                f"Diabetes Risk: {probability:.2f}%\n\n"
                f"**Recommendation**: {recommendation}",
                icon="⚠️"
            )
        elif probability > 30:
            st.info(
                f"**Moderate Risk**\n\n"
                f"Diabetes Risk: {probability:.2f}%\n\n"
                f"**Recommendation**: {recommendation}",
                icon="ℹ️"
            )
        else:
            st.success(
                f"**Low Risk**\n\n"
                f"Diabetes Risk: {probability:.2f}%\n\n"
                f"**Recommendation**: {recommendation}",
                icon="✅"
            )
    
    @staticmethod
    def render_input_section(
        age_options: List[str],
        gender_options: List[str],
        binary_options: List[str]
    ) -> Dict[str, Any]:
        """
        Render comprehensive input form in sidebar.
        
        Args:
            age_options (List): Available age ranges
            gender_options (List): Available genders
            binary_options (List): Yes/No options
            
        Returns:
            Dict: User inputs
        """
        st.sidebar.header("👤 Patient Information")
        st.sidebar.divider()
        
        inputs = {}
        
        # Demographic Information
        with st.sidebar.expander("📋 Demographic Info", expanded=True):
            inputs["age"] = st.selectbox(
                "Select Age Range:",
                options=age_options,
                help="Select the age range that applies to you"
            )
            inputs["sex"] = st.selectbox(
                "Gender:",
                options=gender_options,
                help="Select biological sex"
            )
        
        # Medical History
        with st.sidebar.expander("🏥 Medical History", expanded=True):
            inputs["high_bp"] = st.selectbox(
                "High Blood Pressure:",
                options=binary_options,
                help="Have you been diagnosed with high blood pressure?"
            )
            inputs["high_chol"] = st.selectbox(
                "High Cholesterol:",
                options=binary_options,
                help="Have you been diagnosed with high cholesterol?"
            )
            inputs["smoker"] = st.selectbox(
                "Smoker:",
                options=binary_options,
                help="Have you smoked at least 100 cigarettes in your lifetime?"
            )
        
        # Health Metrics
        with st.sidebar.expander("📊 Health Metrics", expanded=True):
            inputs["bmi"] = st.slider(
                "Body Mass Index (BMI):",
                min_value=10.0,
                max_value=50.0,
                step=0.5,
                help="BMI = Weight(kg) / Height²(m²)"
            )
            inputs["blood_glucose"] = st.slider(
                "Blood Glucose Level (mg/dL):",
                min_value=50,
                max_value=400,
                step=5,
                help="Fasting or random blood glucose level"
            )
            inputs["insulin_level"] = st.slider(
                "Insulin Level (mU/L):",
                min_value=0.0,
                max_value=900.0,
                step=5.0,
                help="Fasting insulin level"
            )
            inputs["hba1c_level"] = st.slider(
                "Hemoglobin A1c Level (%):",
                min_value=3.0,
                max_value=9.0,
                step=0.1,
                help="HbA1c reflects 2-3 month average blood glucose"
            )
        
        return inputs
    
    @staticmethod
    def render_health_metrics_summary(input_data: Dict[str, Any]):
        """
        Render summary of input health metrics.
        
        Args:
            input_data (Dict): User input data
        """
        st.subheader("📊 Health Metrics Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "BMI",
                f"{input_data.get('bmi', 0):.1f}",
                delta="kg/m²",
                help="Body Mass Index"
            )
            st.metric(
                "Blood Glucose",
                f"{input_data.get('blood_glucose', 0):.0f}",
                delta="mg/dL",
                help="Blood glucose level"
            )
        
        with col2:
            st.metric(
                "Insulin Level",
                f"{input_data.get('insulin_level', 0):.1f}",
                delta="mU/L",
                help="Fasting insulin level"
            )
            st.metric(
                "HbA1c Level",
                f"{input_data.get('hba1c_level', 0):.1f}",
                delta="%",
                help="Hemoglobin A1c level"
            )


class Visualizations:
    """Provides interactive visualizations using Plotly."""
    
    @staticmethod
    def create_risk_gauge_chart(probability: float) -> go.Figure:
        """
        Create a gauge chart for risk visualization.
        
        Args:
            probability (float): Risk probability (0-100)
            
        Returns:
            go.Figure: Plotly figure
        """
        # Determine color based on probability
        if probability > 50:
            gauge_color = "#FF4444"
        elif probability > 30:
            gauge_color = "#FFA500"
        else:
            gauge_color = "#00C851"
        
        fig = go.Figure(data=[
            go.Indicator(
                mode="gauge+number+delta",
                value=probability,
                title={"text": "Diabetes Risk Probability"},
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                    "bar": {"color": gauge_color, "thickness": 0.3},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {"range": [0, 30], "color": "#E8F5E9"},
                        {"range": [30, 50], "color": "#FFF3E0"},
                        {"range": [50, 100], "color": "#FFEBEE"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 2},
                        "thickness": 0.75,
                        "value": 50,
                    }
                },
                suffix="%",
                number={"suffix": "%"},
            )
        ])
        
        fig.update_layout(
            font={"family": "Arial", "size": 15},
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    @staticmethod
    def create_risk_distribution_chart(probability: float) -> go.Figure:
        """
        Create a pie chart showing diabetes risk distribution.
        
        Args:
            probability (float): Risk probability (0-100)
            
        Returns:
            go.Figure: Plotly figure
        """
        diabetes_prob = probability
        non_diabetes_prob = 100 - probability
        
        fig = go.Figure(data=[
            go.Pie(
                labels=["Diabetes Risk", "No Diabetes Risk"],
                values=[diabetes_prob, non_diabetes_prob],
                hole=0.4,
                marker=dict(colors=["#FF6B6B", "#51CF66"]),
                textposition="inside",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Probability: %{value:.2f}%<extra></extra>",
            )
        ])
        
        fig.update_layout(
            title="Risk Distribution",
            font={"family": "Arial", "size": 12},
            height=400,
            showlegend=True,
            legend=dict(x=0.85, y=0.5)
        )
        
        return fig
    
    @staticmethod
    def create_combined_risk_visualization(probability: float) -> go.Figure:
        """
        Create combined visualization with gauge and pie charts.
        
        Args:
            probability (float): Risk probability (0-100)
            
        Returns:
            go.Figure: Plotly figure
        """
        # Determine color based on probability
        if probability > 50:
            gauge_color = "#FF4444"
        elif probability > 30:
            gauge_color = "#FFA500"
        else:
            gauge_color = "#00C851"
        
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "indicator"}, {"type": "domain"}]],
            subplot_titles=("Risk Probability", "Risk Distribution")
        )
        
        # Gauge chart
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=probability,
                title={"text": "Diabetes Risk (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": gauge_color},
                    "steps": [
                        {"range": [0, 30], "color": "#E8F5E9"},
                        {"range": [30, 50], "color": "#FFF3E0"},
                        {"range": [50, 100], "color": "#FFEBEE"},
                    ],
                },
                suffix="%",
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=["Diabetes Risk", "No Diabetes Risk"],
                values=[probability, 100 - probability],
                hole=0.3,
                marker=dict(colors=["#FF6B6B", "#51CF66"]),
                textinfo="percent",
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            font={"family": "Arial", "size": 12}
        )
        
        return fig
    
    @staticmethod
    def create_health_metrics_comparison(
        input_data: Dict[str, Any],
        clinical_references: Dict[str, Dict[str, Tuple[float, float]]]
    ) -> go.Figure:
        """
        Create a comparison chart for health metrics against clinical references.
        
        Args:
            input_data (Dict): User input data
            clinical_references (Dict): Clinical reference ranges
            
        Returns:
            go.Figure: Plotly figure
        """
        metrics = ["BMI", "Blood Glucose", "HbA1c"]
        user_values = [
            input_data.get("bmi", 0),
            input_data.get("blood_glucose", 0) / 10,  # Scale for visibility
            input_data.get("hba1c_level", 0) * 10,  # Scale for visibility
        ]
        
        fig = go.Figure(data=[
            go.Bar(name="Your Values", x=metrics, y=user_values, marker_color="indianred"),
        ])
        
        fig.update_layout(
            title="Health Metrics Overview",
            xaxis_title="Health Metrics",
            yaxis_title="Values (scaled)",
            height=400,
            hovermode="x unified",
            template="plotly_white"
        )
        
        return fig


class Tables:
    """Provides table formatting utilities."""
    
    @staticmethod
    def display_patient_summary(input_data: Dict[str, Any]):
        """
        Display patient information in a formatted table.
        
        Args:
            input_data (Dict): Patient input data
        """
        st.subheader("👤 Patient Information Summary")
        
        summary_data = {
            "Age Range": input_data.get("age", "N/A"),
            "Gender": input_data.get("sex", "N/A"),
            "High Blood Pressure": input_data.get("high_bp", "N/A"),
            "High Cholesterol": input_data.get("high_chol", "N/A"),
            "Smoker": input_data.get("smoker", "N/A"),
        }
        
        # Display as two columns
        col1, col2 = st.columns(2)
        
        items = list(summary_data.items())
        mid = len(items) // 2
        
        with col1:
            for key, value in items[:mid]:
                st.markdown(f"**{key}:** {value}")
        
        with col2:
            for key, value in items[mid:]:
                st.markdown(f"**{key}:** {value}")
    
    @staticmethod
    def display_prediction_results(
        probability: float,
        classification: str,
        risk_level: str,
        key_factors: List[str]
    ):
        """
        Display prediction results in a formatted layout.
        
        Args:
            probability (float): Risk probability
            classification (str): Classification result
            risk_level (str): Risk level
            key_factors (List): Contributing factors
        """
        st.subheader("📋 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Risk Probability", f"{probability:.2f}%")
        
        with col2:
            st.metric("Classification", classification)
        
        with col3:
            st.metric("Risk Level", risk_level)
        
        if key_factors:
            st.markdown("**Key Contributing Factors:**")
            for factor in key_factors:
                st.markdown(f"• {factor}")
