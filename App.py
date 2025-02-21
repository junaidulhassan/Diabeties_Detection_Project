import streamlit as st
import pickle
import pandas as pd
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder

# Function to apply Label Encoding
def label_encode_columns(X):
    return X.apply(lambda col: LabelEncoder().fit_transform(col) if col.dtype == 'O' else col)

# Set Page Title
st.set_page_config(page_title="Diabetes Prediction")

# Title
st.title("🩺 Diabetes Prediction Results")

# Mapping for Age Dropdown (Keep as categories)
age_options = ["18-24", "25-29", "30-34", "35-39", "40-44", "45-49", 
               "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80 or older"]

# Sidebar Inputs
st.sidebar.header("Enter Patient Information")
st.sidebar.divider()

age = st.sidebar.selectbox("Select Age Range:", options=age_options)
high_bp = st.sidebar.selectbox("Do you have High Blood Pressure?", ["Yes", "No"])
high_chol = st.sidebar.selectbox("Do you have High Cholesterol?", ["Yes", "No"])
sex = st.sidebar.selectbox("Select Gender:", ["Male", "Female"])
bmi = st.sidebar.number_input("Enter BMI:", min_value=10, max_value=50, step=1)
blood_glucose = st.sidebar.number_input("Enter Blood Glucose Level:", min_value=50, max_value=400, step=3)
smoker = st.sidebar.selectbox("Are you a Smoker?", ["Yes", "No"])
insulin_level = st.sidebar.number_input("Enter Insulin Level:", min_value=0.0, max_value=900.0, step=3.0)
HbA1c_level = st.sidebar.number_input("Enter Hemoglobin Level:", min_value=3.0, max_value=9.0, step=0.5)

# Age	HighBP	HighChol	Smoker	Sex	BMI	blood_glucose_level	Insulin	HbA1c_level	Diabetes_binary
# Convert inputs to a dictionary
input_data = {
    "Age": age,
    "HighBP": high_bp,
    "HighChol": high_chol,
    "Smoker": smoker,
    "Sex": sex,
    "BMI": bmi,
    "blood_glucose_level": blood_glucose,
    "Insulin":insulin_level,
    "HbA1c_level":HbA1c_level
}

# Convert dictionary to DataFrame
input_df = pd.DataFrame([input_data])

# Load the model
model_path = "pipeline.pkl"  # Update the path if needed
if os.path.exists(model_path):
    with open(model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    
    # Make prediction dynamically without button click
    y_pred_proba = loaded_model.predict_proba(input_df)[:, 1]  # Probability of having diabetes
    y_pred_class = loaded_model.predict(input_df)  # Class label
    diabetes_probability = y_pred_proba[0] * 100

    st.write(f"**Probability of Diabetes:** {diabetes_probability:.3f}%")
    st.write(f"**Patient Class:** {'Diabetic' if y_pred_class[0] == 1 else 'Non-Diabetic'}")

    if diabetes_probability>50.0:
        st.warning("Your diabetes risk is estimated to be above 50%. We recommend discussing these results with your doctor for further evaluation and guidance.",icon="🚨")
    
    # Create Subplots
    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{"type": "domain"}, {"type": "indicator"}]]
    )
    
    # Donut Chart
    fig.add_trace(go.Pie(
        values=[diabetes_probability, 100 - diabetes_probability],
        labels=["Diabetes", "No Diabetes"],
        hole=0.4,
        marker=dict(colors=["tomato", "springgreen"]),
        textinfo='percent',
        hoverinfo="label+percent+name"
    ), row=1, col=1)
    
    # Gauge Chart
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=diabetes_probability,
        title={"text": "Diabetes Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "tomato" if diabetes_probability > 50 else "springgreen"},
            "steps": [
                {"range": [0, 100], "color": "White"},
            ]
        }
    ), row=1, col=2)
    
    fig.update_layout(height=400, width=800)
    
    st.plotly_chart(fig)
else:
    st.error("⚠️ Model file not found! Please check the path.")
