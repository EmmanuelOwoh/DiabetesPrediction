import streamlit as st
import pandas as pd
import joblib
import os
import plotly.graph_objects as go
import numpy as np

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #1c2526;
        padding: 20px;
        border-radius: 10px;
        color: #e0e0e0;
    }
    .title {
        color: #60a5fa;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .subtitle {
        color: #94a3b8;
        font-size: 20px;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #3b82f6;
        color: white;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        border-radius: 8px;
        border: 1px solid #4b5e7a;
        padding: 10px;
        font-size: 16px;
        background-color: #2d3748;
        color: #e0e0e0;
    }
    .sidebar .sidebar-content {
        background-color: #2d3748;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        color: #e0e0e0;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
    }
    .success {
        background-color: #2f855a;
        color: #d4edda;
    }
    .warning {
        background-color: #b7791f;
        color: #fefcbf;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    if not os.path.exists('random_forest_model.pkl'):
        st.error("Model file 'random_forest_model.pkl' not found.")
        st.stop()
    return joblib.load('random_forest_model.pkl')

model = load_model()

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
        This application uses a trained Random Forest model to predict the likelihood of diabetes based on medical features.

        **Input Features:**
        - Pregnancies
        - Glucose
        - Blood Pressure
        - Skin Thickness
        - Insulin
        - BMI
        - Diabetes Pedigree Function
        - Age

        Developed by Emmanuel | Powered by Streamlit
    """)

# Page title & logo
st.image("https://img.icons8.com/fluency/96/medical-doctor.png", width=80)
st.markdown('<div class="title">Diabetes Prediction App 🩺</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter patient details to predict diabetes risk</div>', unsafe_allow_html=True)

# Input layout inside expander
with st.expander("📝 Enter Patient Information"):
    st.subheader("Patient Information")
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1,
                                      help="Number of times the patient has been pregnant.")
        glucose = st.number_input("Glucose (mg/dL)", min_value=0.0, max_value=300.0, value=0.0,
                                  help="Plasma glucose concentration after fasting.")
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0.0, max_value=200.0, value=0.0,
                                         help="Diastolic blood pressure in mm Hg.")
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=0.0,
                                         help="Triceps skin fold thickness.")

    with col2:
        insulin = st.number_input("Insulin (mu U/ml)", min_value=0.0, max_value=1000.0, value=0.0,
                                  help="2-Hour serum insulin.")
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=100.0, value=0.0,
                              help="Body mass index (weight in kg/(height in m)^2).")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0, step=0.01,
                              help="Function that scores the likelihood of diabetes based on family history.")
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=0,
                              help="Age of the patient.")

submit = st.button("Predict")

# Prediction logic
def make_prediction(inputs):
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    input_df = pd.DataFrame([inputs], columns=feature_names)
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1] * 100
    return prediction, probability, input_df, feature_names

# Gauge visualization
def display_probability_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability,
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "#2f855a"},
                {'range': [30, 70], 'color': "#b7791f"},
                {'range': [70, 100], 'color': "#dc2626"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability
            }
        },
        title={'text': "Diabetes Risk Probability (%)"}
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("This gauge represents the model's confidence in the patient being diabetic.")

# Feature importance chart
def display_feature_impact(input_df, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        weighted = np.array(input_df.values[0]) * importances
        df_imp = pd.DataFrame({
            'Feature': feature_names,
            'Weighted Contribution': weighted
        }).sort_values(by='Weighted Contribution', ascending=False)

        st.subheader("🧬 Feature Impact")
        st.bar_chart(df_imp.set_index('Feature'))

# Prediction handling
if submit:
    inputs = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]

    try:
        prediction, probability, input_df, feature_names = make_prediction(inputs)

        if prediction == 1:
            st.markdown(f'<div class="prediction-box warning">🧪 <b>Prediction:</b> Diabetic (Probability: {probability:.2f}%)</div>', unsafe_allow_html=True)
            st.warning("⚠️ The patient is at risk of diabetes. Please consult a healthcare professional.")
        else:
            st.markdown(f'<div class="prediction-box success">✅ <b>Prediction:</b> Not Diabetic (Probability: {probability:.2f}%)</div>', unsafe_allow_html=True)
            st.success("🎉 The patient is not likely diabetic. Keep monitoring your health regularly.")

        display_probability_gauge(probability)
        display_feature_impact(input_df, feature_names)

        st.subheader("🔍 Input Summary")
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Footer
st.markdown("""
    <hr style="border-top: 1px solid #4b5e7a; margin-top: 40px;">
    <div style="text-align: center; color: #94a3b8; font-size: 14px;">
        © 2025 Diabetes Prediction App. All rights reserved.
    </div>
""", unsafe_allow_html=True)
