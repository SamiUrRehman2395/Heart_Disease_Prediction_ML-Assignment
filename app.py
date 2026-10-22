
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_disease_svc.pkl")
    scaler = joblib.load("heart_disease_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.title("❤️ Heart Disease Prediction System")
st.write(
    "Enter the four attributes below to get a machine-learning "
    "prediction. This application is for educational purposes "
    "and is not a medical diagnosis."
)

st.subheader("Patient Information")

age = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=45
)

sex = st.selectbox(
    "Sex",
    options=[1, 0],
    format_func=lambda x: "Male (1)" if x == 1 else "Female (0)"
)

cp = st.selectbox(
    "Chest Pain Type (cp)",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Type 1",
        2: "Type 2",
        3: "Type 3",
        4: "Type 4"
    }[x]
)

thalach = st.number_input(
    "Maximum Heart Rate Achieved (thalach)",
    min_value=50,
    max_value=250,
    value=150
)

if st.button("Predict", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )

    scaled_data = scaler.transform(input_data)
    prediction = int(model.predict(scaled_data)[0])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning(
            "The model predicts that heart disease may be present. "
            "This result is not a diagnosis; consult a qualified "
            "healthcare professional for assessment."
        )
    else:
        st.success(
            "The model predicts no heart disease detected in this "
            "input. This does not rule out heart disease."
        )
