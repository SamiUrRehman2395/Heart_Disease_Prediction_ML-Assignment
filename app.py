
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="💙",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("heart_disease_svc.pkl")
    scaler = joblib.load("heart_disease_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

st.title("💙 CardioCare")
st.write(
    "A simple machine-learning application that predicts "
    "heart disease based on four input attributes."
)

st.divider()

st.subheader("Enter Patient Details")

age = st.number_input(
    "Patient Age",
    min_value=1,
    max_value=120,
    value=45
)

sex = st.selectbox(
    "Gender",
    options=[1, 0],
    format_func=lambda x: "Male" if x == 1 else "Female"
)

cp = st.selectbox(
    "Chest Pain Category",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Category 1",
        2: "Category 2",
        3: "Category 3",
        4: "Category 4"
    }[x]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=50,
    max_value=250,
    value=150
)

st.caption("Please check your entered details before continuing.")

if st.button("Check Prediction", type="primary", use_container_width=True):
    input_data = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=["age", "sex", "cp", "thalach"]
    )

    scaled_data = scaler.transform(input_data)
    prediction = int(model.predict(scaled_data)[0])

    st.divider()
    st.subheader("Assessment Result")

    if prediction == 1:
        st.warning(
            "The model predicts that heart disease may be present. "
            "This result is not a diagnosis. Please consult a qualified "
            "healthcare professional."
        )
    else:
        st.success(
            "The model predicts no heart disease detected in this input. "
            "This does not rule out heart disease."
        )

st.caption(
    "For educational purposes only. This application is not a "
    "substitute for professional medical advice."
)

