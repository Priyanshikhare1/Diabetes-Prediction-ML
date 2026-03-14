import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Page title
st.title("🩺 Diabetes Risk Prediction System")

st.markdown("""
This application predicts the **risk of diabetes** based on patient health parameters.

⚠ **Disclaimer:** This tool is for educational purposes only and should not be used as a medical diagnosis.
""")

# Sidebar navigation
st.sidebar.title("Navigation")

page = st.sidebar.selectbox(
    "Go to",
    ["Home", "Prediction", "About Model"]
)

# HOME PAGE
if page == "Home":
    st.subheader("Welcome 👋")
    st.write("""
    This AI system estimates diabetes risk using machine learning.

    You can:
    - Enter patient health data
    - Predict diabetes risk
    - View probability of diabetes
    """)

# ABOUT MODEL PAGE
elif page == "About Model":
    st.subheader("Model Information")

    st.write("""
    **Algorithm Used:** Random Forest Classifier  

    **Dataset:** Pima Indians Diabetes Dataset  

    **Features used for prediction:**
    - Pregnancies
    - Glucose
    - Blood Pressure
    - Skin Thickness
    - Insulin
    - BMI
    - Diabetes Pedigree Function
    - Age
    """)

# PREDICTION PAGE
elif page == "Prediction":

    st.subheader("Enter Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0)
        glucose = st.number_input("Glucose Level", min_value=0)
        bp = st.number_input("Blood Pressure", min_value=0)
        skin = st.number_input("Skin Thickness", min_value=0)

    with col2:
        insulin = st.number_input("Insulin Level", min_value=0)
        bmi = st.number_input("BMI", min_value=0.0)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
        age = st.number_input("Age", min_value=0)

    # Prediction button
    if st.button("Predict Diabetes Risk"):

        input_data = np.array([[pregnancies, glucose, bp, skin,
                                insulin, bmi, dpf, age]])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        risk = probability[0][1] * 100

        st.metric("Diabetes Risk", f"{risk:.2f}%")

        if prediction[0] == 1:
            st.error("⚠ High chance of Diabetes")

            st.markdown("""
            ### Recommended Actions
            - Consult a healthcare professional
            - Maintain a healthy diet
            - Exercise regularly
            - Monitor blood sugar levels
            """)
        else:
            st.success("Low chance of Diabetes")