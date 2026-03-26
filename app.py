import streamlit as st
import numpy as np
import pickle

# Load model and scaler

model = pickle.load(open('insurance_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("💊 Medical Insurance Cost Predictor")

st.write("Enter patient details to predict insurance charges")

# User Inputs

age = st.slider("Age", 18, 100, 30)

sex = st.selectbox("Sex", ["Male", "Female"])
sex = 1 if sex == "Male" else 0

bmi = st.slider("BMI", 10.0, 50.0, 25.0)

children = st.slider("Number of Children", 0, 5, 0)

smoker = st.selectbox("Smoker", ["Yes", "No"])
smoker = 1 if smoker == "Yes" else 0

region = st.selectbox("Region", ["Northeast", "Northwest", "Southeast", "Southwest"])
region_dict = {"Northeast":0, "Northwest":1, "Southeast":2, "Southwest":3}
region = region_dict[region]

# Prediction

if st.button("Predict Insurance Cost"):
	input_data = np.array([[age, sex, bmi, children, smoker, region]])
	input_data = scaler.transform(input_data)

	prediction = model.predict(input_data)

	st.success(f"Estimated Insurance Cost: ${prediction[0]:.2f}")

