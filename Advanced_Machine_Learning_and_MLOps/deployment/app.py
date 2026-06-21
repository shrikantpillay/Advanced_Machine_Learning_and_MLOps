import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="shrikantpillay/Advanced_Machine_Learning_and_MLOps", filename="best_machine_failure_model.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Machine Failure Prediction App")
st.write("""
This application predicts the likelihood of a machine failing based on its operational parameters.
Please enter the sensor and configuration data below to get a prediction.
""")

# User input
Type = st.selectbox("Machine Type", ["H", "L", "M"])
ProdTaken = st.number_input("ProdTaken", min_value=0.0, max_value=1, value=1, step=1)
Age = st.number_input("Age", min_value=18, max_value=61, value=18, step=1)
CityTier = st.number_input("CityTier", min_value=1, max_value=1, value=1, step=1)
DurationOfPitch = st.number_input("DurationOfPitch", min_value=5, max_value=127, value=5,step=1)
NumberOfPersonVisiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=5, value=2,step=1)
NumberOfFollowups = st.number_input("NumberOfFollowups", min_value=1, max_value=6, value=1,step=1)
PreferredPropertyStar = st.number_input("PreferredPropertyStar", min_value=3, max_value=5, value=3)
NumberOfTrips = st.number_input("NumberOfTrips", min_value=1, max_value=22, value=1)
Passport = st.number_input("Passport", min_value=0, max_value=1, value=1)
PitchSatisfactionScore = st.number_input("PitchSatisfactionScore", min_value=1, max_value=5, value=1)
OwnCar = st.number_input("Own Car", min_value=0, max_value=1, value=1)
NumberOfChildrenVisiting = st.number_input("Number of Childern Visiting", min_value=0, max_value=3, value=1)
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=2000)

# Assemble input into DataFramTool Wear (min)e
input_data = pd.DataFrame([{
    'Type': Type,
    'ProdTaken': ProdTaken,
    'Age': Age,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
}])


if st.button("Predict Failure"):
    prediction = model.predict(input_data)[0]
    result = "Machine Failure" if prediction == 1 else "No Failure"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
