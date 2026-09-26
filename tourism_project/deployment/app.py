import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Customer Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase the Wellness Tourism Package
based on their demographics and interaction data.
Enter the customer details below to get a prediction.
""")

# Input fields for numerical features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=200.0, value=15.0)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=3.0)
preferred_property_star = st.number_input("Preferred Property Star (1-5)", min_value=1.0, max_value=5.0, value=3.0)
number_of_trips = st.number_input("Number of Trips per year", min_value=0.0, max_value=50.0, value=3.0)
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score (1-5)", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Owns a Car", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No") # Changed from number_input
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=5.0, value=0.0)
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=100000.0, value=25000.0)
passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Input fields for categorical features
type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
gender = st.selectbox("Gender", ['Male', 'Female'])
product_pitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
designation = st.selectbox("Designation", ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP'])


input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income
}])

if st.button("Predict Purchase"): # Changed button text
    prediction = model.predict(input_data)[0]
    result = "Customer will purchase the package!" if prediction == 1 else "Customer will not purchase the package."
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
