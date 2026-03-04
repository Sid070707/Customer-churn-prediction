import streamlit as st
import pickle
import numpy as np

# Load saved objects
model = pickle.load(open("model/churn_model.pkl", "rb"))
scaler = pickle.load(open("model/scaler.pkl", "rb"))
features = pickle.load(open("model/features.pkl", "rb"))

st.title("Customer Churn Prediction App")

st.write("Enter customer information to predict churn risk.")

# User inputs
tenure = st.slider("Tenure (months)", 0, 72, 12)

monthly_charges = st.slider("Monthly Charges ($)", 0, 150, 70)

total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No"]
)

# Prediction button
if st.button("Predict Churn"):

    # Create empty feature array
    input_data = np.zeros(len(features))

    # Fill numeric features
    input_data[features.get_loc("tenure")] = tenure
    input_data[features.get_loc("MonthlyCharges")] = monthly_charges
    input_data[features.get_loc("TotalCharges")] = total_charges

    # Contract encoding
    if contract == "One year":
        input_data[features.get_loc("Contract_One year")] = 1
    elif contract == "Two year":
        input_data[features.get_loc("Contract_Two year")] = 1

    # Internet service encoding
    if internet_service == "Fiber optic":
        input_data[features.get_loc("InternetService_Fiber optic")] = 1
    elif internet_service == "No":
        input_data[features.get_loc("InternetService_No")] = 1

    # Streaming movies encoding
    if streaming_movies == "Yes":
        input_data[features.get_loc("StreamingMovies_Yes")] = 1

    # Scale input
    input_scaled = scaler.transform([input_data])

    # Prediction
    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    # Display result
    if prediction == 1:
        st.error(f"Customer likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"Customer likely to stay (Probability: {probability:.2f})")