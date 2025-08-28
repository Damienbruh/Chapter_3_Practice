import streamlit as st
import joblib
import numpy as np

# Load the trained pipeline
model = joblib.load("models/car_price_model.pkl")

# Title
st.title("Car price prediction")

# Sidebar inputs
st.sidebar.header("Enter Car Details")

# Define input fields
brand = st.sidebar.selectbox("Brand", ['Kia', 'Chevrolet', 'Mercedes' 'Audi', 'Volkswagen'])
model_name = st.sidebar.text_input("Model", "Rio")
year = st.sidebar.number_input("Year", min_value=1980, max_value=2020)
engine_size = st.sidebar.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
fuel_type = st.sidebar.selectbox("Fuel Type", ['Diesel', 'Hybrid', 'Electric'])
transmission = st.sidebar.selectbox("Transmission", ['Manual', 'Automatic', 'Semi-Automatic'])
mileage = st.sidebar.number_input("Mileage", min_value=0, max_value=500000, value=50000)
doors = st.sidebar.number_input("Doors", min_value=2, max_value=6, value=4)
owner_count = st.sidebar.number_input("Owner Count", min_value=0, max_value=10, value=1)

# Prediction input as DataFrame
input_data = {
    "Brand": [brand],
    "Model": [model_name],
    "Year": [year],
    "Engine_Size": [engine_size],
    "Fuel_Type": [fuel_type],
    "Transmission": [transmission],
    "Mileage": [mileage],
    "Doors": [doors],
    "Owner_Count": [owner_count],
}

# Predict button
if st.sidebar.button("Predict that price baby"):
    import pandas as pd
    import_df = pd.DataFrame(input_data)
    prediction = model.predict(import_df)[0]
    st.success(f"Estimated price is ${int(prediction):,}")

    # Note so you don't forget, run the app in bash with "streamlit run streamlit_app.py"