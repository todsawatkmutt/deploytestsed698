import streamlit as st
import pandas as pd
import pickle

# Load the trained model and encoders (make sure they are in the correct directory)
try:
    with open('model_penguin_66130701715.pkl', 'rb') as file:
        model, species_encoder, island_encoder, sex_encoder = pickle.load(file)
except FileNotFoundError as e:
    st.error(f"Error loading model or encoder files: {e}")
    st.stop()

# Title of the Streamlit app
st.title("Penguin Species Prediction")

# Input fields for the user to enter penguin features
island = st.selectbox("Island", island_encoder.classes_)
culmen_length_mm = st.number_input("Culmen Length (mm)", min_value=0.0)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", min_value=0.0)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=0.0)
body_mass_g = st.number_input("Body Mass (g)", min_value=0.0)
sex = st.selectbox("Sex", sex_encoder.classes_)

# Create a DataFrame from the user input
input_data = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Check for missing values
if input_data.isnull().values.any():
    st.error("Input contains missing values. Please ensure all fields are filled.")
    st.stop()

# Encode categorical variables
try:
    input_data['island'] = island_encoder.transform(input_data['island']).astype(int)
    input_data['sex'] = sex_encoder.transform(input_data['sex']).astype(int)
except Exception as e:
    st.error(f"Error during encoding: {e}")
    st.stop()

# Convert columns to numeric (just to make sure)
input_data['culmen_length_mm'] = pd.to_numeric(input_data['culmen_length_mm'], errors='coerce')
input_data['culmen_depth_mm'] = pd.to_numeric(input_data['culmen_depth_mm'], errors='coerce')
input_data['flipper_length_mm'] = pd.to_numeric(input_data['flipper_length_mm'], errors='coerce')
input_data['body_mass_g'] = pd.to_numeric(input_data['body_mass_g'], errors='coerce')

# Ensure the columns are in the correct format
st.write("User Input (processed):")
st.write(input_data)

# Check if any of the columns were coerced to NaN during conversion
if input_data.isnull().values.any():
    st.error("There are invalid (NaN) values in the input data after conversion. Please check the input values.")
    st.stop()

# Prediction button
if st.button("Predict"):
    try:
        # Make prediction
        prediction = model.predict(input_data)
        
        # Inverse transform the prediction to get the species name
        predicted_species = species_encoder.inverse_transform(prediction)[0]
        
        # Display the result
        st.success(f"Predicted Species: **{predicted_species}**")
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
