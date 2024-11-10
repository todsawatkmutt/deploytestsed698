import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders (make sure they are in the correct directory)
try:
    model = joblib.load('knn_penguin_model.pkl')
    species_encoder = joblib.load('species_encoder.pkl')
    island_encoder = joblib.load('island_encoder.pkl')
    sex_encoder = joblib.load('sex_encoder.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading model or encoder files: {e}")
    st.stop()

# Title of the app
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

# Preprocess the input data (same transformation as during training)
input_data['island'] = island_encoder.transform(input_data['island']).astype(int)
input_data['sex'] = sex_encoder.transform(input_data['sex']).astype(int)

# Show the data the user entered
st.write("User Input:")
st.write(input_data)

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
