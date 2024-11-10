
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the trained model and encoders (replace with your actual paths)
# Assuming you saved them using pickle or joblib
import joblib
model = joblib.load('knn_penguin_model.pkl') 
island_encoder = joblib.load('island_encoder.pkl')
sex_encoder = joblib.load('sex_encoder.pkl')
species_encoder = joblib.load('species_encoder.pkl')


st.title("Penguin Species Prediction")

# Input fields for user to enter penguin features
island = st.selectbox("Island", island_encoder.classes_)
culmen_length_mm = st.number_input("Culmen Length (mm)")
culmen_depth_mm = st.number_input("Culmen Depth (mm)")
flipper_length_mm = st.number_input("Flipper Length (mm)")
body_mass_g = st.number_input("Body Mass (g)")
sex = st.selectbox("Sex", sex_encoder.classes_)

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length_mm],
    'culmen_depth_mm': [culmen_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

# Preprocess the input data (similar to how you trained your model)
input_data['island'] = island_encoder.transform(input_data['island']).astype(int)
input_data['sex'] = sex_encoder.transform(input_data['sex']).astype(int)

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    predicted_species = species_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Species: **{predicted_species}**")


