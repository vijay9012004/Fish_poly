import pickle
import numpy as np
import streamlit as st
import os

# Page Config
st.set_page_config(page_title="Fish Weight Predictor", page_icon="🐟")

# Load Model with Caching to improve performance
@st.cache_resource
def load_model():
    try:
        with open("fish_poly.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'fish_poly.pkl' not found. Please ensure it is in the directory.")
        return None

model_data = load_model()

if model_data:
    PR, poly, le = model_data

    st.title("🐟 Fish Weight Prediction")
    st.write("Enter the fish characteristics below to estimate its weight.")

    # Sidebar or Columns for better UI
    species = st.selectbox("Select Species", le.classes_)
    
    col1, col2 = st.columns(2)
    with col1:
        l1 = st.number_input("Vertical Length (cm)", min_value=0.0, value=10.0)
        l2 = st.number_input("Diagonal Length (cm)", min_value=0.0, value=11.0)
        l3 = st.number_input("Cross Length (cm)", min_value=0.0, value=12.0)
    with col2:
        h = st.number_input("Height (cm)", min_value=0.0, value=5.0)
        w = st.number_input("Diagonal Width (cm)", min_value=0.0, value=3.0)

    if st.button("Predict Weight", type="primary"):
        # Data Transformation
        species_enc = le.transform([species])[0]
        input_array = np.array([[species_enc, l1, l2, l3, h, w]])
        
        # Polynomial feature expansion
        input_poly = poly.transform(input_array)
        weight = PR.predict(input_poly)[0]
        
        # Display Result
        if weight < 0:
            st.warning(f"The predicted weight is {weight:.2f}g. This suggests the input dimensions are unlikely for this species.")
        else:
            st.metric(label="Estimated Weight", value=f"{weight:.2f} grams")
