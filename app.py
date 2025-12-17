import pickle
import numpy as np
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Fish Weight Predictor", page_icon="🐟")

# Load model, poly features, and label encoder
@st.cache_resource
def load_model():
    try:
        with open("fish_poly.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file 'fish_poly.pkl' not found. Please upload it to the directory.")
        return None

model_assets = load_model()

if model_assets:
    PR, poly, le = model_assets

    st.title("🐟 Fish Weight Prediction")
    st.write("Input the dimensions below to estimate the weight of the fish.")

    # Using columns for a cleaner UI
    species = st.selectbox("Select Species", le.classes_)
    
    col1, col2 = st.columns(2)
    with col1:
        l1 = st.number_input("Vertical Length (cm)", min_value=0.0, value=10.0)
        l2 = st.number_input("Diagonal Length (cm)", min_value=0.0, value=11.0)
        l3 = st.number_input("Cross Length (cm)", min_value=0.0, value=12.0)
    
    with col2:
        h = st.number_input("Height (cm)", min_value=0.0, value=5.0)
        w = st.number_input("Width (cm)", min_value=0.0, value=3.0)

    if st.button("Predict Weight", type="primary"):
        # 1. Encode Species
        species_enc = le.transform([species])[0]
        
        # 2. Prepare Input Array
        input_array = np.array([[species_enc, l1, l2, l3, h, w]])
        
        # 3. Transform to Polynomial Features
        input_poly = poly.transform(input_array)
        
        # 4. Predict
        weight = PR.predict(input_poly)[0]
        
        # Display Result
        st.metric(label="Predicted Weight", value=f"{weight:.2f} grams")
