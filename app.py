import streamlit as st
import pickle
import numpy as np

# Load pickle
with open("fish_poly.pkl", "rb") as f:
    PR, poly, le = pickle.load(f)

st.title("🐟 Fish Weight Prediction App")

# User input
species = st.selectbox("Select Species", le_Species.classes_)
l1 = st.number_input("Enter Length1 (L1)", min_value=0.0, step=0.1)
l2 = st.number_input("Enter Length2 (L2)", min_value=0.0, step=0.1)
l3 = st.number_input("Enter Length3 (L3)", min_value=0.0, step=0.1)
h  = st.number_input("Enter Height (H)", min_value=0.0, step=0.1)
w  = st.number_input("Enter Width (W)", min_value=0.0, step=0.1)

# Predict button
if st.button("Predict Weight"):
    # Encode species
    species_enc = le_Species.transform([species])[0]
    # Create input array
    new_fish = np.array([[species_enc, l1, l2, l3, h, w]])
    # Polynomial transform
    new_fish_poly = poly.transform(new_fish)
    # Predict
    predicted_weight = PR.predict(new_fish_poly)
    st.success(f"Predicted Weight: {predicted_weight[0]:.2f} grams")

