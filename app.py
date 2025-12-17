import pickle
import numpy as np
import streamlit as st
import os
print(os.listdir())
with open("/mount/src/fish_poly/fish_poly.pkl", "rb") as f:
    PR, poly, le = pickle.load(f)

# Load model, poly features, and label encoder
with open("fish_poly.pkl", "rb") as f:
    PR, poly, le = pickle.load(f)
st.title("🐟 Fish Weight Prediction")

species = st.selectbox("Select Species", le.classes_)
l1 = st.number_input("Length1")
l2 = st.number_input("Length2")
l3 = st.number_input("Length3")
h  = st.number_input("Height")
w  = st.number_input("Width")

if st.button("Predict Weight"):
    species_enc = le.transform([species])[0]
    input_array = np.array([[species_enc, l1, l2, l3, h, w]])
    input_poly = poly.transform(input_array)
    weight = PR.predict(input_poly)[0]
    st.success(f"Predicted Weight: {weight:.2f} grams")

