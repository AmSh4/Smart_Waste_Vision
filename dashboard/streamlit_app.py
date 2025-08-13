"""
Streamlit dashboard: dashboard/streamlit_app.py
Run: streamlit run dashboard/streamlit_app.py
"""
import streamlit as st
from PIL import Image
import requests, io, os, time
st.set_page_config(page_title="SmartWasteVision Dashboard", layout="centered")

st.title("SmartWasteVision — Waste Sorting Demo")
st.write("Upload an image to get a predicted waste class. A tiny local model is used (demo).")

uploaded = st.file_uploader("Upload waste image", type=["png","jpg","jpeg"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)
    if st.button("Predict with local model"):
        from src.model.model import load_model, predict_image
        model, classes, device = load_model()
        res = predict_image(img, model=model, classes=classes, device=device)
        st.write("**Prediction**:", res["class"], f'({res["confidence"]:.2f})')
        st.json(res)
    st.write("---")

st.sidebar.header("Demo Controls")
st.sidebar.write("This demo uses a tiny CNN trained on synthetic images. For production, replace model with a YOLO or ResNet variant and more data.")
st.sidebar.markdown("**Dataset path**: `data/sample_images`")
