# ==========================================
# Pneumonia Detection Web App
# ==========================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------------------------------
# Page Configuration
# ------------------------------------------

st.set_page_config(page_title="Pneumonia Detection AI", layout="centered")

st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to detect Pneumonia.")

# ------------------------------------------
# Load Model
# ------------------------------------------

@st.cache_resource
def load_trained_model():
    model = load_model("pneumonia_cnn_model.h5")
    return model

model = load_trained_model()

# ------------------------------------------
# Image Preprocessing Function
# ------------------------------------------

def preprocess_image(image):
    # Convert to RGB (important!)
    image = image.convert("RGB")
    
    # Resize
    image = image.resize((224, 224))
    
    # Convert to numpy
    image = np.array(image)
    
    # Normalize
    image = image / 255.0
    
    # Expand dimensions
    image = np.expand_dims(image, axis=0)
    
    return image

# ------------------------------------------
# File Uploader
# ------------------------------------------

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    processed_image = preprocess_image(image)

    prediction = model.predict(processed_image)[0][0]

    confidence = prediction if prediction > 0.5 else 1 - prediction

    if prediction > 0.5:
        st.error(f"⚠️ Pneumonia Detected\nConfidence: {confidence*100:.2f}%")
    else:
        st.success(f"✅ Normal\nConfidence: {confidence*100:.2f}%")
