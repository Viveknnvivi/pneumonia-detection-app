# ==========================================
# 🩺 Pneumonia Detection Web App
# ==========================================

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

# ------------------------------------------
# 🔹 Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Pneumonia Detection from Chest X-Ray")
st.write("Upload a chest X-ray image to detect Pneumonia using Deep Learning.")

# ------------------------------------------
# 🔹 Google Drive Model Download
# ------------------------------------------

MODEL_PATH = "pneumonia_cnn_model.h5"


FILE_ID = "1m9GPEEiwgwF4u0iENha2_xuWlryyFZ6e"


@st.cache_resource
def load_trained_model():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... Please wait ⏳"):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# ------------------------------------------
# 🔹 Image Preprocessing Function
# ------------------------------------------

def preprocess_image(image):
    # Convert grayscale to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize((224, 224))

    # Convert to numpy array
    image = np.array(image)

    # Normalize
    image = image / 255.0

    # Expand dimensions for model
    image = np.expand_dims(image, axis=0)

    return image

# ------------------------------------------
# 🔹 File Uploader
# ------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded X-ray Image",
        use_container_width=True
    )

    processed_image = preprocess_image(image)

    with st.spinner("Analyzing image..."):
        prediction = model.predict(processed_image)[0][0]

    confidence = prediction if prediction > 0.5 else 1 - prediction

    st.subheader("🔍 Prediction Result")

    if prediction > 0.5:
        st.error(f"⚠️ Pneumonia Detected")
        st.write(f"Confidence: {confidence*100:.2f}%")
    else:
        st.success("✅ Normal")
        st.write(f"Confidence: {confidence*100:.2f}%")

# ------------------------------------------
# 🔹 Footer
# ------------------------------------------

st.markdown("---")
st.caption("Built with TensorFlow & Streamlit | Pneumonia Detection Project")
