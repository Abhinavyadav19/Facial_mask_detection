# =====================================================================
# FACE MASK DETECTION - MINIMALIST DEMO DASHBOARD
# =====================================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Mask Detection", layout="centered")

# Hide Streamlit header/footer for clean presentation
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("😷 Face Mask Detection System")
st.caption("Computer Vision Classification • MobileNetV2 Transfer Learning")
st.divider()

# Load cached trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mask_detector_model.keras")

model = load_model()

# Image file uploader
uploaded_file = st.file_uploader("Upload a face image to inspect", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Facial Input", use_container_width=True)
    
    # Preprocess image to match training shape (224x224)
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, 224, 224, 3)
    
    # Run prediction
    prediction = model.predict(img_batch)[0][0]
    
    st.divider()
    # 0 = with_mask, 1 = without_mask
    if prediction < 0.5:
        confidence = (1.0 - prediction) * 100
        st.success(f"✅ Status: **MASK DETECTED** (Confidence: {confidence:.1f}%)")
        st.info("Public Health Enforcement: Compliant with campus safety protocol.")
    else:
        confidence = prediction * 100
        st.error(f"❌ Status: **NO MASK DETECTED** (Confidence: {confidence:.1f}%)")
        st.warning("Public Health Enforcement: Safety violation. Mask required for entry.")