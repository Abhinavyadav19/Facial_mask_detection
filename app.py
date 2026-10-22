# =====================================================================
# FACE MASK DETECTION - FILE UPLOAD + LIVE WEBCAM DASHBOARD
# =====================================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Face Mask Detection", layout="centered")

# Hide Streamlit header/footer for clean UI
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

st.title("😷 Face Mask Detection System")
st.caption("CNN Image Classification • MobileNetV2 Transfer Learning")
st.divider()

# 1. LOAD CACHED TRAINED MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mask_detector_model.keras")

model = load_model()

# 2. SELECT INPUT MODE (Upload vs Webcam)
source_choice = st.radio("Choose Input Method:", ["Upload Image File", "Use Live Webcam"], horizontal=True)

input_image = None

if source_choice == "Upload Image File":
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)
else:
    cam_shot = st.camera_input("Capture Face via Webcam")
    if cam_shot is not None:
        input_image = Image.open(cam_shot)

# 3. PREDICTION PIPELINE (Runs identically for both File & Webcam)
if input_image is not None:
    image_rgb = input_image.convert("RGB")
    st.image(image_rgb, caption="Analyzed Facial Input", use_container_width=True)
    
    img_resized = image_rgb.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    img_batch = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_batch)[0][0]
    
    st.divider()
    if prediction < 0.5:
        confidence = (1.0 - prediction) * 100
        st.success(f"✅ Status: **MASK DETECTED** (Confidence: {confidence:.1f}%)")
        st.info("Public Health Enforcement: Compliant with campus safety rules.")
    else:
        confidence = prediction * 100
        st.error(f"❌ Status: **NO MASK DETECTED** (Confidence: {confidence:.1f}%)")
        st.warning("Public Health Enforcement: Safety violation. Mask required.")
