# =========================================================
# TASK 10 — STREAMLIT DEPLOYMENT
# ROAD DAMAGE DETECTION SYSTEM
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import sys
print(sys.executable)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="Road Damage Detection",

    page_icon="🚧",

    layout="centered"

)

# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(
    "road_damage_model.keras"
)

# =========================================================
# LOAD LABEL MAPPINGS
# =========================================================

classes = np.load(
    "label_mapping.npy",
    allow_pickle=True
)

# =========================================================
# APPLICATION TITLE
# =========================================================

st.title("🚧 Road Damage Detection System")

st.markdown("""
This AI-powered system detects:

- potholes
- cracks
- manholes

using a CNN model.
""")

# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(

    "Upload Road Image",

    type=["jpg", "jpeg", "png"]

)

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    # Convert image to numpy array
    img = np.array(image)

    # Resize image
    img = cv2.resize(img, (128,128))

    # Normalize image
    img = img / 255.0

    # Expand dimensions
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    return predicted_class, confidence

# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # -----------------------------------------------------
    # IMAGE PREVIEW
    # -----------------------------------------------------

    st.subheader("Uploaded Image")

    st.image(

        image,

        use_container_width=True

    )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    with st.spinner("Analyzing Road Image..."):

        predicted_class, confidence = predict_image(image)

    # -----------------------------------------------------
    # PREDICTION RESULT
    # -----------------------------------------------------

    st.subheader("Prediction Result")

    st.success(
        f"{classes[predicted_class]}"
    )

    # -----------------------------------------------------
    # CONFIDENCE SCORE
    # -----------------------------------------------------

    st.subheader("Confidence Score")

    st.write(
        f"{confidence:.2f}"
    )

    st.progress(float(confidence))

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.write(
    "Smart City AI Road Monitoring System"
)