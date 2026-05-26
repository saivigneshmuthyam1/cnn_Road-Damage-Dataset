# =========================================================
# AI-BASED ROAD DAMAGE DETECTION SYSTEM
# STREAMLIT DEPLOYMENT
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="Road Damage Detection System",

    page_icon="🚧",

    layout="wide"

)

# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(
    "road_damage_model.keras"
)

# =========================================================
# LOAD LABEL MAPPING
# =========================================================

classes = np.load(
    "label_mapping.npy",
    allow_pickle=True
)

# =========================================================
# SECTION 1 — HEADER
# =========================================================

st.title("🚧 AI-Based Road Damage Detection System")

st.subheader(
    "Smart City Infrastructure Monitoring using CNN"
)

st.markdown("---")

# =========================================================
# SECTION 2 — ABOUT PROJECT
# =========================================================

st.header("📌 About the Project")

st.write("""
Road monitoring is important for maintaining safe transportation systems.
Delayed identification of potholes and cracks may increase accidents,
traffic congestion, and vehicle damage.

This project uses Convolutional Neural Networks (CNNs) in Computer Vision
to automatically analyze road images and detect different types of road damage.

Industry Applications:
- Smart City Monitoring
- Automated Road Inspection
- Municipal Maintenance Systems
- Highway Safety Analysis
- Infrastructure Management
""")

st.markdown("---")

# =========================================================
# SECTION 3 — UPLOAD AREA
# =========================================================

st.header("📤 Upload Road Image")

uploaded_file = st.file_uploader(

    "Choose a road image",

    type=["jpg", "jpeg", "png"],

    help="Upload JPG, JPEG, or PNG images only"

)

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    # Convert PIL image to array
    img = np.array(image)

    # Resize
    img_resized = cv2.resize(img, (128,128))

    # Normalize
    img_normalized = img_resized / 255.0

    # Expand dimensions
    img_input = np.expand_dims(img_normalized, axis=0)

    # Predict
    prediction = model.predict(img_input)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    return prediction[0], predicted_class, confidence

# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # =====================================================
    # SECTION 4 — IMAGE PREVIEW
    # =====================================================

    with col1:

        st.header("🖼 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    probabilities, predicted_class, confidence = predict_image(image)

    predicted_label = classes[predicted_class]

    # =====================================================
    # SEVERITY LOGIC
    # =====================================================

    if confidence > 0.85:
        severity = "High"

    elif confidence > 0.60:
        severity = "Medium"

    else:
        severity = "Low"

    # =====================================================
    # SECTION 5 — PREDICTION AREA
    # =====================================================

    with col2:

        st.header("📊 Prediction Result")

        st.success(
            f"Prediction: {predicted_label}"
        )

        st.info(
            f"Confidence: {confidence*100:.2f}%"
        )

        st.warning(
            f"Severity Level: {severity}"
        )

    st.markdown("---")

    # =====================================================
    # SECTION 6 — VISUALIZATION AREA
    # =====================================================

    st.header("📈 Class Confidence Graph")

    probability_df = pd.DataFrame({

        "Class": classes,

        "Confidence": probabilities

    })

    st.bar_chart(
        probability_df.set_index("Class")
    )

    # =====================================================
    # SECTION 7 — RECOMMENDATIONS
    # =====================================================

    st.header("🛠 Recommendations")

    if predicted_label == "potholes":

        st.error("""
Immediate maintenance recommended.

High-risk road condition detected.
Potential danger for vehicles and pedestrians.
""")

    elif predicted_label == "cracks":

        st.warning("""
Road surface cracks detected.

Preventive maintenance is recommended
to avoid further road deterioration.
""")

    elif predicted_label == "manholes":

        st.info("""
Manhole structure detected.

Ensure proper alignment and maintenance
to avoid vehicle instability.
""")

    else:

        st.write("No major issue detected.")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.write(
    "🚀 Smart City AI Monitoring System using CNN and Streamlit"
)
