# =========================================================
# AI-BASED ROAD DAMAGE DETECTION SYSTEM
# PROFESSIONAL STREAMLIT UI
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

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(

    page_title="Road Damage Detection",

    page_icon="🚧",

    layout="wide"

)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""

<style>

.main {
    background-color: #f5f7fb;
}

/* HEADER */
.header-container {
    background: linear-gradient(90deg,#031633,#082b63);
    padding: 35px;
    border-radius: 0px 0px 25px 25px;
    margin-bottom: 25px;
}

.header-flex {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.header-text {
    color: white;
}

.header-title {
    font-size: 52px;
    font-weight: 800;
    line-height: 1.1;
}

.header-subtitle {
    color: #f4c542;
    font-size: 28px;
    margin-top: 10px;
    font-weight: 600;
}

.header-img img {
    border-radius: 15px;
}

/* SECTION CARDS */
.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 2px 12px rgba(0,0,0,0.08);
    margin-bottom: 25px;
}

/* SECTION TITLE */
.section-title {
    font-size: 30px;
    font-weight: 700;
    color: #0a2d6b;
    margin-bottom: 15px;
}

/* RESULT BOX */
.result-box {
    background-color: #f8fbff;
    border: 1px solid #dbe7ff;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
}

.result-title {
    color: #0a2d6b;
    font-size: 20px;
    font-weight: 600;
}

.result-value {
    font-size: 35px;
    font-weight: 800;
    margin-top: 10px;
}

.footer {
    background: linear-gradient(90deg,#031633,#082b63);
    color: white;
    padding: 20px;
    border-radius: 20px 20px 0px 0px;
    text-align: center;
    margin-top: 40px;
}

</style>

""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = tf.keras.models.load_model(
    "road_damage_model.keras"
)

# =========================================================
# LOAD LABELS
# =========================================================

classes = np.load(
    "label_mapping.npy",
    allow_pickle=True
)

# =========================================================
# HEADER SECTION
# =========================================================

st.markdown("""

<div class="header-container">

<div class="header-flex">

<div class="header-text">

<div class="header-title">
🚧 AI-Based Road Damage Detection System
</div>

<div class="header-subtitle">
Smart City Infrastructure Monitoring using CNN
</div>

</div>

</div>

</div>

""", unsafe_allow_html=True)

# =========================================================
# ABOUT PROJECT SECTION
# =========================================================

st.markdown("""
<div class="card">

<div class="section-title">
📌 About the Project
</div>

Road monitoring is important for maintaining safe transportation systems.
Delayed identification of potholes and cracks may increase accidents,
traffic congestion, and vehicle damage.

<br><br>

This project uses Convolutional Neural Networks (CNNs) in Computer Vision
to automatically analyze road images and detect different types of road damage.

<br><br>

<b>Industry Applications:</b>

<ul>
<li>Smart City Monitoring</li>
<li>Automated Road Inspection</li>
<li>Municipal Maintenance Systems</li>
<li>Highway Safety Analysis</li>
<li>Infrastructure Management</li>
</ul>

</div>

""", unsafe_allow_html=True)

# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown("""
<div class="section-title">
📤 Upload Road Image
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(

    "Choose a road image",

    type=["jpg", "jpeg", "png"]

)

# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    img = np.array(image)

    img = cv2.resize(img, (128,128))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = np.argmax(prediction)

    confidence = np.max(prediction)

    return prediction[0], predicted_class, confidence

# =========================================================
# PROCESS IMAGE
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    probabilities, predicted_class, confidence = predict_image(image)

    predicted_label = classes[predicted_class]

    # -----------------------------------------------------
    # SEVERITY LEVEL
    # -----------------------------------------------------

    if confidence > 0.85:
        severity = "High"

    elif confidence > 0.60:
        severity = "Medium"

    else:
        severity = "Low"

    # =====================================================
    # IMAGE + RESULT SECTION
    # =====================================================

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # IMAGE PREVIEW
    # -----------------------------------------------------

    with col1:

        st.markdown("""
        <div class="card">
        <div class="section-title">
        🖼 Uploaded Image
        </div>
        """, unsafe_allow_html=True)

        st.image(
            image,
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # PREDICTION RESULT
    # -----------------------------------------------------

    with col2:

        st.markdown("""
        <div class="card">
        <div class="section-title">
        🎯 Prediction Result
        </div>
        """, unsafe_allow_html=True)

        # Prediction
        st.markdown(f"""
        <div class="result-box">

        <div class="result-title">
        Prediction
        </div>

        <div class="result-value" style="color:#16a34a;">
        {predicted_label}
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # Confidence
        st.markdown(f"""
        <div class="result-box">

        <div class="result-title">
        Confidence
        </div>

        <div class="result-value" style="color:#2563eb;">
        {confidence*100:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # Severity
        st.markdown(f"""
        <div class="result-box">

        <div class="result-title">
        Severity Level
        </div>

        <div class="result-value" style="color:#ea580c;">
        {severity}
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # CONFIDENCE GRAPH
    # =====================================================

    st.markdown("""
    <div class="card">
    <div class="section-title">
    📊 Class Confidence Graph
    </div>
    """, unsafe_allow_html=True)

    probability_df = pd.DataFrame({

        "Class": classes,

        "Confidence": probabilities

    })

    st.bar_chart(
        probability_df.set_index("Class")
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    st.markdown("""
    <div class="card">
    <div class="section-title">
    🛠 Recommendations
    </div>
    """, unsafe_allow_html=True)

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

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("""

<div class="footer">

<h3>
🏙 Smart City AI Monitoring System
</h3>

<p>
using CNN and Streamlit
</p>

</div>

""", unsafe_allow_html=True)
