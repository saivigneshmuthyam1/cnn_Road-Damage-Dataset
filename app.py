# =========================================================
# AI-BASED ROAD DAMAGE DETECTION SYSTEM
# STREAMLIT APPLICATION
# PROFESSIONAL BLUE & WHITE UI
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
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(

    page_title="Road Damage Detection System",

    page_icon="🚧",

    layout="wide"

)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""

<style>

/* MAIN BACKGROUND */

.stApp {

    background-color: #050b18;
}

/* REMOVE EXTRA SPACE */

.block-container {

    padding-top: 1.2rem;

    padding-bottom: 1rem;
}

/* HEADER */

.header-container {

    background: linear-gradient(
        90deg,
        #031633,
        #0a2d6b
    );

    padding: 35px;

    border-radius: 22px;

    margin-bottom: 30px;
}

/* HEADER TITLE */

.header-title {

    color: white;

    font-size: 54px;

    font-weight: 800;

    line-height: 1.1;
}

/* HEADER SUBTITLE */

.header-subtitle {

    color: #c7dcff;

    font-size: 24px;

    margin-top: 12px;

    font-weight: 500;
}

/* CARDS */

.card {

    background-color: white;

    padding: 25px;

    border-radius: 18px;

    margin-bottom: 25px;

    box-shadow: 0px 4px 14px rgba(0,0,0,0.08);
}

/* CARD TEXT FIX */

.card p,
.card li,
.card h4 {

    color: #0f172a !important;

    font-size: 17px;

    line-height: 1.8;
}

/* SECTION TITLES */

.section-title {

    color: #0a2d6b;

    font-size: 30px;

    font-weight: 700;

    margin-bottom: 20px;
}

/* RESULT BOX */

.result-box {

    background-color: #f8fbff;

    border: 1px solid #dbeafe;

    border-radius: 15px;

    padding: 22px;

    margin-bottom: 18px;

    text-align: center;
}

/* RESULT TITLE */

.result-title {

    color: #0a2d6b;

    font-size: 18px;

    font-weight: 600;
}

/* RESULT VALUE */

.result-value {

    font-size: 34px;

    font-weight: 800;

    margin-top: 10px;

    color: #2563eb;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {

    background-color: white;

    border: 2px dashed #2563eb;

    border-radius: 15px;

    padding: 20px;
}

/* FOOTER */

.footer {

    background: linear-gradient(
        90deg,
        #031633,
        #0a2d6b
    );

    color: white;

    padding: 25px;

    border-radius: 20px;

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

<div style="
display:flex;
justify-content:space-between;
align-items:center;
gap:30px;
flex-wrap:wrap;
">

<div>

<div class="header-title">
🚧 AI-Based Road <br>
Damage Detection System
</div>

<div class="header-subtitle">
Smart City Infrastructure Monitoring using CNN
</div>

</div>

<div>

<img 
src="https://c.files.bbci.co.uk/af6d/live/b595d020-215f-11f1-b539-43e5d7f861d2.jpg"
width="420"
style="
border-radius:18px;
box-shadow:0px 4px 15px rgba(0,0,0,0.3);
object-fit:cover;
"
/>

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

<p>

Road monitoring is important for maintaining safe transportation systems.
Delayed identification of potholes and cracks may increase accidents,
traffic congestion, and vehicle damage.

</p>

<p>

This project uses Convolutional Neural Networks (CNNs)
to automatically analyze road images and detect different types of road damage.

</p>

<h4>
Industry Applications
</h4>

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
# UPLOAD SECTION TITLE
# =========================================================

st.markdown("""

<div class="section-title">
📤 Upload Road Image
</div>

""", unsafe_allow_html=True)

# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(

    "Choose JPG / PNG Road Image",

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

    # =====================================================
    # SEVERITY LEVEL
    # =====================================================

    if confidence > 0.85:

        severity = "High"

    elif confidence > 0.60:

        severity = "Medium"

    else:

        severity = "Low"

    # =====================================================
    # COLUMNS
    # =====================================================

    col1, col2 = st.columns(2)

    # =====================================================
    # IMAGE PREVIEW
    # =====================================================

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

    # =====================================================
    # PREDICTION AREA
    # =====================================================

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

        # Confidence

        st.markdown(f"""

        <div class="result-box">

        <div class="result-title">
        Confidence
        </div>

        <div class="result-value">
        {confidence*100:.2f}%
        </div>

        </div>

        """, unsafe_allow_html=True)

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

<h2>
🏙 Smart City AI Monitoring System
</h2>

<p style="font-size:18px;">
using CNN and Streamlit
</p>

</div>

""", unsafe_allow_html=True)
