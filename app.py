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
# PAGE CONFIGURATION
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

/* MAIN BACKGROUND */

.stApp {

    background-color: #f5f7fb;
}

/* REMOVE EXTRA SPACE */

.block-container {

    padding-top: 0rem;

    padding-bottom: 1rem;

    max-width: 100%;
}

/* HIDE STREAMLIT MENU */

#MainMenu {

    visibility: hidden;
}

footer {

    visibility: hidden;
}

/* HEADER SECTION */

.header {

    background: linear-gradient(
        90deg,
        #031633,
        #0a2d6b
    );

    border-radius: 0px 0px 25px 25px;

    padding: 30px 50px;

    margin-bottom: 25px;
}

.header-flex {

    display: flex;

    justify-content: space-between;

    align-items: center;

    gap: 40px;

    flex-wrap: wrap;
}

.header-left {

    flex: 1;
}

.header-title {

    color: white;

    font-size: 60px;

    font-weight: 800;

    line-height: 1.1;
}

.header-subtitle {

    color: #ffd84d;

    font-size: 28px;

    margin-top: 15px;

    font-weight: 600;
}

/* HEADER IMAGE */

.header-image img {

    width: 450px;

    border-radius: 18px;

    box-shadow: 0px 4px 18px rgba(0,0,0,0.3);
}

/* CARDS */

.card {

    background-color: white;

    border-radius: 20px;

    padding: 28px;

    margin-bottom: 25px;

    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
}

/* SECTION TITLES */

.section-title {

    color: #0a2d6b;

    font-size: 34px;

    font-weight: 700;

    margin-bottom: 18px;
}

/* ABOUT TEXT */

.about-text {

    color: #1e293b;

    font-size: 18px;

    line-height: 1.9;
}

/* RESULT BOXES */

.result-box {

    background-color: #f8fbff;

    border: 1px solid #dbeafe;

    border-radius: 18px;

    padding: 24px;

    text-align: center;

    margin-bottom: 20px;
}

.result-title {

    color: #0a2d6b;

    font-size: 20px;

    font-weight: 600;
}

.result-value {

    font-size: 38px;

    font-weight: 800;

    margin-top: 10px;
}

/* FILE UPLOADER */

[data-testid="stFileUploader"] {

    background-color: white;

    border: 2px dashed #2563eb;

    border-radius: 18px;

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

    border-radius: 20px 20px 0px 0px;

    text-align: center;

    margin-top: 40px;
}

/* CHART */

[data-testid="stVerticalBlock"] canvas {

    border-radius: 10px;
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

<div class="header">

<div class="header-flex">

<div class="header-left">

<div class="header-title">
🚧 AI-Based Road <br>
Damage Detection System
</div>

<div class="header-subtitle">
Smart City Infrastructure Monitoring using CNN
</div>

</div>

<div class="header-image">

<img src="https://images.unsplash.com/photo-1504307651254-35680f356dfd?q=80&w=1200&auto=format&fit=crop">

</div>

</div>

</div>

""", unsafe_allow_html=True)

# =========================================================
# ABOUT PROJECT
# =========================================================

col1, col2 = st.columns([2,1])

with col1:

    st.markdown("""

    <div class="card">

    <div class="section-title">
    📌 About the Project
    </div>

    <div class="about-text">

    Road monitoring is important for maintaining safe transportation systems.

    Delayed identification of potholes and cracks may increase accidents,
    traffic congestion, and vehicle damage.

    <br><br>

    This project uses Convolutional Neural Networks (CNNs)
    to automatically analyze road images and detect different types of road damage.

    </div>

    </div>

    """, unsafe_allow_html=True)

with col2:

    st.markdown("""

    <div class="card">

    <div class="section-title">
    🏙 Industry Applications
    </div>

    <div class="about-text">

    • Smart City Monitoring <br><br>

    • Automated Road Inspection <br><br>

    • Municipal Maintenance Systems <br><br>

    • Highway Safety Analysis <br><br>

    • Infrastructure Management

    </div>

    </div>

    """, unsafe_allow_html=True)

# =========================================================
# UPLOAD + IMAGE PREVIEW
# =========================================================

col3, col4 = st.columns(2)

with col3:

    st.markdown("""

    <div class="card">

    <div class="section-title">
    📤 Upload Road Image
    </div>

    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(

        "Choose JPG / PNG Road Image",

        type=["jpg", "jpeg", "png"]

    )

    st.markdown("</div>", unsafe_allow_html=True)

with col4:

    st.markdown("""

    <div class="card">

    <div class="section-title">
    🖼 Uploaded Image
    </div>

    """, unsafe_allow_html=True)

    image_placeholder = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)

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

    image_placeholder.image(
        image,
        use_container_width=True
    )

    probabilities, predicted_class, confidence = predict_image(image)

    predicted_label = classes[predicted_class]

    # =====================================================
    # SEVERITY
    # =====================================================

    if confidence > 0.85:

        severity = "High"

    elif confidence > 0.60:

        severity = "Medium"

    else:

        severity = "Low"

    # =====================================================
    # PREDICTION RESULT
    # =====================================================

    st.markdown("""

    <div class="card">

    <div class="section-title">
    🎯 Prediction Result
    </div>

    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:

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

    with c2:

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

    with c3:

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
    📊 Class Confidence (Probabilities)
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
