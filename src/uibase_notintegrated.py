import streamlit as st
import numpy as np

# Set page configuration
st.set_page_config(page_title="Threat Detection Drone", layout="centered")

# Custom CSS for background, text colors, centering, and styling
st.markdown("""
    <style>
    /* Black background for entire app */
    .stApp {
        background-color: black;
        color: white;
    }
    /* Center the header text and set white color */
    h1 {
        color: white !important;
        text-align: center;
    }
    /* Style for the red bordered boxes */
    .custom-box {
        border: 2px solid red;
        padding: 30px;
        border-radius: 8px;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        color: white;
        background-color: #222;  /* dark background inside boxes */
    }
    .custom-box > div:first-child {
        margin-bottom: 10px;
    }
    .spacer {
        height: 20px;
    }
    /* Container for drone footage centered horizontally */
    .drone-footage-container {
        display: flex;
        justify-content: center;
        margin-top: 30px;
        padding: 10px;
    }
    /* Style for image caption text */
    .caption {
        color: white !important;
        text-align: center;
        margin-top: 5px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Page header (using markdown h1 with center and white color handled above)
st.markdown("<h1>THREAT DETECTION DRONE</h1>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])
left_box = col1.empty()
right_box = col2.empty()

st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)

left_box.markdown("""
    <div class='custom-box'>
        <div>Threats Detected:</div>
        <div id='threats-output'>Waiting for data...</div>
    </div>
""", unsafe_allow_html=True)

right_box.markdown("""
    <div class='custom-box'>
        <div>Coordinates:</div>
        <div id='coords-output'>(0, 0)</div>
    </div>
""", unsafe_allow_html=True)

# Blank black image for initialization
blank_image = np.zeros((240, 320, 3), dtype=np.uint8)

# Centered container for drone footage with caption below
st.markdown('<div class="drone-footage-container">', unsafe_allow_html=True)
frame_window = st.image(blank_image, use_container_width=False)
st.markdown('<div class="caption">Tello Drone Footage</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
