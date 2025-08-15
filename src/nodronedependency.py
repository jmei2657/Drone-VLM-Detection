import os
import cv2
import torch
import numpy as np
import streamlit as st
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

BORDER_SIZE = 96

# Streamlit Page Setup
st.set_page_config(page_title="Threat Detection System", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>THREAT DETECTION SYSTEM</h1>", unsafe_allow_html=True)

# Custom Style
st.markdown("""
    <style>
    .custom-box {
        border: 2px solid red;
        padding: 30px;
        border-radius: 8px;
        height: 150px;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: #f9f9f9;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
    }
    .spacer {
        height: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and processor
processor = AutoProcessor.from_pretrained("./models/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained("./models/SmolVLM-500M-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

SECURITY_THREATS = ["laptop", "phone", "id card", "badge", "paper", "notebook"]

system_prompt = (
    "You are a drone-based security inspector. "
    "Identify if any of the following physical threats are in the image: "
    "laptop, phone, id card, badge, paper, notebook. "
    "Return only the names of the threats you see, separated by commas. "
    "If none found, say 'none'."
)

# Threat detection function
def detect_threats(image):
    inputs = processor(system_prompt, image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=32)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0].lower()
    found = [threat for threat in SECURITY_THREATS if threat in caption]
    return found, caption

# Layout
col1, col2 = st.columns([1, 1])
left_box = col1.empty()
right_box = col2.empty()
st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
frame_window = st.image([])

# Image Upload
uploaded_file = st.file_uploader("Upload an Image for Threat Detection", type=["jpg", "jpeg", "png"])

coords = {"x": 0, "y": 0}

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Add colored border using OpenCV
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    bordered_frame = cv2.copyMakeBorder(
        open_cv_image, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
        borderType=cv2.BORDER_CONSTANT, value=(255, 0, 0)
    )

    # Detect threats
    pil_image_with_border = Image.fromarray(cv2.cvtColor(bordered_frame, cv2.COLOR_BGR2RGB))
    threats_found, full_caption = detect_threats(pil_image_with_border)

    coords["x"] += 1
    coords["y"] += 1

    if threats_found:
        left_box.markdown(
            f"<div class='custom-box'>Threat(s) Detected:<br>{', '.join(threats_found)}</div>",
            unsafe_allow_html=True
        )
    else:
        left_box.markdown(
            "<div class='custom-box'>No Threats</div>", unsafe_allow_html=True
        )

    right_box.markdown(
        f"<div class='custom-box'>Coordinates:<br>({coords['x']}, {coords['y']})</div>",
        unsafe_allow_html=True
    )

    frame_window.image(bordered_frame, channels="BGR", caption="Uploaded Image with Border")
