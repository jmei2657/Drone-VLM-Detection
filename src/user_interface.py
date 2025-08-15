import os
import cv2
import time
import torch
import streamlit as st
from PIL import Image
from djitellopy import Tello
from transformers import AutoProcessor, AutoModelForVision2Seq

BORDER_SIZE = 96


st.set_page_config(page_title="Threat Detection Drone", layout="centered")
st.markdown("<h1 style='text-align: center; color: white;'>THREAT DETECTION DRONE</h1>", unsafe_allow_html=True)


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

@st.cache_resource
def connect_tello():
    tello = Tello()
    tello.connect()
    tello.streamon()
    return tello

col1, col2 = st.columns([1, 1])
left_box = col1.empty()
right_box = col2.empty()
st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
frame_window = st.image([])

def detect_threats(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = processor(system_prompt, image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=32)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0].lower()

    found = [threat for threat in SECURITY_THREATS if threat in caption]
    return found

coords = {"x": 0, "y": 0}

use_tello = False
tello = None

try:
    tello = connect_tello()
    st.success(" Connected to Tello drone")
    use_tello = True
except Exception as e:
    st.error(f" Drone connection failed: {e}")

if use_tello:
    if st.button(" Scan Frame"):
        frame = tello.get_frame_read().frame
        frame = cv2.copyMakeBorder(
            frame, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
            borderType=cv2.BORDER_CONSTANT, value=(255, 0, 0)
        )

        threats_found = detect_threats(frame)

        if threats_found:
            coords["x"] += 1
            coords["y"] += 1
            left_box.markdown(
                f"<div class='custom-box'>Threat(s) Detected:<br>{', '.join(threats_found)}</div>",
                unsafe_allow_html=True
            )
            right_box.markdown(
                f"<div class='custom-box'>Coordinates:<br>({coords['x']}, {coords['y']})</div>",
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

        frame_window.image(frame, channels="BGR", caption="Live Drone Frame")
else:
    st.warning("No drone connected.")
