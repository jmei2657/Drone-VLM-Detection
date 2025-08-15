import os
import cv2
import time
import torch
import streamlit as st
from PIL import Image
from djitellopy import Tello
from transformers import AutoProcessor, AutoModelForVision2Seq

# Initialize Streamlit
st.set_page_config(page_title="Drone Threat Detection", layout="centered")
st.title("🔍 Physical Threat Detection with Drone or Webcam")

# Load model and processor
processor = AutoProcessor.from_pretrained("SmolVLM/SmolVLM-500M-Instruct")
model = AutoModelForVision2Seq.from_pretrained("SmolVLM/SmolVLM-500M-Instruct")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Threat keywords to look for
security_threats = ["laptop", "phone", "id card", "badge", "paper", "notebook"]

# Try to connect to Tello drone
use_tello = False
tello = None
cap = None

try:
    tello = Tello()
    tello.connect()
    tello.streamon()
    use_tello = True
except:
    cap = cv2.VideoCapture(0)

# Sidebar drone controls
if use_tello:
    st.sidebar.header("🚁 Drone Controls")
    if st.sidebar.button("🟢 Start Drone"):
        try:
            tello.takeoff()
            st.sidebar.success("Drone taking off")
        except Exception as e:
            st.sidebar.error(f"Takeoff failed: {e}")

    if st.sidebar.button("🔴 Stop Drone"):
        try:
            tello.land()
            st.sidebar.success("Drone landing")
        except Exception as e:
            st.sidebar.error(f"Landing failed: {e}")

# Upload fallback image
uploaded_image = st.file_uploader("📤 Upload an image (if no drone or webcam)", type=["jpg", "png"])

# System prompt to guide captioning
system_prompt = (
    "You are a security analyst reviewing a drone photo for physical cybersecurity threats.\n"
    "List any objects that could indicate a security threat, such as: laptop, phone, ID card, badge, paper, notebook.\n"
    "For each object you find, describe:\n"
    "- What the object looks like (color, size, shape, position...)\n"
    "- Where it is located (top left, center, etc.)"
)

def detect_threats(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prompt = system_prompt
    inputs = processor(prompt, image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=128)
    caption = processor.batch_decode(output, skip_special_tokens=True)[0].lower()
    found = [word for word in security_threats if word in caption]
    return caption, found

frame_placeholder = st.empty()
caption_placeholder = st.empty()

# Main loop
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    frame = cv2.cvtColor(cv2.imread(uploaded_image.name), cv2.COLOR_BGR2RGB)
    caption, found = detect_threats(frame)
    st.image(image, caption=f"Threats Detected: {', '.join(found)}\n\n{caption}")

elif use_tello or cap:
    while True:
        if use_tello:
            frame = tello.get_frame_read().frame
        else:
            ret, frame = cap.read()
            if not ret:
                st.warning("No webcam input detected.")
                break

        caption, found = detect_threats(frame)
        frame_placeholder.image(frame, channels="BGR", caption=f"Threats: {', '.join(found)}")
        caption_placeholder.info(caption)

        if not use_tello:
            time.sleep(0.1)
else:
    st.warning("No drone, webcam, or uploaded image available.")
