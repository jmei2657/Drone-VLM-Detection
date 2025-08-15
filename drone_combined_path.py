import time
import cv2
import torch
import numpy as np
import os
import re
from PIL import Image
from djitellopy import Tello
from transformers import AutoProcessor, AutoModelForVision2Seq

# === CONSTANTS ===
MODEL_DIR = "./models/SmolVLM-500M-Instruct"
MOVE_INCREMENT_CM = 25
MAX_ALTITUDE_CM = 91
HOVER_INCREASE_CM = 5
FRAME_STRIDE = 3
WINDOW_NAME = "Tello Security Feed"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960
CAPTION_BAR_PX = 120

security_threats = [
    "EVERWATCH",
    "computer_left_on",
    "cellphone",
    "phone",
    "lanyard",
    "notebook",
    "key",
    "laptop",
    "computer",
    "book",
    "whiteboard",
    "monitor",
    "name tag",
    "id card",
    "code"
]
threats_set = set([t.lower() for t in security_threats])
system_prompt = (
    "You are a security inspector. Identify any of these objects: "
    f"{', '.join(security_threats)}. For each, give:\n"
    "Object: [Name], Location: [e.g., top-left, under desk], Context: [e.g., on table, behind bag].\n"
    "If none found, reply: 'No threats detected.'"
)
user_prompt = "Analyze the image and list relevant objects."

# === INITIALIZE MODEL ===
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
model = AutoModelForVision2Seq.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
model.eval()

# === TELLO SETUP ===
tello = Tello()
tello.connect()
print(f"[INFO] Battery: {tello.get_battery()}%")
tello.streamon()
reader = tello.get_frame_read()

# === TAKEOFF ===
tello.takeoff()
time.sleep(2)
if tello.get_height() + HOVER_INCREASE_CM <= MAX_ALTITUDE_CM:
    tello.move_up(HOVER_INCREASE_CM)

# === UI ===
def compose_canvas(frame, caption, drone_coords=None):
    overlay = frame.copy()
    bar_h = CAPTION_BAR_PX
    overlay[-bar_h:] = (0, 0, 0)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = frame.shape[0] - bar_h + 30
    lines = caption.split('\n')

    for line in lines:
        if drone_coords:
            line = f"({drone_coords[0]},{drone_coords[1]}) - {line}"
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y += 30

    return frame

# == Caption Cleaner ==
def clean_caption(caption):
    if "assistant" in caption.lower():
        parts = caption.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
            if ":" in response:
                response = response.split(":", 1)[-1].strip()
            return response
    return caption

# === THREAT DETECTION ===
def detect_threats(frame):
    image = Image.fromarray(frame)
    prompt = processor.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}
    ], add_generation_prompt=True)

    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=48)
    
    caption = processor.batch_decode(output, skip_special_tokens=True)[0].lower()
    caption = clean_caption(caption)
    # Extract objects
    objects = re.split(r'[,\n;]| and |:', caption)
    detected_threats = []
    for obj in objects:
        obj = obj.strip()
        for threat in threats_set:
            if threat in obj or obj in threat:
                detected_threats.append(threat)
                break
            else:
                # Check for partial matches (e.g., "phone" in "cellphone")
                for threat in threats_set:
                    if threat in obj or obj in threat:
                        detected_threats.append(threat)
                        break

    return ", ".join(detected_threats) if detected_threats else "No threats detected."

# === SPIRAL TRAVERSAL + DETECTION ===
def spiral_traverse(rows, cols):
    visited = [[False]*cols for _ in range(rows)]
    result = []
    dr = [0, 1, 0, -1]
    dc = [1, 0, -1, 0]
    r = c = di = 0

    for _ in range(rows * cols):
        # Detect threats
        frame = reader.frame
        if frame is not None:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            threats = detect_threats(frame)
            canvas = compose_canvas(bgr.copy(), threats, drone_coords=(r, c))
            cv2.imshow(WINDOW_NAME, canvas)
            print(f"[INFO] ({r},{c}) - {threats}")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        visited[r][c] = True
        result.append((r, c))
        tello.move_forward(MOVE_INCREMENT_CM)
        time.sleep(1)

        # Determine next step
        nr, nc = r + dr[di], c + dc[di]
        if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
            r, c = nr, nc
        else:
            di = (di + 1) % 4
            r, c = r + dr[di], c + dc[di]
            tello.rotate_clockwise(90)
            time.sleep(1)

# === RUN ===
try:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    print("[INFO] Starting spiral scan with threat detection")
    spiral_traverse(4, 4)  # Customize grid size

finally:
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()
    print("[INFO] Drone landed and stream closed.")
