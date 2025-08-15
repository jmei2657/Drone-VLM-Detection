

import os, time
import json
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Optional on Apple Silicon when you switch to M3:
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import torch
from PIL import Image
from djitellopy import Tello
from transformers import AutoProcessor, AutoModelForVision2Seq
import numpy as np
import textwrap
import re
import time

MAX_ALTITUDE_CM = 91  # 3 feet = ~91.44 cm = 1 yard
HOVER_INCREASE_CM = 5
MOVE_INCREMENT_CM = 25

tello = Tello()
tello.connect()

tello.takeoff()
time.sleep(2)


# Get current height
current_height = tello.get_height()
print(f"Current height: {current_height} cm")

# Hover up a couple of inches (~5 cm) if under limit
if current_height + HOVER_INCREASE_CM <= MAX_ALTITUDE_CM:
    tello.move_up(HOVER_INCREASE_CM)
    print(f"Moved up {HOVER_INCREASE_CM} cm")
else:
    print("Hover increase would exceed 3 ft limit — skipping")

MODEL_DIR = "./models/SmolVLM-500M-Instruct"
FRAME_STRIDE = 5

security_threats = [
    "EverWatch",
    "white_rectangle",
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
    "code",
    "text"
]

threats_inline = ', '.join(security_threats)
# Lowercase set for easy case-insensitive match
threats_set = set([t.lower() for t in security_threats])

system_prompt = (
    f"You are a security inspector. "
    f"Given an image, list only these objects if present: {threats_inline}. "
    "If none are found, reply 'No threats detected.'"
)
user_prompt = "What objects from the list are visible in this image?"

WINDOW_NAME = "Tello Feed"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 960
CAPTION_BAR_PX = 120
SIDE_MARGIN = 20
LINE_SPACING = 8
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
BAR_COLOR = (0, 0, 0)
BAR_ALPHA = 0.55

def suppress_opencv_errors():
    null_fd = os.open(os.devnull, os.O_RDWR)
    old_stderr = os.dup(2)
    os.dup2(null_fd, 2)
    os.close(null_fd)
    return old_stderr

def restore_stderr(old_stderr):
    os.dup2(old_stderr, 2)
    os.close(old_stderr)

def wrap_text_to_width(text, max_width_px, font, font_scale, thickness):
    words = text.split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        test = cur + " " + w
        (tw, th), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw <= max_width_px:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def compose_canvas(frame_bgr, caption, canvas_w, canvas_h,
                   bar_h=CAPTION_BAR_PX, side_margin=SIDE_MARGIN):
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    video_h = canvas_h - bar_h
    fh, fw = frame_bgr.shape[:2]
    if fh == 0 or fw == 0:
        return canvas
    scale = min(canvas_w / fw, video_h / fh)
    new_w = max(1, int(fw * scale))
    new_h = max(1, int(fh * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = (canvas_w - new_w) // 2
    y0 = (video_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    overlay = canvas.copy()
    bar_y0 = canvas_h - bar_h
    cv2.rectangle(overlay, (0, bar_y0), (canvas_w, canvas_h), BAR_COLOR, -1)
    cv2.addWeighted(overlay, BAR_ALPHA, canvas, 1 - BAR_ALPHA, 0, dst=canvas)
    max_text_width = canvas_w - 2 * side_margin
    lines = wrap_text_to_width(caption, max_text_width, FONT, FONT_SCALE, THICKNESS)
    line_heights = []
    for ln in lines:
        (_, lh), _ = cv2.getTextSize(ln, FONT, FONT_SCALE, THICKNESS)
        line_heights.append(lh)
    total_text_h = sum(line_heights) + LINE_SPACING * (len(lines) - 1)
    y = bar_y0 + (bar_h - total_text_h) // 2 + line_heights[0]
    for i, ln in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(ln, FONT, FONT_SCALE, THICKNESS)
        x = (canvas_w - tw) // 2
        cv2.putText(canvas, ln, (x, y), FONT, FONT_SCALE, TEXT_COLOR, THICKNESS, cv2.LINE_AA)
        if i < len(lines) - 1:
            y += line_heights[i + 1] + LINE_SPACING
    return canvas

def clean_caption(caption):
    if "assistant" in caption.lower():
        parts = caption.split("assistant")
        if len(parts) > 1:
            response = parts[-1].strip()
            if ":" in response:
                response = response.split(":", 1)[-1].strip()
            return response
    return caption

def extract_detected_threats(caption, threats_set):
    """Extract detected objects and check if any are threats"""
    caption = caption.lower()
    
    # Debug: print the raw caption
    print(f"[DEBUG] Raw caption: '{caption}'")
    
    # Split by common delimiters and clean up
    objects = re.split(r'[,\n]| and |;|:', caption)
    detected_objects = []
    
    for obj in objects:
        obj = obj.strip()
        if obj and obj not in ['no threats detected', 'none', 'nothing', '']:
            detected_objects.append(obj)
    
    # Debug: print all detected objects
    print(f"[DEBUG] All detected objects: {detected_objects}")
    
    # Check which detected objects are in our threats list
    detected_threats = []
    for obj in detected_objects:
        # Check for exact matches and partial matches
        if obj in threats_set:
            detected_threats.append(obj)
        else:
            # Check for partial matches (e.g., "phone" in "cellphone")
            for threat in threats_set:
                if threat in obj or obj in threat:
                    detected_threats.append(threat)
                    break
    
    # Debug: print threats found
    print(f"[DEBUG] Threats found: {detected_threats}")
    
    return detected_threats, detected_objects

old_stderr = suppress_opencv_errors()

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)
print(f"[INFO] Using device: {device}")

print("[INFO] Loading SmolVLM-500M-Instruct (offline)...")
print(MODEL_DIR)
processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)

if device.type == "cuda":
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16,
        _attn_implementation="flash_attention_2"
    ).to(device)
else:
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_DIR, local_files_only=True, _attn_implementation="eager"
    ).to(device)

model.eval()
print("[INFO] Model loaded.")

tello = Tello()
tello.connect()
print(f"[INFO] Battery: {tello.get_battery()}%")
tello.streamon()
reader = tello.get_frame_read()

print("[INFO] Warming up video stream...")
time.sleep(3.0)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
frame_count = 0
valid_frames_received = 0
current_caption = "predicting..."

try:
    with torch.no_grad():
        while True:
            frame = reader.frame
            if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                time.sleep(0.1)
                continue

            frame_count += 1
            valid_frames_received += 1

            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            canvas = compose_canvas(bgr, current_caption, WINDOW_WIDTH, WINDOW_HEIGHT, CAPTION_BAR_PX)
            cv2.imshow(WINDOW_NAME, canvas)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if valid_frames_received > 10 and frame_count % FRAME_STRIDE == 0:
                current_caption = "Analyzing..."

                try:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": user_prompt},
                            ],
                        },
                    ]
                    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                    image = Image.fromarray(frame)
                    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
                    out_ids = model.generate(**inputs, max_new_tokens=48)
                    raw_caption = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
                    clean_caption_text = clean_caption(raw_caption)

                    detected_threats, detected_objects = extract_detected_threats(clean_caption_text, threats_set)
                    
                    # Show all detected objects for debugging
                    if detected_objects:
                        debug_message = f"All objects: {', '.join(detected_objects)}"
                        print(f"[Frame {frame_count}] {debug_message}")
                    
                    if detected_threats:
                        message = "Threats detected: " + ", ".join(detected_threats)
                        print(f"[Frame {frame_count}] {message}")
                    else:
                        message = "No threats detected."
                        print(f"[Frame {frame_count}] {message}")
                    
                    current_caption = message

                except Exception as e:
                    current_caption = "Analysis failed"
                    continue

finally:
    restore_stderr(old_stderr)
    tello.streamoff()
    cv2.destroyAllWindows()
    print(f"[INFO] Processed {valid_frames_received} valid frames out of {frame_count} total frames")
    print("stop")


# Movement
lengt, width = 8, 10
area = [[{'x': 0, 'y': 0, 'state': "U"} for _ in range(lengt)] for _ in range(width)]

# go thru row
row, col = 0, 0

def spiral_traverse(matrix):
    if not matrix:
        return []

    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Traverse top row (from left to right)
        for col in range(left, right + 1):
            result.append(matrix[top][col])
            matrix[top][col].update({'x':top,'y':col,'state':"S"})
            tello.move_forward(MOVE_INCREMENT_CM)
        top += 1
        tello.rotate_clockwise(90)
        

        # Traverse right column (top to bottom)
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
            matrix[row][right].update({'x':row,'y':right,'state':"S"})
            tello.move_forward(MOVE_INCREMENT_CM)
        right -= 1
        tello.rotate_clockwise(90)
        

        if top <= bottom:
            # Traverse bottom row (right to left)
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
                matrix[bottom][col].update({'x':bottom,'y':col,'state':"S"})
                tello.move_forward(MOVE_INCREMENT_CM)
            bottom -= 1
            tello.rotate_clockwise(90)

        if left <= right:
            # Traverse left column (bottom to top)
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
                matrix[row][left].update({'x':row,'y':left,'state':"S"})
                tello.move_forward(MOVE_INCREMENT_CM)
            left += 1
            tello.rotate_clockwise(90)

    return result

import threading

current_pos = {'x': 0, 'y': 0}  # shared position
area_lock = threading.Lock()
terminate_flag = threading.Event()  # signal to stop threads

def movement_loop():
    """Drone movement pattern with position updates."""
    global current_pos
    spiral_path = spiral_traverse(area)  # Already updates area, now also update current_pos
    for cell in spiral_path:
        with area_lock:
            current_pos['x'] = cell['x']
            current_pos['y'] = cell['y']
        time.sleep(1.5)  # Delay to allow vision system to analyze the frame
        if terminate_flag.is_set():
            break

def vision_loop():
    """Vision model loop with detection updates."""
    global current_caption
    try:
        with torch.no_grad():
            while not terminate_flag.is_set():
                frame = reader.frame
                if frame is None or frame.size == 0:
                    continue

                frame_count += 1
                if frame_count % FRAME_STRIDE != 0:
                    continue

                # Preprocess and generate caption
                image = Image.fromarray(frame)
                inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
                out_ids = model.generate(**inputs, max_new_tokens=48)
                raw_caption = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
                clean_caption_text = clean_caption(raw_caption)
                detected_threats, _ = extract_detected_threats(clean_caption_text, threats_set)

                # Update area with threat info
                with area_lock:
                    x, y = current_pos['x'], current_pos['y']
                    if detected_threats:
                        area[x][y]['state'] = "T"  # Threat detected
                        print(f"[VISION] Threats at ({x},{y}): {detected_threats}")
                    else:
                        area[x][y]['state'] = "C"  # Clear

                time.sleep(0.1)  # Keep loop responsive

    except Exception as e:
        print(f"[ERROR] Vision loop failed: {e}")

vision_thread = threading.Thread(target=vision_loop)
movement_thread = threading.Thread(target=movement_loop)

vision_thread.start()
movement_thread.start()

# Wait for movement to finish
movement_thread.join()

# Signal vision thread to stop
terminate_flag.set()
vision_thread.join()

print("[INFO] Mission complete")
print("[INFO] Final Grid:")
for row in area:
    print([cell['state'] for cell in row])
