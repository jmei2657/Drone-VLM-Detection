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

MODEL_DIR = "./models/SmolVLM-500M-Instruct"
FRAME_STRIDE = 5
MOVE_INCREMENT_CM = 25
MAX_ALTITUDE_CM = 91
HOVER_INCREASE_CM = 5

security_threats = [
    "EVERWATCH",
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
    "You are a security inspector. Identify any of these objects: "
    f"{', '.join(security_threats)}. For each, give:\n"
    "Object: [Name], Location: [e.g., top-left, under desk], Context: [e.g., on table, behind bag].\n"
    "If none found, reply: 'No threats detected.'"
)
user_prompt = "Analyze the image and list relevant objects."

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

def spiral_traverse(tello, reader, processor, model, device, rows=4, cols=4):
    """Perform spiral traversal with threat detection at each position"""
    visited = [[False]*cols for _ in range(rows)]
    dr = [0, 1, 0, -1]  # Direction vectors for spiral
    dc = [1, 0, -1, 0]
    r = c = di = 0
    
    frame_count = 0
    current_caption = "Starting spiral scan..."
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    
    try:
        for step in range(rows * cols):
            print(f"[INFO] Position ({r},{c}) - Step {step + 1}/{rows * cols}")
            
            # Wait for frame and perform detection
            frame = reader.frame
            if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                time.sleep(0.1)
                continue
                
            frame_count += 1
            
            # Convert for display
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Perform threat detection
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
                
                if "User:" in clean_caption_text:
                    caption = clean_caption_text.split("objects.")[-1]
                    caption = caption.strip()

                detected_threats, detected_objects = extract_detected_threats(clean_caption_text, threats_set)
                
                # Show all detected objects for debugging
                if detected_objects:
                    debug_message = f"All objects: {', '.join(detected_objects)}"
                    print(f"[Frame {frame_count}] {debug_message}")
                
                if detected_threats:
                    message = f"Position ({r},{c}) - Threats detected: {', '.join(detected_threats)}"
                    print(f"[Frame {frame_count}] {message}")
                else:
                    message = f"Position ({r},{c}) - No threats detected."
                    print(f"[Frame {frame_count}] {message}")
                
                current_caption = message
                
            except Exception as e:
                current_caption = f"Position ({r},{c}) - Analysis failed"
                print(f"[ERROR] {e}")
            
            # Display with caption
            canvas = compose_canvas(bgr, current_caption, WINDOW_WIDTH, WINDOW_HEIGHT, CAPTION_BAR_PX)
            cv2.imshow(WINDOW_NAME, canvas)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] User requested exit")
                break
            
            # Mark current position as visited
            visited[r][c] = True
            
            # Move to next position in spiral
            if step < rows * cols - 1:  # Don't move after last detection
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
            
    except Exception as e:
        print(f"[ERROR] Error in spiral traversal: {e}")

def main():
    old_stderr = suppress_opencv_errors()
    
    # Device setup
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

    # Tello setup
    tello = Tello()
    tello.connect()
    print(f"[INFO] Battery: {tello.get_battery()}%")
    tello.streamon()
    reader = tello.get_frame_read()

    print("[INFO] Taking off...")
    tello.takeoff()
    time.sleep(2)
    
    # Adjust height
    if tello.get_height() + HOVER_INCREASE_CM <= MAX_ALTITUDE_CM:
        tello.move_up(HOVER_INCREASE_CM)
        print(f"[INFO] Moved up {HOVER_INCREASE_CM} cm")

    print("[INFO] Warming up video stream...")
    time.sleep(3.0)

    try:
        # Start spiral traversal with threat detection
        print("[INFO] Starting spiral scan with threat detection")
        spiral_traverse(tello, reader, processor, model, device, rows=4, cols=4)
        
    finally:
        restore_stderr(old_stderr)
        print("[INFO] Landing...")
        try:
            tello.land()
        except Exception:
            pass
        tello.streamoff()
        cv2.destroyAllWindows()
        print("[INFO] Done.")

if __name__ == "__main__":
    main()
