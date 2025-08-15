import time, cv2, torch, numpy as np, os, re
from PIL import Image
from djitellopy import Tello
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_DIR = "./models/SmolVLM-500M-Instruct"
security_threats = ["EverWatch", "white_rectangle", "computer_left_on", "cellphone", "phone",
    "lanyard", "notebook", "key", "laptop", "computer", "book", "whiteboard", "monitor",
    "name tag", "id card", "code", "text"]
threats_set = set(t.lower() for t in security_threats)
system_prompt = (
    "You are a physical security inspector analyzing an image for potential threats. "
    f"Identify and describe the following: {', '.join(security_threats)}.\n\n"
    "For each object, describe appearance and location. If none, say 'No threats detected.'"
)
user_prompt = "List and describe the relevant objects in this image based on the instructions."

class DroneMonitor:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available()
                                   else "cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = AutoModelForVision2Seq.from_pretrained(MODEL_DIR, local_files_only=True).to(self.device).eval()

        self.tello = Tello()
        self.tello.connect()
        self.tello.streamon()
        self.reader = self.tello.get_frame_read()

        self.frame = None
        self.caption = "Initializing..."
        self.detected_threats = []

    def get_frame(self):
        return self.reader.frame

    def detect(self):
        frame = self.get_frame()
        if frame is None: return

        image = Image.fromarray(frame)
        prompt = self.processor.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_prompt}]}
        ], add_generation_prompt=True)

        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        output = self.model.generate(**inputs, max_new_tokens=128)
        caption = self.processor.batch_decode(output, skip_special_tokens=True)[0]

        self.caption = caption
        self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Extract threats
        objects = re.split(r'[,\n;]| and |:', caption.lower())
        self.detected_threats = [t for o in objects for t in threats_set if t in o or o in t]

    def shutdown(self):
        self.tello.streamoff()
        self.tello.end()
