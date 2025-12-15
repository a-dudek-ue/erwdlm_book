# --- Import required libraries ---
import random
import cv2                         # OpenCV for image decoding and drawing
import numpy as np                 # NumPy for array operations
import requests                    # For downloading image from the web
from pathlib import Path           # For file path management
from PIL import Image              # For high-resolution image saving
from ultralytics import YOLO       # YOLOv8 model for object (face) detection

# ============================================================
# === 1. DOWNLOAD AN IMAGE FROM WIKIMEDIA COMMONS ============
# ============================================================

# URL of the famous "Lunch atop a Skyscraper" photo
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Lunch_atop_a_Skyscraper.jpg/2560px-Lunch_atop_a_Skyscraper.jpg"

# Add headers to avoid potential request blocking
headers = {
    'User-Agent': 'MyCustomUserAgent/1.0'
}

# Send GET request to download the image
resp = requests.get(url, headers=headers, stream=True)
resp.raise_for_status()            # Ensure the request was successful

# Convert downloaded bytes into an OpenCV image
img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode as color image (BGR)

# ============================================================
# === 2. LOAD YOLOv8 FACE DETECTION MODEL ====================
# ============================================================

# Path to the YOLOv8 face detection model (weights)
# You can download it manually from:
# https://huggingface.co/junjiang/GestureFace/blob/main/yolov8n-face.pt
model_path = Path("yolov8n-face.pt")

# Load the YOLO model
model = YOLO(str(model_path))

# Perform prediction on the loaded image
# - imgsz: image size (resize to 640x640)
# - conf: confidence threshold for detections (0.3 = 30%)
results = model.predict(source=img, imgsz=640, conf=0.3)

# ============================================================
# === 3. DRAW FACE DETECTION RESULTS ON IMAGE ================
# ============================================================

# Iterate over all detection results (usually 1 for a single image)
for result in results:
    # Each detected face is represented as a bounding box (x1, y1, x2, y2)
    for box in result.boxes.xyxy:
        # Expand bounding box slightly by ±10 pixels for better visibility
        x1, y1, x2, y2 = (z - 10 if i < 2 else z + 10 for (i, z) in enumerate(map(int, box)))

        # Draw magenta dotted lines along the top and bottom edges of the box
        for i in range(x1, x2, 10):
            cv2.line(img, (i, y1), (i + 5, y1), (255, 0, 255), 6)
            cv2.line(img, (i, y2), (i + 5, y2), (255, 0, 255), 6)

        # Draw magenta dotted lines along the left and right edges of the box
        for i in range(y1, y2, 10):
            cv2.line(img, (x1, i), (x1, i + 5), (255, 0, 255), 6)
            cv2.line(img, (x2, i), (x2, i + 5), (255, 0, 255), 6)

# ============================================================
# === 4. SAVE AND EXPORT OUTPUT IMAGE ========================
# ============================================================

output_path = "faces_detected.jpg"

# Save the image with magenta dotted face rectangles using OpenCV
cv2.imwrite(output_path, img)

# Reopen and save again using PIL to ensure high DPI (for printing or publishing)
im = Image.open(output_path)
im.save(output_path, dpi=(300, 300))

print("Saved faces_detected.jpg with magenta dotted rectangle face markers!")
