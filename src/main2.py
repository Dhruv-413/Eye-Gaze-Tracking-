#!/usr/bin/env python3
# src/main2.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from tensorflow.keras.models import load_model

# ───────────────────────────────────────────────────────────────────────────────
# Argument parsing
p = argparse.ArgumentParser(description="Real‑time gaze → cursor")
p.add_argument('--model_path', required=True,
               help="Path to your .h5 gaze model")
p.add_argument('--cam_index', type=int, default=0,
               help="Webcam index (default=0)")
args = p.parse_args()

# ───────────────────────────────────────────────────────────────────────────────
# Load model
model_path = Path(args.model_path).resolve()
if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    exit(1)

print(f"Loading model from {model_path}")
model = load_model(str(model_path), compile=False)
print("✅ Model loaded")

# ───────────────────────────────────────────────────────────────────────────────
# MediaPipe FaceMesh setup
mpf       = mp.solutions.face_mesh
face_mesh = mpf.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ───────────────────────────────────────────────────────────────────────────────
# Eye‑crop helper (left eye)
LEFT_EYE_IDX = [33,133,160,159,158,157,173,246]
def get_eye_crop(frame, lm, idxs, pad=10):
    h, w, _ = frame.shape
    xs = [lm.landmark[i].x * w for i in idxs]
    ys = [lm.landmark[i].y * h for i in idxs]
    x1, x2 = int(min(xs)) - pad, int(max(xs)) + pad
    y1, y2 = int(min(ys)) - pad, int(max(ys)) + pad
    return frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]

scr_w, scr_h = pyautogui.size()
print(f"Screen: {scr_w}×{scr_h}")
print("Starting real‑time gaze tracking. Press ESC to quit.")

cap = cv2.VideoCapture(args.cam_index)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res   = face_mesh.process(rgb)

    if res.multi_face_landmarks:
        lm  = res.multi_face_landmarks[0]
        eye = get_eye_crop(frame, lm, LEFT_EYE_IDX)

        if eye.size > 0:
            eye_in = cv2.resize(eye, (224,224)).astype("float32")/255.0
            raw    = model.predict(np.expand_dims(eye_in,0), verbose=0)[0]

            x_pix = int(np.clip(raw[0] * scr_w, 0, scr_w-1))
            y_pix = int(np.clip(raw[1] * scr_h, 0, scr_h-1))
            pyautogui.moveTo(x_pix, y_pix, duration=0.01)

    cv2.imshow("Gaze Cursor", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
