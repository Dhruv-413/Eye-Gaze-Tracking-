import cv2
import numpy as np

def preprocess_image(image, target_size=(64, 64)):
    """Preprocess an image by resizing and normalizing."""
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    normalized = resized / 255.0
    return normalized

def draw_rectangle(image, bbox, color=(0, 255, 0), thickness=2):
    """Draw a rectangle on an image."""
    x, y, w, h = bbox
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

def draw_text(image, text, position, color=(255, 255, 255), font_scale=0.5, thickness=1):
    """Draw text on an image."""
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
