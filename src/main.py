#!/usr/bin/env python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suppress TF logging

import sys
import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import cv2
import pyautogui
from tensorflow.keras.models import load_model

from utils.camera_utils      import CameraCapture
from utils.mediapipe_utils   import FaceMeshDetector
from utils.metadata_utils    import generate_metadata, extract_metadata_features
from utils.calibration_utils import CalibrationManager
from utils.gaze_processing   import GazeProcessor

# disable PyAutoGUI’s fail‑safe
pyautogui.FAILSAFE = False


class GazeTracker:
    def __init__(self, model_path, cam_index=0, debug=False, calibration_points=9):
        """
        Initialize the gaze tracker.
        model_path:          Path to your .h5 model
        cam_index:           Which webcam to use
        debug:               Whether to show debug windows
        calibration_points:  3, 5 or 9 points for screen calibration
        """
        self.model_path        = model_path
        self.cam_index         = cam_index
        self.debug             = debug
        self.calibration_points= calibration_points

        # screen dims
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Screen: {self.screen_width}×{self.screen_height}")

        # these will be filled in start()
        self.camera        = None
        self.face_detector = None
        self.model         = None

        # calibration manager
        self.calibration = CalibrationManager(
            screen_width = self.screen_width,
            screen_height= self.screen_height,
            points       = self.calibration_points
        )

        # gaze processor (smoothing / blink / fixation detection)
        self.gaze_processor = GazeProcessor(
            smoothing_factor         = 0.7,
            blink_threshold          = 0.18,
            saccade_velocity_thresh  = 50.0,
            fixation_duration_thresh = 0.1
        )

        # tracking flags & stats
        self.running        = False
        self.is_tracking    = False
        self.tracking_paused= False
        self.last_click     = 0
        self.click_delay    = 0.5  # sec

        # hybrid vs iris only vs model only
        self.tracking_mode = "model"  # Changed default from "hybrid" to "model"
        self.blink_click_enabled = False

        # success rates
        self.frames_processed   = 0
        self.model_success_rate = 0
        self.iris_success_rate  = 0

    def extract_face_eyes(self, frame):
        """
        Run MediaPipe face‐mesh and return:
          face_crop, left_eye_crop, right_eye_crop,
          left_eye_landmarks, right_eye_landmarks, face_landmarks
        or (None, …) if no face found.
        """
        try:
            results = self.face_detector.process_frame(frame)
            if (not results or
                not hasattr(results, 'multi_face_landmarks') or
                not results.multi_face_landmarks):
                return (None,)*6

            face_lm = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]

            # build a numpy array of all landmarks to get face bounding box
            pts = np.array([[lm.x * w, lm.y * h] for lm in face_lm.landmark])
            min_x, min_y = pts.min(axis=0).astype(int)
            max_x, max_y = pts.max(axis=0).astype(int)
            padding = 10
            min_x = max(0, min_x - padding)
            min_y = max(0, min_y - padding)
            max_x = min(w, max_x + padding)
            max_y = min(h, max_y + padding)

            # face crop
            face_crop = frame[min_y:max_y, min_x:max_x].copy()

            # eye landmarks (for EAR, debug, etc.)
            left_idxs  = [33,133,160,159,158,157,173,246]
            right_idxs = [362,382,381,380,374,373,390,249]
            left_pts  = np.array([(int(face_lm.landmark[i].x*w),
                                   int(face_lm.landmark[i].y*h))
                                  for i in left_idxs])
            right_pts = np.array([(int(face_lm.landmark[i].x*w),
                                   int(face_lm.landmark[i].y*h))
                                  for i in right_idxs])

            # eye crops with padding
            eye_pad = 15
            lx1, ly1 = left_pts.min(axis=0)  - eye_pad
            lx2, ly2 = left_pts.max(axis=0)  + eye_pad
            rx1, ry1 = right_pts.min(axis=0) - eye_pad
            rx2, ry2 = right_pts.max(axis=0) + eye_pad

            # clamp
            lx1, ly1 = max(0,lx1), max(0,ly1)
            lx2, ly2 = min(w,lx2), min(h,ly2)
            rx1, ry1 = max(0,rx1), max(0,ry1)
            rx2, ry2 = min(w,rx2), min(h,ry2)

            left_eye_crop  = frame[ly1:ly2, lx1:lx2].copy()
            right_eye_crop = frame[ry1:ry2, rx1:rx2].copy()

            return (
                face_crop,
                left_eye_crop, right_eye_crop,
                left_pts.tolist(), right_pts.tolist(),
                face_lm
            )

        except Exception as e:
            print(f"[extract_face_eyes] {e}")
            traceback.print_exc()
            return (None,)*6

    def load_hybrid_model(self):
        """Load the Keras model (no compile)."""
        print(f"Loading model from {self.model_path}…")
        try:
            self.model = load_model(self.model_path, compile=False)
            print("Model loaded.")
            return True
        except Exception as e:
            print(f"❌ could not load model: {e}")
            traceback.print_exc()
            return False

    def preprocess_image(self, img, size=(224,224)):
        """Resize + normalize."""
        if img is None or img.size == 0:
            return np.zeros((size[1],size[0],3), dtype=np.float32)
        x = cv2.resize(img, size)
        return (x.astype('float32') / 255.0)

    def predict_gaze(self, face_img, left_img, right_img, meta):
        """
        Run the model.  Expects your model to have inputs
        named ['face_input','left_eye_input','right_eye_input','metadata_input']
        """
        if self.model is None:
            return None

        # preprocess
        f = np.expand_dims(self.preprocess_image(face_img), axis=0)
        l = np.expand_dims(self.preprocess_image(left_img), axis=0)
        r = np.expand_dims(self.preprocess_image(right_img), axis=0)
        m = np.expand_dims(extract_metadata_features(meta), axis=0)

        # construct dict based on actual model input names
        inp = {}
        for layer in self.model.inputs:
            name = layer.name.split(':')[0]
            if   'face'     in name: inp[layer.name] = f
            elif 'left_eye' in name: inp[layer.name] = l
            elif 'right_eye'in name: inp[layer.name] = r
            elif 'metadata' in name: inp[layer.name] = m

        try:
            pred = self.model.predict(inp, verbose=0)
            gaze = pred[0] if isinstance(pred, np.ndarray) else pred[0][0]
            # clamp [0..1]
            gaze[0] = np.clip(gaze[0], 0.0, 1.0)
            gaze[1] = np.clip(gaze[1], 0.0, 1.0)
            self.model_success_rate += 1
            return gaze
        except Exception as e:
            print(f"[predict_gaze] {e}")
            traceback.print_exc()
            return None

    def run_iris_calibration(self):
        """Calibrate iris‐based tracker by looking at two corners."""
        corners = ["top-left","bottom-right"]
        pts = []
        print("→ Iris calibration: look at each corner and press 'C'")
        while len(pts) < 2:
            ret, frame = self.camera.read()
            if not ret: continue
            frame = cv2.flip(frame,1)
            H, W = frame.shape[:2]
            res = self.face_detector.process_frame(frame)

            msg = f"Calib {len(pts)+1}/2: look at {corners[len(pts)]} & press 'C'"
            cv2.putText(frame, msg, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            if res and res.multi_face_landmarks:
                lm  = res.multi_face_landmarks[0].landmark
                cen = self.gaze_processor.get_iris_centers(lm, frame.shape)
                if cen is not None:
                    cv2.circle(frame, tuple(cen.astype(int)), 5, (0,255,0), -1)

            cv2.imshow("Iris Calibration", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and res and res.multi_face_landmarks:
                pts.append(cen.copy())
                time.sleep(0.5)
            elif key == 27:
                break

        cv2.destroyWindow("Iris Calibration")
        if len(pts) == 2:
            self.iris_tl_calib, self.iris_br_calib = pts
            self.use_iris_tracking = True
            print("Iris calibration done.")
        else:
            print("Iris calibration aborted, fallback to hybrid.")

    def update_cursor(self, gaze_pt):
        """Map a normalized [0..1] gaze_pt to screen and move mouse."""
        if gaze_pt is None or not self.is_tracking or self.tracking_paused:
            return
        # smooth
        sg = self.gaze_processor.smooth_gaze(gaze_pt)

        # only move during fixations
        if (self.gaze_processor.current_event == "FIXATION"
                and not self.gaze_processor.is_blinking):
            sx = int(sg[0] * self.screen_width)
            sy = int(sg[1] * self.screen_height)
            pyautogui.moveTo(sx, sy, duration=0.01)

        # blink→click?
        now = time.time()
        if (self.blink_click_enabled
            and self.gaze_processor.is_blinking
            and (now - self.last_click) > self.click_delay):
            pyautogui.click()
            self.last_click = now

    def process_frame(self, frame):
        """
        Extract crops, run model/iris as per mode,
        then package into a single dict for visualization & cursor update.
        """
        face_crop, le, re, le_lm, re_lm, fm = self.extract_face_eyes(frame)
        if face_crop is None:
            return None

        meta = generate_metadata(
            frame.shape, fm,
            left_eye_data = None,
            right_eye_data= None,
            screen_width  = self.screen_width,
            screen_height = self.screen_height,
            frame_time    = time.time()
        )

        gaze_results = {
            "model":  None,
            "iris":   None,
            "selected": None,
            "method_used": "none"
        }

        # model
        if self.model and self.tracking_mode in ("hybrid","model"):
            gm = self.predict_gaze(face_crop, le, re, meta)
            gaze_results["model"] = gm

        # iris
        if getattr(self, "use_iris_tracking", False) and self.tracking_mode in ("hybrid","iris"):
            gi = self.gaze_processor.iris_based_tracking(
                fm, frame.shape,
                tl=self.iris_tl_calib, br=self.iris_br_calib
            )
            gaze_results["iris"] = gi
            self.iris_success_rate += 1

        # select
        if self.tracking_mode == "model":
            gaze_results["selected"] = gaze_results["model"]
            gaze_results["method_used"] = "model"
        elif self.tracking_mode == "iris":
            gaze_results["selected"] = gaze_results["iris"]
            gaze_results["method_used"] = "iris"
        else:  # hybrid
            if gaze_results["model"] is not None:
                gaze_results["selected"] = gaze_results["model"]
                gaze_results["method_used"] = "model"
            else:
                gaze_results["selected"] = gaze_results["iris"]
                gaze_results["method_used"] = "iris"

        # feed into comprehensive gaze processor
        return self.gaze_processor.process_gaze_data(
            left_eye_img=le, right_eye_img=re,
            left_eye_landmarks=le_lm, right_eye_landmarks=re_lm,
            face_landmarks=fm,
            frame_shape=frame.shape,
            predicted_gaze=gaze_results["selected"]
        )

    def draw_normal_view(self, frame, gr):
        """A simple debug overlay."""
        out = frame.copy()
        h, w = out.shape[:2]

        if gr.get("smoothed_gaze_point") is not None:
            x = int(gr["smoothed_gaze_point"][0]*w)
            y = int(gr["smoothed_gaze_point"][1]*h)
            c = (0,255,0)
            cv2.line(out, (x-10,y),(x+10,y),c,2)
            cv2.line(out, (x,y-10),(x,y+10),c,2)

        status = gr.get("gaze_event","?")
        color = (0,255,0) if status=="FIXATION" else (0,0,255)
        cv2.putText(out, f"Event: {status}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(out, f"Mode: {gr.get('method_used','-')}", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        return out

    def start(self):
        """Load everything and enter main loop."""
        if not self.load_hybrid_model():
            return False

        # debug window
        if self.debug:
            cv2.namedWindow("Gaze Tracker", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Gaze Tracker", 800, 600)

        # camera
        print("Starting camera…")
        self.camera = CameraCapture(self.cam_index, width=640, height=480)
        self.camera.start()

        # face detector
        print("Starting face‐mesh detector…")
        self.face_detector = FaceMeshDetector(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5
        )

        # Skip iris calibration by default as we're using model mode
        # Only run iris calibration if explicitly in hybrid or iris mode
        if self.tracking_mode in ("hybrid", "iris"):
            self.run_iris_calibration()
        else:
            print("Using model-based tracking (skipping iris calibration)")

        print("Entering main loop…")
        self.running = True
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame = cv2.flip(frame,1)
                gr    = self.process_frame(frame)

                if not self.calibration.calibration_complete:
                    frame = self.calibration.draw_calibration_target(frame,
                                frame.shape[1], frame.shape[0])
                else:
                    if not self.is_tracking:
                        self.is_tracking = True
                        print("Calibration done → tracking!")
                    self.update_cursor(gr)

                if self.debug:
                    disp = self.draw_normal_view(frame, gr or {})
                    cv2.imshow("Gaze Tracker", disp)

                key = cv2.waitKey(1) & 0xFF
                if   key == 27:              # ESC
                    self.running = False
                elif key == ord(' '):        # SPACE
                    if not self.calibration.calibration_complete:
                        if gr and gr.get("gaze_point") is not None:
                            self.calibration.collect_data_point(gr["gaze_point"])
                    else:
                        self.tracking_paused = not self.tracking_paused
                elif key == ord('c'):        # C to recalibrate
                    self.calibration = CalibrationManager(
                        self.screen_width,
                        self.screen_height,
                        self.calibration_points
                    )
                    self.is_tracking = False
                elif key == ord('d'):        # toggle debug
                    self.debug = not self.debug
                elif key == ord('m'):        # toggle mode
                    modes = ["hybrid","model","iris"]
                    idx   = modes.index(self.tracking_mode)
                    self.tracking_mode = modes[(idx+1)%3]
                    print("Mode →", self.tracking_mode)
                elif key == ord('b'):        # blink click toggle
                    self.blink_click_enabled = not self.blink_click_enabled
                    print("Blink click →", self.blink_click_enabled)
                elif key == ord('i'):        # rerun iris calib
                    self.run_iris_calibration()

                self.frames_processed += 1

        except Exception as e:
            print(f"[main loop] {e}")
            traceback.print_exc()
        finally:
            self.cleanup()
        return True

    def cleanup(self):
        """Release everything."""
        print("Cleaning up…")
        try:
            if self.camera:
                self.camera.stop()
            if self.face_detector:
                self.face_detector.close()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"[cleanup] {e}")
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser("Eye Gaze Tracker")
    parser.add_argument('--model_path',       required=True, help="Path to .h5 model")
    parser.add_argument('--cam_index',   type=int, default=0, help="Camera index")
    parser.add_argument('--debug',      action='store_true', help="Show debug UI")
    parser.add_argument('--calibration_points', type=int, default=9,
                        choices=[3,5,9], help="3, 5 or 9 screen points")
    parser.add_argument('--mode', default="model", choices=["hybrid", "model", "iris"], 
                        help="Tracking mode: model (default), hybrid, or iris")
    args = parser.parse_args()

    mp = Path(args.model_path)
    if not mp.exists():
        print(f"❌ model not found: {mp}")
        sys.exit(1)

    tracker = GazeTracker(
        model_path       = str(mp),
        cam_index        = args.cam_index,
        debug            = args.debug,
        calibration_points = args.calibration_points
    )
    
    # Set tracking mode from command line argument
    tracker.tracking_mode = args.mode

    print("\n--- Eye Gaze Tracker ---")
    print(f"Tracking mode: {tracker.tracking_mode.upper()}")
    print("1. SPACE to collect calibration points")
    print("2. After that, gaze controls cursor")
    print("3. ESC to quit, C to recalibrate, D to toggle debug")
    print("4. M to switch modes (current: model), B to enable blink-click\n")
    print("Starting…\n")

    if not tracker.start():
        print("Tracker failed to start.")
        sys.exit(1)


if __name__ == "__main__":
    main()
