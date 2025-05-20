import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ─── PARAMETERS ───────────────────────────────────────────────────────────────
BLINK_THRESH = 0.015    # how small the lid gap must be to count as a blink
CLICK_DELAY  = 0.5      # seconds to wait after a click
CAM_INDEX    = 0        # which camera to open
# ────────────────────────────────────────────────────────────────────────────────

def get_iris_pos(landmarks, W, H):
    """Return the (x,y) pixel coords of the iris center (avg of 4 iris pts)."""
    ids = [474, 475, 476, 477]
    xs = [landmarks[i].x * W for i in ids]
    ys = [landmarks[i].y * H for i in ids]
    return np.array([np.mean(xs), np.mean(ys)], dtype=float)

def calibrate(cam, face_mesh):
    """Ask user to look at top‑left and bottom‑right corners, pressing 'C' each time."""
    corners = ["top‑left", "bottom‑right"]
    pts = []

    print("CALIBRATION:")
    print("  • Look at the TOP‑LEFT corner of your screen and press 'C'")
    print("  • Then look at the BOTTOM‑RIGHT corner and press 'C'")

    while len(pts) < 2:
        ret, frame = cam.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        # draw prompt
        text = f"Calib {len(pts)+1}/2: look at {corners[len(pts)]} & press 'C'"
        cv2.putText(frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # if we have a face, show the iris point
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            iris_xy = get_iris_pos(lm, W, H)
            cv2.circle(frame, tuple(iris_xy.astype(int)), 5, (0,255,0), -1)

        cv2.imshow("Calibration", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and res.multi_face_landmarks:
            pts.append(iris_xy.copy())
            time.sleep(0.5)    # debounce

    cv2.destroyWindow("Calibration")
    return pts[0], pts[1]   # (top-left), (bottom-right)

def main():
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cam = cv2.VideoCapture(CAM_INDEX)
    if not cam.isOpened():
        print("❌ Could not open camera")
        return

    screen_w, screen_h = pyautogui.size()

    # 1) Calibration
    tl_iris, br_iris = calibrate(cam, face_mesh)

    last_click = 0
    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                iris_xy = get_iris_pos(lm, W, H)

                # normalize into [0..1] using calibration extremes
                nx = (iris_xy[0] - tl_iris[0]) / (br_iris[0] - tl_iris[0])
                ny = (iris_xy[1] - tl_iris[1]) / (br_iris[1] - tl_iris[1])

                # clamp
                nx = np.clip(nx, 0.0, 1.0)
                ny = np.clip(ny, 0.0, 1.0)

                # map to screen
                sx = nx * screen_w
                sy = ny * screen_h
                pyautogui.moveTo(sx, sy)

                # blink? use left eye lids (145 upper, 159 lower)
                top = lm[145]
                bot = lm[159]
                if (top.y - bot.y) < BLINK_THRESH and (time.time() - last_click) > CLICK_DELAY:
                    pyautogui.click()
                    last_click = time.time()

                # debug: show iris
                cv2.circle(frame, tuple(iris_xy.astype(int)), 5, (0,255,0), -1)

            cv2.imshow("Gaze‑Controlled Mouse", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
