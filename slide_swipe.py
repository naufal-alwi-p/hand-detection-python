from pathlib import Path
import math
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import pyautogui

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

if not MODEL_DIR.is_dir():
    raise NotADirectoryError(f"Folder {MODEL_DIR} tidak ditemukan")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"{MODEL_PATH} tidak ada")

# =========================
# MediaPipe Drawing Utils
# =========================

mp_hands = vision.HandLandmarksConnections
mp_drawing = vision.drawing_utils
mp_drawing_styles = vision.drawing_styles

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = HandLandmarker.create_from_options(options)

# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Tidak bisa membuka kamera")

# =========================================================
# CONFIG
# =========================================================

previous_hand_x = 0

swipe_threshold = 30
pinch_threshold = 25

cooldown_time = 1.0
last_action_time = 0

presentation_active = False
action_status = "None"

timestamp = 0
pTime = 0

# =========================================================
# MAIN LOOP
# =========================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Gagal membaca frame")
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert ke MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp += 1

    # =====================================================
    # DETECT HAND
    # =====================================================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp
    )

    current_time = time.time()

    action_ready = (
        current_time - last_action_time
    ) > cooldown_time

    # =====================================================
    # HAND DETECTED
    # =====================================================

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        h, w, _ = frame.shape

        # =================================================
        # DRAW LANDMARKS
        # =================================================

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=hand_landmarks,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=(
                mp_drawing_styles
                .get_default_hand_landmarks_style()
            ),
            connection_drawing_spec=(
                mp_drawing_styles
                .get_default_hand_connections_style()
            )
        )

        # =================================================
        # GET IMPORTANT LANDMARKS
        # =================================================

        wrist = hand_landmarks[0]

        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]

        # =================================================
        # PIXEL COORDINATES
        # =================================================

        hand_x = int(wrist.x * w)

        x1 = int(thumb_tip.x * w)
        y1 = int(thumb_tip.y * h)

        x2 = int(index_tip.x * w)
        y2 = int(index_tip.y * h)

        # =================================================
        # DRAW PINCH VISUALIZATION
        # =================================================

        cv2.circle(
            frame,
            (x1, y1),
            10,
            (255, 0, 255),
            cv2.FILLED
        )

        cv2.circle(
            frame,
            (x2, y2),
            10,
            (255, 0, 255),
            cv2.FILLED
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 255),
            2
        )

        # =================================================
        # PINCH DISTANCE
        # =================================================

        pinch_distance = math.hypot(
            x2 - x1,
            y2 - y1
        )

        # =================================================
        # START / END PRESENTATION
        # =================================================

        if (
            pinch_distance < pinch_threshold
            and action_ready
        ):

            if not presentation_active:

                pyautogui.press("f5")

                presentation_active = True
                action_status = (
                    "Start Presentation"
                )

            else:

                pyautogui.press("esc")

                presentation_active = False
                action_status = (
                    "End Presentation"
                )

            last_action_time = current_time

        # =================================================
        # SWIPE DETECTION
        # =================================================

        if (
            previous_hand_x != 0
            and action_ready
        ):

            x_movement = (
                hand_x - previous_hand_x
            )

            # Swipe kiri
            if x_movement < -swipe_threshold:

                pyautogui.press("left")

                action_status = "Previous Slide"

                last_action_time = current_time

            # Swipe kanan
            elif x_movement > swipe_threshold:

                pyautogui.press("right")

                action_status = (
                    "Next Slide"
                )

                last_action_time = current_time

        previous_hand_x = hand_x

    # =====================================================
    # UI
    # =====================================================

    status_text = (
        "Active"
        if presentation_active
        else "Inactive"
    )

    cv2.putText(
        frame,
        f"Presentation: {status_text}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        f"Action: {action_status}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "Pinch = Start / End",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "Swipe = Change Slide",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # =====================================================
    # FPS
    # =====================================================

    cTime = time.time()

    fps = (
        1 / (cTime - pTime)
        if cTime != pTime
        else 0
    )

    pTime = cTime

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(
        "MediaPipe Hand Presentation Controller",
        frame
    )

    # =====================================================
    # EXIT
    # =====================================================

    if (
        cv2.waitKey(1) & 0xFF
        == ord("q")
    ):
        break

# =========================================================
# CLEANUP
# =========================================================

cap.release()
cv2.destroyAllWindows()
