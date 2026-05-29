from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "gesture_recognizer.task"

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
GestureRecognizer = vision.GestureRecognizer
GestureRecognizerOptions = vision.GestureRecognizerOptions
VisionRunningMode = vision.RunningMode

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

recognizer = GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak bisa membuka kamera")
    exit()

frame_timestamp = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR -> RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert ke MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp wajib naik terus untuk VIDEO mode
    frame_timestamp += 1

    # Recognize gesture
    result = recognizer.recognize_for_video(
        mp_image,
        frame_timestamp
    )

    # =========================
    # Draw Result
    # =========================

    if result.hand_landmarks:

        h, w, _ = frame.shape

        for hand_idx, hand_landmarks in enumerate(result.hand_landmarks):

            # =========================
            # Draw Landmarks
            # =========================

            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            )

            # =========================
            # Draw Gesture Label
            # =========================

            if result.gestures and len(result.gestures) > hand_idx:

                top_gesture = result.gestures[hand_idx][0]

                gesture_name = top_gesture.category_name
                score = top_gesture.score

                x_text = int(hand_landmarks[0].x * w)
                y_text = int(hand_landmarks[0].y * h) - 20

                cv2.putText(
                    frame,
                    f"{gesture_name} ({score:.2f})",
                    (x_text, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

    # =========================
    # Show Frame
    # =========================

    cv2.imshow("MediaPipe Gesture Recognition", frame)

    # Tekan Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
