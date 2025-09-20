import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from filterpy.kalman import KalmanFilter

# Initialize MediaPipe Hands (Hand tracking model)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,  # Only track the right hand
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)

# Get screen size
screen_w, screen_h = pyautogui.size()

# Optimized Kalman Filter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
kf.P *= 500  # Lower initial uncertainty
kf.R = np.array([[0.5, 0], [0, 0.5]])  # Lower measurement noise
kf.Q = np.eye(4) * 0.005  # Smoother motion

# Optimized webcam settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 60)  # Max out FPS to match display refresh rate
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def fingers_touching(finger1, finger2, threshold=0.04):
    """Check if two fingers are touching."""
    return np.linalg.norm(np.array([finger1.x, finger1.y]) - np.array([finger2.x, finger2.y])) < threshold

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)  # Mirror effect
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            middle_tip = hand_landmarks.landmark[12]
            index_tip = hand_landmarks.landmark[8]
            thumb_tip = hand_landmarks.landmark[4]

            # Get middle finger tip coordinates
            middle_x = middle_tip.x * w
            middle_y = middle_tip.y * h

            # Normalize screen coordinates
            mapped_x = np.interp(middle_x, (0, w), (0, screen_w))
            mapped_y = np.interp(middle_y, (0, h), (0, screen_h))

            # Apply Kalman filter
            kf.predict()
            kf.update([mapped_x, mapped_y])
            smoothed_x, smoothed_y = kf.x[:2]

            # Move cursor instantly with GPU acceleration
            pyautogui.moveTo(smoothed_x, smoothed_y, _pause=False)

            # Left Click Detection (Index Finger & Thumb Touch)
            if fingers_touching(index_tip, thumb_tip):
                pyautogui.click()

    cv2.imshow("Hand Tracking Mouse (GPU-Optimized)", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




