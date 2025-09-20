import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from filterpy.kalman import KalmanFilter
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Get screen size
screen_w, screen_h = pyautogui.size()
print(f"Screen resolution: {screen_w}x{screen_h}")

# Optimized Kalman Filter for nose tracking
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])  # State transition
kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Measurement function
kf.P *= 300  # Initial uncertainty
kf.R = np.array([[0.3, 0], [0, 0.3]])  # Measurement noise (lower = more responsive)
kf.Q = np.eye(4) * 0.003  # Process noise (lower = smoother)

# Initialize Kalman filter state
kf_initialized = False

# Optimized webcam settings
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Calibration variables for smaller movement area
calibration_frames = 0
max_calibration_frames = 60
nose_positions = []
center_x, center_y = 0, 0
movement_scale = 2.5  # Amplification factor for small movements

# Nose tip landmark index in MediaPipe Face Mesh
NOSE_TIP = 1

# Eye landmarks for blink detection (simplified and more reliable)
LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Key eye landmarks for EAR calculation
LEFT_EYE_CORNERS = [33, 133]  # Left and right corners
LEFT_EYE_VERTICAL = [159, 145]  # Top and bottom points
RIGHT_EYE_CORNERS = [362, 263]  # Left and right corners  
RIGHT_EYE_VERTICAL = [386, 374]  # Top and bottom points

# Blink detection variables
blink_threshold = 0.21  # EAR threshold for blink detection (lowered for better sensitivity)
consecutive_blinks = 0
min_blink_frames = 3  # Minimum consecutive frames for a valid blink
max_blink_frames = 15  # Maximum frames to prevent long blinks from multiple clicks
last_click_time = 0
click_cooldown = 0.8  # Minimum time between clicks (seconds)
ear_buffer = []
buffer_size = 3

def calculate_ear(eye_corners, eye_vertical, face_landmarks, w, h):
    """Calculate Eye Aspect Ratio (EAR) for blink detection"""
    try:
        # Get corner points (horizontal distance)
        left_corner = face_landmarks.landmark[eye_corners[0]]
        right_corner = face_landmarks.landmark[eye_corners[1]]
        
        # Get vertical points
        top_point = face_landmarks.landmark[eye_vertical[0]]
        bottom_point = face_landmarks.landmark[eye_vertical[1]]
        
        # Convert to pixel coordinates
        left_x, left_y = left_corner.x * w, left_corner.y * h
        right_x, right_y = right_corner.x * w, right_corner.y * h
        top_x, top_y = top_point.x * w, top_point.y * h
        bottom_x, bottom_y = bottom_point.x * w, bottom_point.y * h
        
        # Calculate distances
        horizontal_dist = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
        vertical_dist = np.sqrt((top_x - bottom_x)**2 + (top_y - bottom_y)**2)
        
        # Calculate EAR
        if horizontal_dist > 0:
            ear = vertical_dist / horizontal_dist
        else:
            ear = 0
            
        return ear
    except:
        return 0.3  # Default EAR value if calculation fails

def is_blink_detected(left_ear, right_ear):
    """Detect blink based on both eyes' EAR with improved logic"""
    # Use the lower EAR (more closed eye)
    min_ear = min(left_ear, right_ear)
    
    # Add to buffer for smoothing
    ear_buffer.append(min_ear)
    if len(ear_buffer) > buffer_size:
        ear_buffer.pop(0)
    
    # Use smoothed EAR
    smooth_ear = np.mean(ear_buffer) if ear_buffer else min_ear
    
    return smooth_ear < blink_threshold

print("Starting enhanced nose tracking cursor control with blink clicking...")
print("Features:")
print("- Move cursor: Move your nose")
print("- Left click: Blink your eyes")
print("Look straight ahead for the first 2 seconds for calibration")
print("Press 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)  # Mirror effect
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            # Get nose tip coordinates
            nose_tip = face_landmarks.landmark[NOSE_TIP]
            nose_x = nose_tip.x * w
            nose_y = nose_tip.y * h

            # Calculate EAR for both eyes using improved method
            left_ear = calculate_ear(LEFT_EYE_CORNERS, LEFT_EYE_VERTICAL, face_landmarks, w, h)
            right_ear = calculate_ear(RIGHT_EYE_CORNERS, RIGHT_EYE_VERTICAL, face_landmarks, w, h)
            avg_ear = (left_ear + right_ear) / 2.0

            # Improved blink detection logic
            current_time = time.time()
            if is_blink_detected(left_ear, right_ear):
                consecutive_blinks += 1
            else:
                # Check if we had a valid blink sequence
                if min_blink_frames <= consecutive_blinks <= max_blink_frames:
                    # Perform click if cooldown period has passed
                    if current_time - last_click_time > click_cooldown:
                        try:
                            pyautogui.click()
                            last_click_time = current_time
                            print(f"✓ CLICK! Blink duration: {consecutive_blinks} frames, EAR: {avg_ear:.3f}")
                        except Exception as e:
                            print(f"Click failed: {e}")
                
                consecutive_blinks = 0

            # Calibration phase - establish center point
            if calibration_frames < max_calibration_frames:
                nose_positions.append((nose_x, nose_y))
                calibration_frames += 1
                
                # Draw calibration indicator
                cv2.putText(frame, f"Calibrating... {calibration_frames}/{max_calibration_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.circle(frame, (int(nose_x), int(nose_y)), 5, (0, 255, 255), -1)
                
                if calibration_frames == max_calibration_frames:
                    # Calculate center position
                    center_x = np.mean([pos[0] for pos in nose_positions])
                    center_y = np.mean([pos[1] for pos in nose_positions])
                    print(f"Calibration complete! Center: ({center_x:.1f}, {center_y:.1f})")
                    
            else:
                # Calculate relative movement from center
                rel_x = (nose_x - center_x) * movement_scale
                rel_y = (nose_y - center_y) * movement_scale
                
                # Map to screen coordinates (smaller movement area)
                # Limit the movement range to prevent extreme cursor positions
                max_movement = min(w, h) * 0.3  # 30% of frame size
                rel_x = np.clip(rel_x, -max_movement, max_movement)
                rel_y = np.clip(rel_y, -max_movement, max_movement)
                
                # Convert to screen coordinates
                mapped_x = screen_w/2 + (rel_x / max_movement) * (screen_w/2)
                mapped_y = screen_h/2 + (rel_y / max_movement) * (screen_h/2)
                
                # Ensure cursor stays within screen bounds
                mapped_x = np.clip(mapped_x, 0, screen_w)
                mapped_y = np.clip(mapped_y, 0, screen_h)

                # Initialize or update Kalman filter
                if not kf_initialized:
                    kf.x = np.array([mapped_x, mapped_y, 0, 0])
                    kf_initialized = True
                    smoothed_x, smoothed_y = mapped_x, mapped_y
                else:
                    kf.predict()
                    kf.update([mapped_x, mapped_y])
                    
                    # Get smoothed coordinates
                    smoothed_x, smoothed_y = kf.x[:2]
                
                # Move cursor
                pyautogui.moveTo(smoothed_x, smoothed_y, _pause=False)

                # Visual feedback
                cv2.circle(frame, (int(nose_x), int(nose_y)), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(center_x), int(center_y)), 3, (255, 0, 0), -1)
                
                # Draw movement area indicator
                area_size = int(max_movement)
                cv2.rectangle(frame, 
                            (int(center_x - area_size), int(center_y - area_size)),
                            (int(center_x + area_size), int(center_y + area_size)),
                            (255, 255, 0), 2)
                
                # Draw eye corner landmarks for visual feedback (more focused)
                # Left eye corners
                for idx in LEFT_EYE_CORNERS + LEFT_EYE_VERTICAL:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                # Right eye corners  
                for idx in RIGHT_EYE_CORNERS + RIGHT_EYE_VERTICAL:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
                
                # Show movement and blink info
                cv2.putText(frame, f"Movement: ({rel_x:.1f}, {rel_y:.1f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Cursor: ({smoothed_x:.0f}, {smoothed_y:.0f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"L_EAR: {left_ear:.3f} | R_EAR: {right_ear:.3f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Enhanced blink status indicator
                if consecutive_blinks > 0:
                    blink_status = f"BLINKING ({consecutive_blinks}/{min_blink_frames})"
                    blink_color = (0, 165, 255)  # Orange when detecting
                else:
                    blink_status = f"EYES OPEN (Threshold: {blink_threshold})"
                    blink_color = (0, 255, 0)  # Green when open
                    
                cv2.putText(frame, blink_status, 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, blink_color, 2)
                
                # Click cooldown indicator
                time_since_click = current_time - last_click_time
                if time_since_click < click_cooldown:
                    cv2.putText(frame, f"Click cooldown: {click_cooldown - time_since_click:.1f}s", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # Instructions
    cv2.putText(frame, "Nose: Move cursor | Blink: Left click | 'q': Quit", 
               (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("Enhanced Nose Tracking with Blink Control", frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Enhanced nose tracking cursor control stopped.")