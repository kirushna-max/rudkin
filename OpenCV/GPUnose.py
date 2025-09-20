import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from filterpy.kalman import KalmanFilter
import time
import speech_recognition as sr
import pyaudio
import threading
import wave
import tempfile
import os
from collections import deque

class EnhancedFaceController:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Screen settings
        self.screen_w, self.screen_h = pyautogui.size()
        print(f"Screen resolution: {self.screen_w}x{self.screen_h}")
        
        # Enhanced Kalman Filter for nose tracking
        self.setup_kalman_filter()
        
        # Moving average filter for additional smoothing
        self.position_history = deque(maxlen=8)  # Keep last 8 positions
        self.velocity_history = deque(maxlen=5)  # Track velocity for prediction
        
        # Webcam settings
        self.setup_camera()
        
        # Calibration variables
        self.calibration_frames = 0
        self.max_calibration_frames = 60
        self.nose_positions = []
        self.center_x, self.center_y = 0, 0
        self.movement_scale = 1.8  # Reduced for smoother control
        
        # Landmark indices
        self.NOSE_TIP = 1
        self.LEFT_EYE_CORNERS = [33, 133]
        self.LEFT_EYE_VERTICAL = [159, 145]
        self.RIGHT_EYE_CORNERS = [362, 263]
        self.RIGHT_EYE_VERTICAL = [386, 374]
        
        # Mouth landmarks for MAR (Mouth Aspect Ratio) calculation
        # Using more reliable mouth landmarks
        self.MOUTH_OUTER = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        # Key mouth points for aspect ratio - using inner lip for better detection
        self.MOUTH_HORIZONTAL = [61, 291]  # Left and right corners
        self.MOUTH_VERTICAL = [13, 14, 269, 270]  # Multiple vertical points for accuracy
        
        # Blink detection variables - ADJUSTED FOR EASIER DETECTION
        self.blink_threshold = 0.25  # INCREASED from 0.21 to 0.25 for easier blink detection
        self.consecutive_blinks = 0
        self.min_blink_frames = 2  # REDUCED from 3 to 2 for faster response
        self.max_blink_frames = 15
        self.last_click_time = 0
        self.click_cooldown = 0.8
        self.ear_buffer = deque(maxlen=5)  # Using deque for efficiency
        
        # IMPROVED Mouth detection variables - ADJUSTED FOR MORE SENSITIVE DETECTION
        self.mouth_threshold = 0.25  # LOWERED from 0.35 to 0.25 for easier mouth detection
        self.mouth_open_frames = 0
        self.mouth_closed_frames = 0
        self.min_mouth_open_frames = 2  # REDUCED from 3 to 2 for faster response
        self.recording_stop_delay = 60  # 2 seconds at 30fps (was 30)
        self.mar_buffer = deque(maxlen=8)  # Larger buffer for more stable detection
        
        # Additional mouth detection improvements
        self.mouth_state_history = deque(maxlen=10)  # Track mouth state over time
        self.mouth_confidence_threshold = 0.5  # LOWERED from 0.6 to 0.5 for easier detection
        
        # Speech recognition setup
        self.setup_speech_recognition()
        
        # Recording state
        self.is_recording = False
        self.recording_start_time = 0
        self.audio_data = []
        self.recording_thread = None
        
        # Audio settings
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.audio = pyaudio.PyAudio()
        
        # Smoothing parameters
        self.smoothing_factor = 0.7  # For exponential smoothing
        self.last_smooth_x = 0
        self.last_smooth_y = 0
        
    def setup_kalman_filter(self):
        """Initialize enhanced Kalman filter for smoother cursor movement"""
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        # State transition model (position and velocity)
        self.kf.F = np.array([[1, 0, 1, 0], 
                              [0, 1, 0, 1], 
                              [0, 0, 0.95, 0],  # Velocity decay factor
                              [0, 0, 0, 0.95]])
        # Measurement function
        self.kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        # Covariance matrices - tuned for smoother movement
        self.kf.P *= 100  # Reduced initial uncertainty
        self.kf.R = np.array([[0.1, 0], [0, 0.1]])  # Lower measurement noise
        self.kf.Q = np.eye(4) * 0.001  # Lower process noise
        self.kf_initialized = False
        
    def setup_camera(self):
        """Initialize camera with optimized settings"""
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Reduced from 60 for stability
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def setup_speech_recognition(self):
        """Initialize speech recognition"""
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Calibrate microphone
        print("Calibrating microphone for ambient noise...")
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            print("Microphone calibration complete!")
        except Exception as e:
            print(f"Microphone calibration failed: {e}")
    
    def calculate_ear(self, eye_corners, eye_vertical, face_landmarks, w, h):
        """Calculate Eye Aspect Ratio for blink detection"""
        try:
            left_corner = face_landmarks.landmark[eye_corners[0]]
            right_corner = face_landmarks.landmark[eye_corners[1]]
            top_point = face_landmarks.landmark[eye_vertical[0]]
            bottom_point = face_landmarks.landmark[eye_vertical[1]]
            
            left_x, left_y = left_corner.x * w, left_corner.y * h
            right_x, right_y = right_corner.x * w, right_corner.y * h
            top_x, top_y = top_point.x * w, top_point.y * h
            bottom_x, bottom_y = bottom_point.x * w, bottom_point.y * h
            
            horizontal_dist = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
            vertical_dist = np.sqrt((top_x - bottom_x)**2 + (top_y - bottom_y)**2)
            
            if horizontal_dist > 0:
                ear = vertical_dist / horizontal_dist
            else:
                ear = 0
                
            return ear
        except:
            return 0.3
    
    def calculate_mar(self, face_landmarks, w, h):
        """IMPROVED Mouth Aspect Ratio calculation with multiple measurement points"""
        try:
            # Get mouth corner points (horizontal distance)
            left_corner = face_landmarks.landmark[self.MOUTH_HORIZONTAL[0]]
            right_corner = face_landmarks.landmark[self.MOUTH_HORIZONTAL[1]]
            
            # Get multiple vertical measurement points for more accuracy
            vertical_distances = []
            for i in range(0, len(self.MOUTH_VERTICAL), 2):
                if i + 1 < len(self.MOUTH_VERTICAL):
                    top_point = face_landmarks.landmark[self.MOUTH_VERTICAL[i]]
                    bottom_point = face_landmarks.landmark[self.MOUTH_VERTICAL[i + 1]]
                    
                    top_x, top_y = top_point.x * w, top_point.y * h
                    bottom_x, bottom_y = bottom_point.x * w, bottom_point.y * h
                    
                    vertical_dist = np.sqrt((top_x - bottom_x)**2 + (top_y - bottom_y)**2)
                    vertical_distances.append(vertical_dist)
            
            # Calculate horizontal distance
            left_x, left_y = left_corner.x * w, left_corner.y * h
            right_x, right_y = right_corner.x * w, right_corner.y * h
            horizontal_dist = np.sqrt((right_x - left_x)**2 + (right_y - left_y)**2)
            
            # Average vertical distance
            avg_vertical_dist = np.mean(vertical_distances) if vertical_distances else 0
            
            # Calculate MAR
            if horizontal_dist > 0:
                mar = avg_vertical_dist / horizontal_dist
            else:
                mar = 0
                
            return mar
        except Exception as e:
            return 0.3
    
    def is_blink_detected(self, left_ear, right_ear):
        """Detect blink based on EAR"""
        min_ear = min(left_ear, right_ear)
        
        self.ear_buffer.append(min_ear)
        smooth_ear = np.mean(self.ear_buffer) if self.ear_buffer else min_ear
        return smooth_ear < self.blink_threshold
    
    def is_mouth_open(self, mar):
        """IMPROVED mouth detection with confidence-based filtering"""
        # Add current MAR to buffer
        self.mar_buffer.append(mar)
        
        # Calculate smoothed MAR
        smooth_mar = np.mean(self.mar_buffer) if self.mar_buffer else mar
        
        # Determine if mouth is open based on threshold
        is_open = smooth_mar > self.mouth_threshold
        
        # Add to state history for confidence calculation
        self.mouth_state_history.append(is_open)
        
        # Calculate confidence: percentage of recent frames that agree
        if len(self.mouth_state_history) >= 3:  # Need at least 3 frames
            open_count = sum(self.mouth_state_history)
            confidence = open_count / len(self.mouth_state_history)
            
            # Return True if confidence is high enough
            return confidence > self.mouth_confidence_threshold
        
        return is_open
    
    def apply_smoothing_filter(self, raw_x, raw_y):
        """Apply multiple layers of smoothing to reduce jitter"""
        # Add to position history
        self.position_history.append((raw_x, raw_y))
        
        if len(self.position_history) < 3:
            return raw_x, raw_y
        
        # 1. Moving average filter
        recent_positions = list(self.position_history)[-5:]  # Last 5 positions
        avg_x = np.mean([pos[0] for pos in recent_positions])
        avg_y = np.mean([pos[1] for pos in recent_positions])
        
        # 2. Exponential smoothing
        if self.last_smooth_x == 0 and self.last_smooth_y == 0:
            self.last_smooth_x, self.last_smooth_y = avg_x, avg_y
        
        smooth_x = self.smoothing_factor * avg_x + (1 - self.smoothing_factor) * self.last_smooth_x
        smooth_y = self.smoothing_factor * avg_y + (1 - self.smoothing_factor) * self.last_smooth_y
        
        self.last_smooth_x, self.last_smooth_y = smooth_x, smooth_y
        
        return smooth_x, smooth_y
    
    def start_recording(self):
        """Start recording audio"""
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = time.time()
            self.audio_data = []
            self.recording_thread = threading.Thread(target=self._record_audio)
            self.recording_thread.start()
            print("🎤 Recording started (mouth opened)...")
    
    def stop_recording(self):
        """Stop recording and process audio"""
        if self.is_recording:
            self.is_recording = False
            recording_duration = time.time() - self.recording_start_time
            print(f"🛑 Recording stopped after {recording_duration:.1f}s. Processing...")
            
            if self.recording_thread:
                self.recording_thread.join()
            self._process_audio()
    
    def _record_audio(self):
        """Record audio in separate thread"""
        try:
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk
            )
            
            while self.is_recording:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    self.audio_data.append(data)
                except Exception as e:
                    print(f"Error during recording: {e}")
                    break
            
            stream.stop_stream()
            stream.close()
        except Exception as e:
            print(f"Audio stream error: {e}")
    
    def _process_audio(self):
        """Process recorded audio and type the recognized text"""
        if not self.audio_data:
            print("No audio data recorded.")
            return
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        
        try:
            # Write WAV file
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.rate)
                wf.writeframes(b''.join(self.audio_data))
            
            # Read audio file for speech recognition
            with sr.AudioFile(temp_file.name) as source:
                audio = self.recognizer.record(source)
            
            # Perform speech recognition
            try:
                print("🔄 Converting speech to text...")
                text = self.recognizer.recognize_google(audio)
                print(f"✅ Recognized: '{text}'")
                
                # Type the recognized text
                if text:
                    print("⌨️ Typing text...")
                    pyautogui.typewrite(text)
                    pyautogui.press('enter')  # Automatically press Enter after typing
                    print("✅ Text typed and Enter pressed!")
                
            except sr.UnknownValueError:
                print("❌ Could not understand the audio")
            except sr.RequestError as e:
                print(f"❌ Could not request results; {e}")
                
        except Exception as e:
            print(f"Error processing audio: {e}")
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except:
                pass
    
    def run(self):
        """Main control loop"""
        print("\n" + "="*60)
        print("🎯 ENHANCED FACE CONTROL WITH SPEECH RECOGNITION")
        print("="*60)
        print("Features:")
        print("• Move cursor: Move your nose")
        print("• Left click: Blink your eyes")
        print("• Speech-to-text: Open mouth to record, close to type")
        print("• Press 'q' to quit")
        print("="*60)
        print("Look straight ahead for the first 2 seconds for calibration")
        
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue
            
            frame = cv2.flip(frame, 1)  # Mirror effect
            h, w, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb_frame)
            
            if result.multi_face_landmarks:
                for face_landmarks in result.multi_face_landmarks:
                    # Get nose tip coordinates
                    nose_tip = face_landmarks.landmark[self.NOSE_TIP]
                    nose_x = nose_tip.x * w
                    nose_y = nose_tip.y * h
                    
                    # Calculate EAR for both eyes
                    left_ear = self.calculate_ear(self.LEFT_EYE_CORNERS, self.LEFT_EYE_VERTICAL, face_landmarks, w, h)
                    right_ear = self.calculate_ear(self.RIGHT_EYE_CORNERS, self.RIGHT_EYE_VERTICAL, face_landmarks, w, h)
                    
                    # Calculate MAR for mouth
                    mar = self.calculate_mar(face_landmarks, w, h)
                    
                    # Blink detection and clicking
                    current_time = time.time()
                    if self.is_blink_detected(left_ear, right_ear):
                        self.consecutive_blinks += 1
                    else:
                        if self.min_blink_frames <= self.consecutive_blinks <= self.max_blink_frames:
                            if current_time - self.last_click_time > self.click_cooldown:
                                try:
                                    pyautogui.click()
                                    self.last_click_time = current_time
                                    print(f"✓ CLICK! Blink duration: {self.consecutive_blinks} frames")
                                except Exception as e:
                                    print(f"Click failed: {e}")
                        self.consecutive_blinks = 0
                    
                    # IMPROVED Mouth detection for speech recording
                    mouth_is_open = self.is_mouth_open(mar)
                    
                    if mouth_is_open:
                        self.mouth_open_frames += 1
                        self.mouth_closed_frames = 0
                        
                        # Start recording if mouth has been open long enough
                        if self.mouth_open_frames >= self.min_mouth_open_frames and not self.is_recording:
                            self.start_recording()
                    else:
                        self.mouth_closed_frames += 1
                        self.mouth_open_frames = 0
                        
                        # Stop recording ONLY after mouth has been closed for the full delay
                        if self.mouth_closed_frames >= self.recording_stop_delay and self.is_recording:
                            self.stop_recording()
                    
                    # Calibration phase
                    if self.calibration_frames < self.max_calibration_frames:
                        self.nose_positions.append((nose_x, nose_y))
                        self.calibration_frames += 1
                        
                        cv2.putText(frame, f"Calibrating... {self.calibration_frames}/{self.max_calibration_frames}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.circle(frame, (int(nose_x), int(nose_y)), 5, (0, 255, 255), -1)
                        
                        if self.calibration_frames == self.max_calibration_frames:
                            self.center_x = np.mean([pos[0] for pos in self.nose_positions])
                            self.center_y = np.mean([pos[1] for pos in self.nose_positions])
                            print(f"Calibration complete! Center: ({self.center_x:.1f}, {self.center_y:.1f})")
                    
                    else:
                        # Cursor control with ENHANCED smoothing
                        rel_x = (nose_x - self.center_x) * self.movement_scale
                        rel_y = (nose_y - self.center_y) * self.movement_scale
                        
                        max_movement = min(w, h) * 0.3
                        rel_x = np.clip(rel_x, -max_movement, max_movement)
                        rel_y = np.clip(rel_y, -max_movement, max_movement)
                        
                        mapped_x = self.screen_w/2 + (rel_x / max_movement) * (self.screen_w/2)
                        mapped_y = self.screen_h/2 + (rel_y / max_movement) * (self.screen_h/2)
                        
                        mapped_x = np.clip(mapped_x, 0, self.screen_w)
                        mapped_y = np.clip(mapped_y, 0, self.screen_h)
                        
                        # Apply custom smoothing filter first
                        smooth_x, smooth_y = self.apply_smoothing_filter(mapped_x, mapped_y)
                        
                        # Then apply Kalman filter
                        if not self.kf_initialized:
                            self.kf.x = np.array([smooth_x, smooth_y, 0, 0])
                            self.kf_initialized = True
                            final_x, final_y = smooth_x, smooth_y
                        else:
                            self.kf.predict()
                            self.kf.update([smooth_x, smooth_y])
                            final_x, final_y = self.kf.x[:2]
                        
                        # Move cursor with the highly smoothed coordinates
                        pyautogui.moveTo(final_x, final_y, _pause=False)
                        
                        # Visual feedback
                        cv2.circle(frame, (int(nose_x), int(nose_y)), 5, (0, 255, 0), -1)
                        cv2.circle(frame, (int(self.center_x), int(self.center_y)), 3, (255, 0, 0), -1)
                        
                        # Draw mouth landmarks for visual feedback
                        for idx in self.MOUTH_OUTER:
                            try:
                                landmark = face_landmarks.landmark[idx]
                                x, y = int(landmark.x * w), int(landmark.y * h)
                                color = (0, 255, 0) if mouth_is_open else (255, 0, 0)
                                cv2.circle(frame, (x, y), 2, color, -1)
                            except:
                                pass
                        
                        # Status display
                        cv2.putText(frame, f"Cursor: ({final_x:.0f}, {final_y:.0f})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(frame, f"EAR: L={left_ear:.3f} R={right_ear:.3f}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        cv2.putText(frame, f"MAR: {mar:.3f} ({'OPEN' if mouth_is_open else 'CLOSED'})", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        # IMPROVED Recording status display
                        if self.is_recording:
                            recording_time = time.time() - self.recording_start_time
                            cv2.putText(frame, f"🎤 RECORDING... {recording_time:.1f}s", 
                                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        elif self.mouth_open_frames > 0:
                            cv2.putText(frame, f"Mouth open: {self.mouth_open_frames}/{self.min_mouth_open_frames}", 
                                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        elif self.mouth_closed_frames > 0 and self.mouth_closed_frames < self.recording_stop_delay:
                            seconds_left = (self.recording_stop_delay - self.mouth_closed_frames) / 30.0
                            cv2.putText(frame, f"Stopping in: {seconds_left:.1f}s", 
                                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
            
            # Instructions
            cv2.putText(frame, "Nose: Move | Blink: Click | Open mouth: Record | Close 2s: Type | 'q': Quit", 
                       (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow("Enhanced Face Control with Speech Recognition", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        if self.is_recording:
            self.stop_recording()
        
        self.cap.release()
        cv2.destroyAllWindows()
        self.audio.terminate()
        print("Enhanced face control stopped.")

if __name__ == "__main__":
    controller = EnhancedFaceController()
    controller.run()