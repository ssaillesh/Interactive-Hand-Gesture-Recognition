import pyautogui
import cv2
import mediapipe as mp
import numpy as np
import time

# Constants
FRAME_WIDTH = 160  # Further reduced frame size for faster processing
FRAME_HEIGHT = 120
TOUCH_THRESHOLD = 8  # Distance threshold for detecting touch (in pixels)
DEBOUNCE_TIME = 0.2  # Debounce time in seconds to prevent multiple clicks

# Initialize Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

# Initialize MediaPipe Hands with optimized settings
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Screen dimensions
screen_width, screen_height = pyautogui.size()

# Last click time for debounce
last_click_time = 0

# Skip every nth frame to reduce processing load
FRAME_SKIP = 2
frame_count = 0

while True:
    frame_count += 1
    success, img = cap.read()
    img = cv2.flip(img, 1)

    if not success:
        continue

    if frame_count % FRAME_SKIP == 0:
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process frame for hand landmarks
        result = hand.process(rgb_frame)
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks on the image
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmark positions
                index_finger_tip = None
                thumb_tip = None
                
                for id, lm in enumerate(hand_landmarks.landmark):
                    x = int(lm.x * FRAME_WIDTH)
                    y = int(lm.y * FRAME_HEIGHT)
                    
                    if id == 8:  # Index finger tip
                        index_finger_tip = (x, y)
                        screen_x = np.interp(x, [0, FRAME_WIDTH], [0, screen_width])
                        screen_y = np.interp(y, [0, FRAME_HEIGHT], [0, screen_height])
                        pyautogui.moveTo(screen_x, screen_y)
                        
                    if id == 4:  # Thumb tip
                        thumb_tip = (x, y)
                
                if index_finger_tip and thumb_tip:
                    distance = np.hypot(index_finger_tip[0] - thumb_tip[0], index_finger_tip[1] - thumb_tip[1])
                    
                    if distance < TOUCH_THRESHOLD:
                        current_time = time.time()
                        if current_time - last_click_time > DEBOUNCE_TIME:
                            pyautogui.click()
                            last_click_time = current_time

    # Display the result
    cv2.imshow("Camera Feed", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
