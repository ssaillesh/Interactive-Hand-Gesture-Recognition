import cv2
import mediapipe as mp

frameWidth = 600
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands()

while True:
   success, img = cap.read()
   if success:
      rgb_frame = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
      result = hand.process(rgb_frame)
      if result.multi_hand_landmarks:
         for hand_landmarks in result.multi_hand_landmarks:
            print(hand_landmarks)
            mp_drawing.draw_landmarks(img,hand_landmarks,mp_hands.HAND_CONNECTIONS)
   cv2.imshow("Result", img)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break




    