import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands 
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# get frame rates
pTime = 0;
cTime = 0;
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks: # landmark exists-> hand detected
        for hand_i_landmarks in results.multi_hand_landmarks: # for each hand detected in frame
            for id, lm in enumerate(hand_i_landmarks.landmark): # id is index of landmark (which point on hand), lm is actual landmark (x, y, z)
                # print(id, lm)
                h, w, c = img.shape # height, width, channel
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                if id == 4: # if id matches tip of pinky
                    #cv2.circle(img to draw on, loc, radius, color, filled?)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)


            mpDraw.draw_landmarks(img, hand_i_landmarks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    #display fps (img to put text on, text, location, font, scale, color, thickness)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)