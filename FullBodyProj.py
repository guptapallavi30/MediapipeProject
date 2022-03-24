import cv2
import time
import FullBodyModule as fbm

cap = cv2.VideoCapture(0)
detector = fbm.fullBodyDetector()
pTime = 0;
cTime = 0;
while True:
    success, img = cap.read()
    img = detector.findBody(img)
    poseLmList, handLmList = detector.findPosition(img)

    if poseLmList:
        print(poseLmList)
    if handLmList:
        print(handLmList)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    #display fps (img to put text on, text, location, font, scale, color, thickness)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
