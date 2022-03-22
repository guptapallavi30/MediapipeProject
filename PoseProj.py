import cv2
import time
import PoseModule as pm

# cap = cv2.VideoCapture('./PoseVideos/2.mp4')
cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.poseDetector();
while True:
    success, img = cap.read() # returns true if image is read (false ow), and the array of images (empty ow)
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    # print(lmList[14])
    # cv2.circle(img, (lmList[14][1], lmList[14][2]), 40, (255, 0, 255), cv2.FILLED) # draw over point with bigger circle

    if not success:
        break

    cTime = time.time() # get current time
    fps = 1 / (cTime - pTime) # frames per seconds
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3) # add text to show on top of image
    cv2.imshow("Image",img)
    cv2.waitKey(1) # wait 1 ms
