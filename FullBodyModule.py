import cv2
import time
import PoseModule as pm
import HandTrackingModule as htm

class fullBodyDetector:
    def __init__(self, mode=False, model_complexity=1, smooth_lm=True, enable_seg=False, smooth_seg=True, detect_conf=0.5, track_conf=0.5, maxHands=2):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_lm = smooth_lm
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detect_conf = detect_conf
        self.track_conf = track_conf
        self.maxHands = maxHands
        
        self.poseDetector = pm.poseDetector(self.mode, self.model_complexity, self.smooth_lm, self.enable_seg, self.smooth_seg, self.detect_conf, self.track_conf)
        self.handsDetector = htm.handDetector(self.mode, self.maxHands, self.model_complexity, self.detect_conf, self.track_conf)

    def findBody(self, img, draw=True):
        img = self.poseDetector.findPose(img, draw)
        img = self.handsDetector.findHands(img, draw)
        return img
        
    def findPosition(self, img, draw=True):
        self.poseLmList = self.poseDetector.findPosition(img)
        self.handLmList = self.handsDetector.findPosition(img)
        return self.poseLmList, self.handLmList



def main():
    cap = cv2.VideoCapture(0)
    detector = fullBodyDetector()
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
    

if __name__ == "__main__":
    main()