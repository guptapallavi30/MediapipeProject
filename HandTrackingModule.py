import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, model_complexity=1, detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.model_complexity = model_complexity
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.model_complexity, self.detectionConf, self.trackingConf)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks: # landmark exists-> hand detected
            for hand_i_landmarks in self.results.multi_hand_landmarks: # for each hand detected in frame
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_i_landmarks, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0, draw=True):

        lmList = [] # will contain all landmark pos
        if self.results.multi_hand_landmarks: # landmark exists-> hand detected
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark): # id is index of landmark (which point on hand), lm is actual landmark (x, y, z)
                # print(id, lm)
                h, w, c = img.shape # height, width, channel
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    #cv2.circle(img to draw on, loc, radius, color, filled?)
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        return lmList


    



def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    pTime = 0;
    cTime = 0;
    while True:
        success, img = cap.read()
        detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        #display fps (img to put text on, text, location, font, scale, color, thickness)
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    
        


if __name__ == "__main__":
    main()