# !pip install mediapipe

import cv2
import mediapipe as mp
import time

class poseDetector():
    
    def __init__(self, mode=False, model_complexity=1, smooth_lm=True, enable_seg=False, smooth_seg=True, detect_conf=0.5, track_conf=0.5):
        self.mode = mode
        self.model_complexity = model_complexity
        self.smooth_lm = smooth_lm
        self.enable_seg = enable_seg
        self.smooth_seg = smooth_seg
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth_lm, self.enable_seg, self.smooth_seg, self.detect_conf, self.track_conf)
        self.landmarks = []
    
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert so img compatible w mediapipe lib
        self.results = self.pose.process(imgRGB) # process all landmark points
        if self.results.pose_landmarks: # draw landmark points
            if(draw):
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                
        return img;
        
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark): # organize the landmarks into list for easier access
                h, w, c = img.shape # height, width, channel
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 20, (255, 0, 0), cv2.FILLED) # draw over point with bigger circle
        return lmList

    


def main():
    # cap = cv2.VideoCapture('./PoseVideos/5.mp4')
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector();
    while True:
        success, img = cap.read() # returns true if image is read (false ow), and the array of images (empty ow)
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        # print(lmList[14])
        if not success:
            break

        cTime = time.time() # get current time
        fps = 1 / (cTime - pTime) # frames per seconds
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3) # add text to show on top of image
        cv2.imshow("Image",img)
        cv2.waitKey(1) # wait 1 ms

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()