# !pip install mediapipe

import cv2
import mediapipe as mp
import time

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

print('mpDraw')
mpDraw

cap = cv2.VideoCapture('./PoseVideos/5.mp4')
pTime = 0

success, img = cap.read()
print("success", success)
print("img", img)

print("img.shape", img.shape)

import matplotlib.pyplot as plt

# # set size
# plt.figure(figsize=(5,5))
# plt.axis("off")

# # convert color from CV2 BGR back to RGB
# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(image)
# plt.show()

while True:
    success, img = cap.read() # returns true if image is read (false ow), and the array of images (empty ow)
    if not success:
      break
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert so img compatible w mediapipe lib
    results = pose.process(imgRGB) # process all landmark points
    #print(results.pose_landmarks)
    if results.pose_landmarks: # draw landmark points
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark): # organize the landmarks into list for easier access
            h, w, c = img.shape # height, width, channel
            #print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 50, (255, 0, 0), cv2.FILLED) # draw over point with bigger circle

    cTime = time.time() # get current time
    fps = 1 / (cTime - pTime) # frames per seconds
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3) # add text to show on top of image

    cv2.imshow("Image",img)
    # Plot pose world landmarks.
    # mpDraw.plot_landmarks(results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    cv2.waitKey(1) # wait 1 ms