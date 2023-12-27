import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

#from google.colab.patches import cv2_imshow
#create two variables
mp_holistic = mp.solutions.holistic #holistic model
mp_drawing = mp.solutions.drawing_utils #drawing utlities

def mediapipe_detection(image,model):
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #colour conversion BGR 2 RGB
  image.flags.writeable = False #image no longer writable
  results = model.process(image) # make detection using mediape
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #colour conversion RGB 2 BGR
  return image, results
  pass

def draw_landmarks (image, results):
  mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)


cap = cv2.VideoCapture('43.mp4')  # use webcam as input
# set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # read feed -> frame is image from webcam
        ret, frame = cap.read()  # grab webcam frame at this point in time

        # make detection
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # draw landmarks
        draw_landmarks(image, results)

        # show frame to user screen
        cv2.imshow('baby',image)

        # breaking gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

draw_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#extract landmarks for input
pose = []
# NOTE: you want to flatten array to use it in LSTM
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    #face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks]).flatten() if results.face_landmarks else np.zeros(132)
    #return np.concatenate ([pose, face])
    return pose