
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical


class poseDetector():
    def __init__(self, mode = False, model_complexity = 1, smooth_landmarks = True, enable_segmentation = False, smooth_segmentation = True,
                  detectionCon = 0.9, trackCon = 0.5):

        self.mode = mode #whenever we create a new object it will have it's own variables
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.detectionCon, self.trackCon)



    def findPose(self, img, draw = True ): # will ask user if we want to draw or not
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img, results


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    return pose


mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

#set mediapipe model
detector = poseDetector()

actions = np.array(['gangnam style','under the sea','no dance'])
#actions = np.array(['fidgety', 'cramped'])

cap = cv2.VideoCapture(1)

#input shape 15,132 for the 2 second model
#input shape 38,132 for the 5 second model

#load model that we trained
model = Sequential()
model.add(LSTM(64,return_sequences=True, activation='relu', input_shape=(15,132)))
model.add(LSTM(128,return_sequences=True, activation='relu')) #true because there is another layer next
model.add(LSTM(64,return_sequences=False, activation='relu')) # false because this is the last layer
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights('SeaVSGangamVSNothing_dataset_750vid_2_sec_7fps.h5')

#new detection variables
sequence =[] #collects 120 frames to generate prediction
sentence =[] # allows us to concatenate history of detections together
threshold = 0.9 #confidence metrics -> only render results above a threshold
res =0
pTime = 0
while cap.isOpened():
    #read feed
    ret, frame = cap.read()

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert image to RGB
    results = pose.process(imgRGB)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    #frame rate
    #cv2.putText(frame, str(int(fps)), (0, 50),
                #cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

    # display_results!!!!!!!!!!!!
    if (results.pose_landmarks):
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        # for id, lm in enumerate(results.pose_landmarks.landmark):
    cv2.putText(frame, actions[np.argmax(res)], (40, 50), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 3)
    cv2.imshow("image", frame)

    # Delay until next frame is shown
    cv2.waitKey(1)

    #prediction logic
    keypoints = keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    sequence.insert(0,keypoints)
    sequence = sequence[:15] #we take 120 frames since we trained with 120 frames of data
    #print (sequence)

    if len(sequence) == 15:
        test = np.expand_dims(sequence, axis=0)
        res = model.predict(test)[0]
        print (actions[np.argmax(res)])



