# NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED
# NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED
# NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED
# NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED NOT USED


import numpy as np
import os

#actions that we try to detect
actions = np.array(['fidgety','writhing','cramped_synchronized', 'abnormal'])

#number of videos
no_sequences = 140
#number of frames we use to detect
sequence_length = 125

cap = cv2.VideoCapture('??????????????????') #add video name

with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence=0.5) as holistic:
    #loop through actions
    for action in actions:
        #loop through sequences aka videos
        for sequence in range (sequence_length):
            for frame_num in range(sequence_length):
                #read feed
                ret, frame = cap.read()

                #make directions
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                #draw landmarks
                draw_styled_landmarks(image,results)

                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                #show landmarks
                cv2.imshow('openCV feed', image)

    #break gracefully
    if cv2.waitKey(10) & 0XFF == ord('q'):
        break

    cap.release()
    cv2.destroyAllWindows()

