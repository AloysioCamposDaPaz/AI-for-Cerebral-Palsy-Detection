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
        cv2_imshow(image)

        # breaking gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

draw_landmarks(frame, results)
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
