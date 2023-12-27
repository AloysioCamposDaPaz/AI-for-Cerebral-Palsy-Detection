#extract landmarks for input
pose = []
# NOTE: you want to flatten array to use it in LSTM
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    #face = np.array([[res.x, res.y, res.z, res.visibility] for res in results.face_landmarks]).flatten() if results.face_landmarks else np.zeros(132)
    #return np.concatenate ([pose, face])
    return pose