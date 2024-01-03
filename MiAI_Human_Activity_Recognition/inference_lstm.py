import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf

label = "Warmup...."
n_time_steps = 4
lm_list = []

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

model = tf.keras.models.load_model("model.h5")

# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/success4-30fps-front.mov")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/fail.mov")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test3.mp4")

# PUSH
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test6.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test9.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test10.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test5.mp4")
cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Push.mp4")

# TOPSPIN
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test7.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test4.mp4")

# PUSH BACKHAND
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/pushbackhand.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/pushbackhand2.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/pushbackhand3.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Pushbackhand.mp4")

# TOPSPIN BACKHAND
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/topspinbackhand2.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspinbackhand.mp4")

def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm


def draw_landmark_on_image(mpDraw, results, img):
    # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # for id, lm in enumerate(results.pose_landmarks.landmark):
    #     h, w, c = img.shape
    #     print(id, lm)
    #     cx, cy = int(lm.x * w), int(lm.y * h)
    #     cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
    for id, lm in enumerate(results.pose_landmarks.landmark[:mpPose.PoseLandmark.LEFT_HIP.value]):
        h, w, c = img.shape
        # print(id, lm)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

    upper_body_connections = [
            (mpPose.PoseLandmark.LEFT_SHOULDER.value, mpPose.PoseLandmark.LEFT_ELBOW.value),
            (mpPose.PoseLandmark.RIGHT_SHOULDER.value, mpPose.PoseLandmark.RIGHT_ELBOW.value),
            (mpPose.PoseLandmark.LEFT_ELBOW.value, mpPose.PoseLandmark.LEFT_WRIST.value),
            (mpPose.PoseLandmark.RIGHT_ELBOW.value, mpPose.PoseLandmark.RIGHT_WRIST.value),
            (mpPose.PoseLandmark.LEFT_SHOULDER.value, mpPose.PoseLandmark.RIGHT_SHOULDER.value),
            (mpPose.PoseLandmark.LEFT_WRIST.value, mpPose.PoseLandmark.LEFT_PINKY.value),
            (mpPose.PoseLandmark.LEFT_PINKY.value, mpPose.PoseLandmark.LEFT_INDEX.value),
            (mpPose.PoseLandmark.LEFT_WRIST.value, mpPose.PoseLandmark.LEFT_INDEX.value),
            (mpPose.PoseLandmark.LEFT_WRIST.value, mpPose.PoseLandmark.LEFT_THUMB.value),
            (mpPose.PoseLandmark.RIGHT_WRIST.value, mpPose.PoseLandmark.RIGHT_PINKY.value),
            (mpPose.PoseLandmark.RIGHT_PINKY.value, mpPose.PoseLandmark.RIGHT_INDEX.value),
            (mpPose.PoseLandmark.RIGHT_WRIST.value, mpPose.PoseLandmark.RIGHT_INDEX.value),
            (mpPose.PoseLandmark.RIGHT_WRIST.value, mpPose.PoseLandmark.RIGHT_THUMB.value),
            (mpPose.PoseLandmark.NOSE.value, mpPose.PoseLandmark.LEFT_EYE_INNER.value),
            (mpPose.PoseLandmark.LEFT_EYE_INNER.value, mpPose.PoseLandmark.LEFT_EYE_OUTER.value),
            (mpPose.PoseLandmark.LEFT_EYE_OUTER.value, mpPose.PoseLandmark.LEFT_EAR.value),
            (mpPose.PoseLandmark.RIGHT_EYE_INNER.value, mpPose.PoseLandmark.RIGHT_EYE_OUTER.value),
            (mpPose.PoseLandmark.RIGHT_EYE_OUTER.value, mpPose.PoseLandmark.RIGHT_EAR.value),
            (mpPose.PoseLandmark.NOSE.value, mpPose.PoseLandmark.RIGHT_EYE_INNER.value),
            (mpPose.PoseLandmark.LEFT_EYE_INNER.value, mpPose.PoseLandmark.LEFT_EYE.value),
            (mpPose.PoseLandmark.RIGHT_EYE_INNER.value, mpPose.PoseLandmark.RIGHT_EYE.value),
            # Add more connections as needed for your specific case
        ]
    for connection in upper_body_connections:
            start_idx, end_idx = connection
            start_point = (int(upper_body_landmarks[start_idx].x * w), int(upper_body_landmarks[start_idx].y * h))
            end_point = (int(upper_body_landmarks[end_idx].x * w), int(upper_body_landmarks[end_idx].y * h))

            # Draw a line between the start and end points
            cv2.line(img, start_point, end_point, (0, 255, 0), 2, cv2.FILLED)  # Green line for each connection

    return img


def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img


def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    # print(lm_list.shape)
    results = model.predict(lm_list)
    print(results[0][0], results[0][1], results[0][2], results[0][3])

    if results[0][0] > 0.9:
        label = "TOPSPIN"
    elif results[0][1] > 0.9:
        label = "TOPSPIN BACKHAND"
    elif results[0][2] > 0.9:
        label = "PUSH"
    elif results[0][3] > 0.9:
        label = "PUSH BACK HAND"
    else:
        label = "???"
    return label


i = 0
warmup_frames = 5

while True:

    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    i = i + 1
    if i > warmup_frames:
        # print("Start detect....")

        if results.pose_landmarks:
            c_lm = make_landmark_timestep(results)
            upper_body_landmarks = results.pose_landmarks.landmark[:mpPose.PoseLandmark.LEFT_HIP.value]

            lm_list.append(c_lm)
            if len(lm_list) == n_time_steps:
                # predict
                t1 = threading.Thread(target=detect, args=(model, lm_list,))
                t1.start()
                lm_list = []

            img = draw_landmark_on_image(mpDraw, results, img)

    img = draw_class_on_image(label, img)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
