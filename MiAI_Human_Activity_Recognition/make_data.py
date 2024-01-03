import cv2
import mediapipe as mp
import pandas as pd

# Đọc ảnh từ webcam
# cap = cv2.VideoCapture(0)

# TOPSPIN
# video_paths = ["/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test4.mp4", "/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspin.mp4"]

# TOPSPIN BACKHAND
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspinbackhand.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspinbackhand2.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspinbackhand3.mp4")

# TOPSPIN
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspin.mp4")

# Make data
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspin2.mp4")
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspin4.mp4")

# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Topspin.mp4")
## (cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test4.mp4"))

# PUSH
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Push.mp4")

# Make data
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Push2.mp4")

# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test10.mp4")
## (cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/test5.mp4"))


# URH
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/30fps-doc.mov")

# PUSH BACK HAND
# cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Pushbackhand.mp4")
## (cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/pushbackhand.mp4"))

# Make data
cap = cv2.VideoCapture("/Users/thanhlong/Documents/HCMUT/Test_code/videoTest/dataset_Pushbackhand2.mp4")


# Khởi tạo thư viện mediapipe
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []
label = "PUSHBACKHAND"
no_of_frames = 700

def make_landmark_timestep(results):
    print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # Vẽ các đường nối
    # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

    # Vẽ các điểm nút
    for id, lm in enumerate(results.pose_landmarks.landmark[:mpPose.PoseLandmark.LEFT_HIP.value]):
        h, w, c = img.shape
        print(id, lm)
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


while len(lm_list) <= no_of_frames:
    ret, frame = cap.read()
    if ret:
        # Nhận diện pose
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)

        if results.pose_landmarks:
            # Draw only upper body landmarks (excluding legs)
            upper_body_landmarks = results.pose_landmarks.landmark[:mpPose.PoseLandmark.LEFT_HIP.value]

            # Ghi nhận thông số khung xương
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # Vẽ khung xương lên ảnh
            frame = draw_landmark_on_image(mpDraw, results, frame)

        cv2.imshow("image", frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Write vào file csv
df  = pd.DataFrame(lm_list)
df.to_csv(label + ".txt")
cap.release()
cv2.destroyAllWindows()