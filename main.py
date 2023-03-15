import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np

def classify_allo(keypoints):
    """ Usefull keypoints for allo posture classification
        Keypoint 12: Right Shoulder
        Keypoint 14: Right Elbow
        Keypoint 16: Right Wrist
    """
    a = computeAngle(keypoints[12]['Data'], keypoints[14]['Data'], keypoints[16]['Data'])

    print(a)

def classify_drible(keypoints):
    """ Usefull keypoints for drible posture classification
        Keypoint 12: Right Shoulder
        Keypoint 16: Right Wrist
        Keypoint 24: Right Hip
    """
    a = computeAngle(keypoints[12]['Data'], keypoints[16]['Data'], keypoints[24]['Data'])
    print(a)

def classify_control(keypoints):
    """ Usefull keypoints for help posture classification
        Keypoint 12: Right Shoulder
        Keypoint 16: Right Wrist
        Keypoint 24: Right Hip

        Keypoint 11: Left Shoulder
        Keypoint 15: Left Wrist
        Keypoint 23: Left Hip
    """
    a = computeAngle(keypoints[12]['Data'], keypoints[16]['Data'], keypoints[24]['Data'])
    b = computeAngle(keypoints[11]['Data'], keypoints[15]['Data'], keypoints[23]['Data'])
    sum = a + 180 - b
    print(a,b,sum)
def classify_help(keypoints):
    """ Usefull keypoints for help posture classification
        Keypoint 12: Right Shoulder
        Keypoint 16: Right Wrist
        Keypoint 24: Right Hip

        Keypoint 11: Left Shoulder
        Keypoint 15: Left Wrist
        Keypoint 23: Left Hip
    """
    a = computeAngle(keypoints[12]['Data'], keypoints[16]['Data'], keypoints[24]['Data'])
    b = computeAngle(keypoints[11]['Data'], keypoints[15]['Data'], keypoints[23]['Data'])
    sum = a + b
    print(a,b,sum)

def classify_lift(keypoints):
    """ Usefull keypoints for lift posture classification
        Keypoint 12: Right Shoulder
        Keypoint 16: Right Wrist
        Keypoint 24: Right Hip

        Keypoint 11: Left Shoulder
        Keypoint 15: Left Wrist
        Keypoint 23: Left Hip
    """
    a = computeAngle(keypoints[12]['Data'], keypoints[16]['Data'], keypoints[24]['Data'])
    b = computeAngle(keypoints[11]['Data'], keypoints[15]['Data'], keypoints[23]['Data'])
    sum = a + b
    print(a,b,sum)


def computeAngle(a,b,c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def make_1080p():
    cap.set(3, 1920)
    cap.set(4, 1080)

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

# For webcam input:
#cap = cv2.VideoCapture("GH010444.MP4")
#cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture("psychopy/films/allo.mp4")

#make_1080p()

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    keypoints = []
    for data_point in results.pose_landmarks.landmark:
        keypoints.append({
            'Data': np.array([data_point.x,data_point.y,data_point.z]),
            'Visibility': data_point.visibility,
        })

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    classify_allo(keypoints)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))
    #cv2.imshow('MediaPipe Pose', cv2.flip(cv2.resize(image, (540, 960))    , 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()

