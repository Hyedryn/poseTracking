import os
import cv2
import mediapipe as mp
from matplotlib import pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
import numpy as np
import json
import csv
from pathlib import Path
from scipy.interpolate import make_interp_spline
import scipy.signal


def classify_allo(keypoints, currentFrame):
    """ Usefull keypoints for allo posture classification
        Keypoint 12: Right Shoulder
        Keypoint 14: Right Elbow
        Keypoint 16: Right Wrist
    """
    a = computeAngle(keypoints[12]['Data'], keypoints[14]['Data'], keypoints[16]['Data'])
    b = computeAngle(keypoints[14]['Data'], keypoints[12]['Data'], keypoints[24]['Data'])
    sum = a + b
    return [a, b, a, currentFrame]

def classify_drible(keypoints, currentFrame):
    """ Usefull keypoints for drible posture classification
        Keypoint 12: Right Shoulder
        Keypoint 16: Right Wrist
        Keypoint 24: Right Hip
    """
    a = computeAngle(keypoints[12]['Data'], keypoints[16]['Data'], keypoints[24]['Data'])
    return [a, 0, a, currentFrame]

def classify_control(keypoints, currentFrame):
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
    return [a,b,sum, currentFrame]
def classify_help(keypoints, currentFrame):
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
    return [a,b,sum, currentFrame]

def classify_lift(keypoints, currentFrame):
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
    return [a,b,sum, currentFrame]


def computeAngle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    #a = a[0:1]
    #b = b[0:1]
    #c = c[0:1]

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def analyseVideo(filepath, bidsBasePath, subj, sess, task, rep):

    derivatives = os.path.join(bidsBasePath, 'derivatives', "analysisQuentin", subj, sess, "mediapose")
    Path(derivatives).mkdir(parents=True, exist_ok=True)

    def change_res(width, height):
        cap.set(3, width)
        cap.set(4, height)

    fps = 15.01
    # For webcam input:
    cap = cv2.VideoCapture(filepath)
    currentFrame = 1
    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(os.path.join(derivatives, subj + "_" + sess + "_task-" + task + "_rep-" + rep + "_annotatedVideo.avi"), fourcc, fps, (int(width), int(height)))
    change_res(1920, 1080)

    landmarkList = []
    angleList = []

    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2) as pose:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            keypoints = []
            for data_point in results.pose_world_landmarks.landmark:
                keypoints.append({
                    'Data': [data_point.x, data_point.y, data_point.z],
                    'Visibility': data_point.visibility,
                })
            landmarkList.append(keypoints)

            # Draw the pose annotation on the image.
            image.flags.writeable = True

            if task == "allo":
                s = classify_allo(keypoints, currentFrame)
            elif task == "drible":
                s = classify_drible(keypoints, currentFrame)
            elif task == "controle":
                s = classify_control(keypoints, currentFrame)
            elif task == "help":
                s = classify_help(keypoints, currentFrame)
            elif task == "lift":
                s = classify_lift(keypoints, currentFrame)
            else:
                print("Task not found", task)
                s = [0,0,0,0]
            angleList.append(s)
            image = cv2.putText(image, "a:" + str(int(s[0])), (100, 185), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            image = cv2.putText(image, "b:" + str(int(s[1])), (100, 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            image = cv2.putText(image, "s:" + str(int(s[2])), (100, 255), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # Saves for video
            out.write(image)

            currentFrame += 1

            if currentFrame % int(fps) == 0:
                print("Second processed: " + str(currentFrame/int(fps)))
                if currentFrame >= 30*fps:
                    break

    cap.release()
    out.release()


    # Save angle as plot
    angleList = np.array(angleList)
    ang = np.radians(angleList[:, 2])
    ang = angleList[:, 2]
    time = np.linspace(0, len(angleList)/fps, len(angleList))
    plt.figure()
    plt.plot(time, ang, label='data')

    # apply a 3-pole lowpass filter at 0.1x Nyquist frequency
    b, a = scipy.signal.butter(3, 0.075)
    filtered = scipy.signal.filtfilt(b, a, ang)
    plt.plot(time, filtered, label='filtered')

    max, prop_max = scipy.signal.find_peaks(filtered, distance=5)
    min, prop_min = scipy.signal.find_peaks(-filtered, distance=5)

    plt.plot(max/fps, filtered[max], "x", label='max')
    plt.plot(min/fps, filtered[min], "x", label='min')

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.xlim(0, 30)
    plt.legend()
    plt.title("Angle in function of time for task "+task)
    plt.savefig(os.path.join(derivatives, subj + "_" + sess + "_task-" + task + "_rep-" + rep + "_angle.png"))


    # Compute max frequency of the signal
    max_recorded = max[max >= fps*20]
    max_estim = max[max >= fps * 10]
    min_recorded = min[min >= fps*20]
    min_estim = min[min >= fps * 10]
    freq_recorded_max = (len(max_recorded)-1)/((max_recorded[-1]-max_recorded[0])/fps)
    freq_estim_max = (len(max_estim) - 1) / ((max_estim[-1] - max_estim[0]) / fps)
    freq_recorded_min = (len(min_recorded)-1)/((min_recorded[-1]-min_recorded[0])/fps)
    freq_estim_min = (len(min_estim) - 1) / ((min_estim[-1] - min_estim[0]) / fps)

    if os.path.exists(os.path.join(derivatives, subj + "_" + sess + "_stats.json")):
        with open(os.path.join(derivatives, subj + "_" + sess + "_stats.json"), "r") as f:
            data = json.load(f)
    else:
        data = {}


    if task not in data:
        data[task] = {}
    if rep not in data[task]:
        data[task][rep] = {}

    data[task][rep]["freq_recorded_max"] = freq_recorded_max
    data[task][rep]["freq_estim_max"] = freq_estim_max
    data[task][rep]["freq_recorded_min"] = freq_recorded_min
    data[task][rep]["freq_estim_min"] = freq_estim_min


    with open(os.path.join(derivatives, subj + "_" + sess + "_stats.json"), "w") as f:
        json.dump(data, f)


    # Save angleList to csv
    with open(os.path.join(derivatives, subj + "_" + sess + "_task-" + task + "_rep-" + rep + "_stats.csv"), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["angleRight", "angleLeft", "angleSum", "frame"])
        writer.writerows(angleList)

    #Save landmarkslist as json
    with open(os.path.join(derivatives, subj + "_" + sess + "_task-" + task + "_rep-" + rep + "_landmarks.json"), 'w') as outfile:
        json.dump(landmarkList, outfile)



_thisDir = os.path.dirname(os.path.abspath(__file__))
BIDS_path = os.path.join(_thisDir,"BIDS_dataset")

for subject in os.listdir(BIDS_path):
    if not subject.startswith("sub"):
        continue
    for session in os.listdir(os.path.join(BIDS_path, subject)):
        exp_path = os.path.join(BIDS_path, subject, session)
        for video in os.listdir(os.path.join(exp_path,"camera")):

            if not video.endswith(".mp4"):
                continue

            print("Processing video: ", video)
            analyseVideo(os.path.join(exp_path,"camera",video), BIDS_path, subject, session, video.split("_")[2].split("-")[1], video.split("_")[3].split("-")[1])



