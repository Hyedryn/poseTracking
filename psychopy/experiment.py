import os
import time
from pathlib import Path
import pandas as pd
import random
import logging
from ffpyplayer.player import MediaPlayer
import cv2
from winsound import Beep
import threading
import numpy as np
import openpyxl
import pygame
import moviepy.editor


global end_video_thread
end_video_thread = False

def alert_1():
    threading.Thread(target=Beep, args=(294,500), daemon=True).start()

def alert_2():
    threading.Thread(target=Beep, args=(440,250), daemon=True).start()

def play_Trial_Video(trial, name):
    global end_video_thread
    filePath = "films/"+trial + ".mp4"

    os.environ['DISPLAY'] = ":0.1"
    res = (1920, 1080)
    scaled_res = tuple(ti/2 for ti in res)
    # Create a VideoCapture object and read from input file
    cap = cv2.VideoCapture(filePath, apiPreference=cv2.CAP_ANY, params=[
    cv2.CAP_PROP_FRAME_WIDTH, res[0],
    cv2.CAP_PROP_FRAME_HEIGHT, res[1]])

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video file")

    # Read until video is completed
    while (cap.isOpened()):

        if end_video_thread:
            break

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            # Display the resulting frame
            cv2.imshow(name, frame)
            #cv2.imshow(name, cv2.resize(frame, scaled_res))
            cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release
    # the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
def launch_Video_Thread(trial,name):
    video_thread = threading.Thread(target=play_Trial_Video, args=(trial,name), daemon=True)
    video_thread.start()

    return video_thread

#Experiment data
df = pd.read_excel('cond.xlsx') # can also index sheet by name or fetch all sheets
trial_list = df["mouvements"].tolist()
n_reps = 20
random_trial_list = []
assert (n_reps % len(trial_list)) == 0, "Number of repetitions must be a multiple of the number of trials"

for i in range(n_reps//len(trial_list)):
    random_trial_list.extend(trial_list)

repCount = {}
for t in trial_list:
    repCount[t] = 0

random.shuffle(random_trial_list)

_thisDir = os.path.dirname(os.path.abspath(__file__))
BIDS_path = os.path.join(_thisDir,"BIDS_dataset")

print("Welcome to the pose analysis experiment!")
print("First, we need to set up the experiment. Please enter the following information:")
while True:
    sub = input("Subject name: ")
    sess = input("Session name: ")

    exp_path = os.path.join(BIDS_path, "sub-" + sub, "ses-" + sess)
    if os.path.exists(exp_path):
        print("Warning: this subject and session already exists. Overwrite? (y/n)")
        if input() == "y":
            print("Confirm subject and session name. (y/n)", sub, sess)
            if input() == "y":
                break
    else:
        print("Confirm subject and session name. (y/n)", sub, sess)
        if input() == "y":
            break

startDatetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
Path(exp_path).mkdir(parents=True, exist_ok=True)

cam_path = os.path.join(exp_path,"camera")
Path(cam_path).mkdir(parents=True, exist_ok=True)

logfile = os.path.join(exp_path, "sub-" + sub + "_ses-" + sess + "_log.txt")
logging.basicConfig(filename=logfile,
                    filemode='w',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# Create the experiment

for i in range(len(random_trial_list)):
    alert_2_Done = False

    current_time = time.time()
    trial = random_trial_list[i]

    cam_file = "sub-" + sub + "_ses-" + sess + "_task-" + trial + "_rep-" + str(
        repCount[trial]) + "_camera.mp4"

    repCount[trial] += 1

    logging.info("Start of trial " + str(i) + ": " + trial)
    print("Trial", i+1, "of", len(random_trial_list), ":", trial)

    file = 'films' + '/' + trial + '.mp4'

    # Capturing video from webcam:
    cap = cv2.VideoCapture(0)
    currentFrame = 0
    # Get current width of frame
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    # Get current height of frame
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(os.path.join(cam_path,cam_file), fourcc, 30, (int(width), int(height)))

    # Wait for a total of 10 seconds
    while (current_time + 10) > time.time():
        time.sleep(0.05)

    logging.info("[Trial " + str(i) + "] Start of video")
    alert_1()
    logging.info("[Trial " + str(i) + "] Alert 1 played")
    print("Start of video")
    video_thread = launch_Video_Thread(trial,"Subject " + sub + " - Session "+ sess + " - Trial " + str(i))

    mirror = False
    while (True):
        ret, frame = cap.read()

        if ret == True:
            if mirror == True:
                # Mirror the output video frame
                frame = cv2.flip(frame, 1)
            # Saves for video
            out.write(frame)
            # Display the resulting frame
            #cv2.imshow('frame', frame)
            #if cv2.waitKey(5) & 0xFF == 27:
            #    break
        else:
            break

        if alert_2_Done == False and (current_time + 30) <= time.time():
            alert_2()
            logging.info("[Trial " + str(i) + "] Alert 2 played")
            alert_2_Done = True

        if alert_2_Done and (current_time + 40) <= time.time():
            break

        # To stop duplicate images
        currentFrame += 1

    end_video_thread = True
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Kill video thread
    video_thread.join()

    end_video_thread = False

    print("Time delta:", time.time() - current_time)

    logging.info("[Trial " + str(i) + "] End of video")

# Save metadata into subject & session specific JSON
subject_sess_metadata = {
'subject':sub,
'session':sess,
'sessionMouvementChronology': random_trial_list,
'sessinMouvementFrequency': repCount,
'startDatetime': startDatetime,
'endDatetime': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
}

import json

with open(os.path.join(exp_path,"sub-"+sub + "_ses-"+sess + '_metadata.json'), 'w') as fp:
    json.dump(subject_sess_metadata, fp)

logging.info("End of experiment")
print("End of experiment")
