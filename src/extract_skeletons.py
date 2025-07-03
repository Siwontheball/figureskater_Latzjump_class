# extract_skeletons.py

import os                             # for creating directories & listing files
import cv2                            # OpenCV: video I/O and image processing
import numpy as np                    # NumPy: numerical arrays and operations
import mediapipe as mp                # MediaPipe: for pose (keypoint) estimation

# Base folders: where raw videos live, and where we'll save processed .npy
DATA_DIR = "data"                  # path to your "inside"/"right"/"flat" folders
OUT_DIR  = "processed"             # where to dump the extracted sequences
SEQUENCE_LENGTH = 60                  # number of frames per clip (pad/truncate)

# Initialize MediaPipe’s pose module
mp_pose = mp.solutions.pose.Pose(
    static_image_mode=False,          # video stream, not single images
    min_detection_confidence=0.5      # threshold for detecting a person
)

def process_video(video_path):
    """
    Reads a video, runs pose estimation on each frame, and
    returns a fixed-length array of flattened keypoints.
    """
    cap = cv2.VideoCapture(video_path)  # open video file for frame-by-frame reading
    keypoints = []                      # list to hold each frame’s keypoints

    # Read until we have SEQUENCE_LENGTH frames or video ends
    while len(keypoints) < SEQUENCE_LENGTH:
        ret, frame = cap.read()         # ret=False when no more frames
        if not ret:
            break                       # exit if video shorter than SEQUENCE_LENGTH

        # OpenCV reads frames in BGR color order by default.
        # We convert to RGB so MediaPipe interprets colors correctly.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run pose estimation
        res = mp_pose.process(rgb)

        if res.pose_landmarks:
            # Extract (x,y) for each of 33 landmarks, flattened into one vector
            pts = [(lm.x, lm.y) for lm in res.pose_landmarks.landmark]
        else:
            # If no person detected, use zeros as a fallback
            pts = [(0.0, 0.0)] * 33

        # Flatten and store this frame’s keypoints
        keypoints.append(np.array(pts).flatten())

    cap.release()  # close video file handle

    # If fewer than SEQUENCE_LENGTH frames, repeat the last to pad out
    if len(keypoints) < SEQUENCE_LENGTH:
        pad_count = SEQUENCE_LENGTH - len(keypoints)
        keypoints.extend([keypoints[-1]] * pad_count)
    else:
        # If more, truncate to exactly SEQUENCE_LENGTH
        keypoints = keypoints[:SEQUENCE_LENGTH]

    # Stack into a single NumPy array of shape (SEQUENCE_LENGTH, 66)
    return np.stack(keypoints)

def main():
    # Loop over each class folder
    for label in ["Inside", "Right", "Flat"]:
        in_dir  = os.path.join(DATA_DIR, label)  # e.g. "../data/inside"
        out_dir = os.path.join(OUT_DIR,  label)  # e.g. "../processed/inside"
        os.makedirs(out_dir, exist_ok=True)      # create output folder if needed

        # Process every .mp4 in that folder
        for fn in os.listdir(in_dir):
            if not fn.endswith(".mp4"):
                continue                        # skip non-video files
            video_path = os.path.join(in_dir, fn)
            seq = process_video(video_path)    # extract pose sequence
            # Save as .npy, preserving class label in folder structure
            out_path = os.path.join(out_dir, fn.replace(".mp4", ".npy"))
            np.save(out_path, seq)

if __name__ == "__main__":
    main()  # kick off the extraction when this script is run
