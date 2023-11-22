import cv2
import numpy as np
import os

path = "data/dynerf/sear_steak"
videos_paths = os.listdir(path)
os.makedirs(os.path.join(path, "frames"), exist_ok=True)

video_paths = [x for x in videos_paths if x.endswith(".mp4")]

for video_path in videos_paths:
    save_path = os.path.join(path, "frames", video_path.split(".")[0])
    os.makedirs(save_path, exist_ok=True)
    video_path = os.path.join(path, video_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, fx=0.5, fy=0.5)
        if not ret:
            break
        cv2.imwrite(os.path.join(save_path, f"{frame_count:04d}.png"), frame)
        frame_count += 1
    print(f"Done with {video_path}")