# Source - https://stackoverflow.com/q
# Posted by Gumeo, modified by community. See post 'Timeline' for change history
# Retrieved 2025-11-27, License - CC BY-SA 3.0

import numpy as np
import cv2
import os

# Blue cicle means that the annotation haven't started
# Green circle is a good pose
# Red is a bad pose
# White circle means we are done, press d for that

# Instructions on how to use!
# Press space to swap between states, you have to press space when the person
# starts doing poses. 
# Press d when the person finishes.
# press q to quit early, then the annotations are not saved, you should only 
# use this if you made a mistake and need to start over.

import time
cap = cv2.VideoCapture(r'F:\social_sniffing\rawdata\1106077\olfactory_ctrls\2025-09-29T13-20-52\Video\\1106077.avi')

# Use original video FPS
fps_val = cap.get(cv2.CAP_PROP_FPS)
try:
    fps = float(fps_val)
except Exception:
    fps = 0.0
if fps <= 1.0 or np.isnan(fps):
    fps = 50.0  # fallback
BASE_PERIOD = 1.0 / fps
FAST_MULT = 5.0  # 3x speed when holding RIGHT

# Start with the beginning state as 10 to indicate that the procedure has not started
current_state = 10
saveAnnotations = True
annotation_list = []

# per-key annotation lists and frame counter
grooming = []
approaches_port = []
rearing = []
chase = []
sampling = []
frame_idx = 0

# total frames (for clamping seeks)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Windows OpenCV keycodes for arrow keys (use raw value from waitKey)
KEY_LEFT  = 2424832
KEY_RIGHT = 2555904
KEY_UP    = 2490368
KEY_DOWN  = 2621440

cap.isOpened()
colCirc = (255,0,0)
while cap.isOpened():
    t0 = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    cv2.circle(frame, (50,50), 50, colCirc, -1)
    cv2.imshow('frame', frame)

    # Poll keys quickly; we'll control timing ourselves
    k_raw = cv2.waitKey(1)
    k = k_raw & 0xFF

    # Hold RIGHT arrow to play 3x faster
    fast_play = (k_raw == ord('t'))

    # Skip backward 10 frames with LEFT arrow
    if k_raw == ord('e'):
        frame_idx = max(frame_idx - 100, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        continue

    # mark this frame
    if k == ord('a'):
        grooming.append(frame_idx)
        colCirc = (0,165,255)
    if k == ord('s'):
        sampling.append(frame_idx)
        colCirc = (255,165,0)
    if k == ord('p'):
        approaches_port.append(frame_idx)
        colCirc = (0,255,255)
    if k == ord('r'):
        rearing.append(frame_idx)
        colCirc = (255,0,255)
    if k == ord('c'):
        chase.append(frame_idx)
        colCirc = (128,0,128)

    if k == ord('d'):
        current_state = 11
        colCirc = (255,255,255)

    if k == ord('q'):
        print("You quit! Restart the annotations by running this script again!")
        saveAnnotations = True
        break

    annotation_list.append(current_state)

    # Timing: sleep to match desired playback rate
    desired_period = BASE_PERIOD / (FAST_MULT if fast_play else 1.0)
    dt = time.perf_counter() - t0
    if desired_period > dt:
        time.sleep(desired_period - dt)

cap.release()
cv2.destroyAllWindows()

if saveAnnotations:
    np.save(r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\2025-09-29T13-20-52\Video\grooming.npy', np.array(grooming))
    np.save(r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\2025-09-29T13-20-52\Video\sampling.npy', np.array(sampling))
    np.save(r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\2025-09-29T13-20-52\Video\approaches_port.npy', np.array(approaches_port))
    np.save(r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\2025-09-29T13-20-52\Video\rearing.npy', np.array(rearing))
    np.save(r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\2025-09-29T13-20-52\Video\chase.npy', np.array(chase))
    print("Annotations saved successfully!")