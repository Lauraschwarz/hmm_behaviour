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
file_path = (r'F:\social_sniffing\rawdata\1125563_olf\2025-09-09T13-41-23\Video\\1125563_olf.avi')
outpath = (r'F:\social_sniffing\derivatives\1125563\olfactory_ctrls\CTRL\2025-09-09T13-41-23\Video\sampling_bouts.npy')
cap = cv2.VideoCapture(file_path)

# Use original video FPS
fps_val = cap.get(cv2.CAP_PROP_FPS)
try:
    fps = float(fps_val)
except Exception:
    fps = 0.0
if fps <= 1.0 or np.isnan(fps):
    fps = 50.0  # fallback
BASE_PERIOD = 1.0 / fps
FAST_MULT = 5.0  # 5x speed when holding RIGHT

# Start with the beginning state as 10 to indicate that the procedure has not started
current_state = 10
saveAnnotations = True
annotation_list = []

# per-key annotation lists and frame counter
grooming = []
approaches_port = []
rearing = []
chase = []
sampling = []           # remove if you don't want single-frame marks anymore
sampling_active = False
sampling_start = None
sampling_bouts = []
frame_idx = 0

# total frames (for clamping seeks)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Windows OpenCV keycodes for arrow keys (use raw value from waitKey)
KEY_LEFT  = 2424832
KEY_RIGHT = 2555904
KEY_UP    = 2490368
KEY_DOWN  = 2621440
fast_mode = False
quit_now = False

cap.isOpened()
colCirc = (255,0,0)
while cap.isOpened():
    t0 = time.perf_counter()

    ret, frame = cap.read()
    if not ret:
        break
    frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

    # frame counter: current/total (top-right)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 2
    text = f"{frame_idx+1}/{total_frames}"
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = frame.shape[1] - tw - 10
    y = 10 + th
    cv2.rectangle(frame, (x-6, y-th-6), (x+tw+6, y+baseline+6), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    cv2.circle(frame, (50,50), 50, colCirc, -1)
    cv2.imshow('frame', frame)

    k_raw = cv2.waitKey(1)
    k = k_raw & 0xFF

    # toggle fast mode with 't'
    if k == ord('t'):
        fast_mode = not fast_mode

    # skip backward 100 frames with LEFT arrow (use arrow keycode)
    if k_raw == ord('e'):
        frame_idx = max(frame_idx - 100, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        continue

    # Timing: sleep to match desired playback rate
    desired_period = BASE_PERIOD / (FAST_MULT if fast_mode else 1.0)
    dt = time.perf_counter() - t0
    if desired_period > dt:
        time.sleep(desired_period - dt)
    # mark this frame
    
    if k == ord('s'):
        if not sampling_active:
            sampling_active = True
            sampling_start = frame_idx
            colCirc = (255,165,0)  # orange while sampling active
        else:
            sampling_active = False
            if sampling_start is not None and frame_idx >= sampling_start:
                sampling_bouts.append((sampling_start, frame_idx))
                print(f"Recorded sampling bout: ({sampling_start}, {frame_idx})")
            sampling_start = None
            colCirc = (255,255,255)
    
    if k == ord('x') or k == 8:  # 'x' or Backspace
            if sampling_active:
                print(f"Cancelled active sampling bout starting at {sampling_start}")
                sampling_active = False
                sampling_start = None
                colCirc = (255,255,255)
            elif sampling_bouts:
                removed = sampling_bouts.pop()
                print(f"Removed last sampling bout: {removed}")
            else:
                print("No sampling bout to remove.")
        
    # PAUSE/RESUME on 'p' or Space
    if k in (ord('p'), 32):
        while True:
            frame_paused = frame.copy()
            cv2.rectangle(frame_paused, (10, 10), (140, 40), (0, 0, 0), -1)
            cv2.putText(frame_paused, "PAUSED", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow('frame', frame_paused)
            k2_raw = cv2.waitKey(50)
            k2 = k2_raw & 0xFF
            if k2 in (ord('p'), 32):  # resume
                break
            if k2 == ord('q'):        # quit from pause
                print("You quit! Restart the annotations by running this script again!")
                saveAnnotations = True
                quit_now = True
                break
        if quit_now:
            break
        # continue outer loop to read next frame
        continue

    if k == ord('q'):
        print("You quit! Restart the annotations by running this script again!")
        saveAnnotations = True
        break

    annotation_list.append(current_state)

cap.release()
cv2.destroyAllWindows()

if saveAnnotations:
    np.save(outpath,
            np.array(sampling_bouts, dtype=np.int32))

    # sampling fraction (frames under sampling / total frames)
    samp_frames = sum((end - start + 1) for start, end in sampling_bouts)
    frac = (samp_frames / float(max(total_frames, 1))) if total_frames > 0 else 0.0
    print(f"Sampling bouts: {len(sampling_bouts)}; total frames: {samp_frames}; fraction: {frac:.4f}")
    print("Annotations saved successfully!")

