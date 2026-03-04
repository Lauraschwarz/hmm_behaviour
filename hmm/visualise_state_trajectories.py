import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.colors as mcolors



import cv2

def close_state_gaps(df, max_gap):
    states = df['states'].values
    states = np.asarray(states).copy()
    if len(states) == 0:
        return states
    i = 0
    while i < len(states):
        # Start of a run
        j = i + 1
        while j < len(states) and states[j] == states[i]:
            j += 1

        # states[i:j] is one run of states[i]
        # Look ahead for a short gap
        if j < len(states):
            gap_start = j
            gap_state = states[j]

            k = gap_start + 1
            while k < len(states) and states[k] == gap_state:
                k += 1

            gap_len = k - gap_start

            # Check if flanked by same state
            if k < len(states) and states[k] == states[i] and gap_len <= max_gap:
                states[gap_start:k] = states[i]
                # Do NOT advance i — re-evaluate merged run
                continue
        i = j

    return states    



def extract_state_overlay_videos(
    path,
    video_path,
    session_id,
    model_type,
    out_dir,
    tile_width=320,
    min_run_frames=25,
    pre_seconds=0.5,
    post_seconds=4.0,
    align_to="onset",          # "onset" | "midpoint" | "offset"
    blend_mode="mean",         # "mean" | "median" | "max"
    clip_alpha=1.0,
    gamma=0.5,
    bg_mode="median",
    bg_samples=150,
    use_fg_mask=True,
    mask_thresh=20,
    invert=True,
    K=5,
):
    """
    Produce ONE overlay (crowd) video per state.
    Each occurrence contributes a fixed temporal window aligned to a chosen anchor.
    """

    assert align_to in {"onset", "midpoint", "offset"}

    path = Path(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path / "model_predictions" / model_type / f"all_session_data_{K}states.csv")
    df = df[df["session_id"] == session_id].copy()
    if df.empty:
        raise RuntimeError(f"No data for session {session_id}")

    df.reset_index(drop=True, inplace=True)
    df = df[df['states_confidence'] >= 0.5].copy()
    states = df["states_smoothed"]
    unique_states = sorted(states.unique())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    drop_start = max(0, total_frames - len(df))
    aligned_total = total_frames - drop_start

    cap.set(cv2.CAP_PROP_POS_FRAMES, drop_start)
    ok, fr0 = cap.read()
    if not ok:
        raise RuntimeError("Could not read video")

    h0, w0 = fr0.shape[:2]
    tile_height = int(h0 * tile_width / w0)

    pre_frames = int(round(pre_seconds * fps))
    post_frames = int(round(post_seconds * fps))
    total_window = pre_frames + post_frames + 1

    def read_frame(pos):
        if pos < 0 or pos >= aligned_total:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, drop_start + int(pos))
        ok, fr = cap.read()
        if not ok:
            return None
        return cv2.resize(fr, (tile_width, tile_height))

    # ---------- background ----------
    def compute_bg():
        if bg_mode == "none":
            return None
        idxs = np.linspace(
            0, aligned_total - 1, min(bg_samples, aligned_total)
        ).astype(int)
        frames = []
        for i in idxs:
            fr = read_frame(i)
            if fr is not None:
                frames.append(fr.astype(np.float32))
        if not frames:
            return None
        return np.median(np.stack(frames), axis=0)

    bg = compute_bg()
    kernel = np.ones((3, 3), np.uint8)

    def process(fr):
        img = fr.astype(np.float32)
        if bg is not None:
            img = cv2.absdiff(img, bg)

        weights = None
        if use_fg_mask:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, mask_thresh, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            weights = (mask / 255.0).astype(np.float32)[..., None]
            img *= weights

        if invert:
            img = 255.0 - img

        return img, weights

    def find_runs(state):
        idx = np.flatnonzero(states.values == state)
        if idx.size == 0:
            return []
        splits = np.where(np.diff(idx) != 1)[0] + 1
        runs = np.split(idx, splits)
        return [r for r in runs if len(r) >= min_run_frames]

    def anchor_from_run(run):
        if align_to == "onset":
            return run[0]
        elif align_to == "midpoint":
            return run[len(run) // 2]
        elif align_to == "offset":
            return run[-1]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for s in unique_states:
        runs = find_runs(s)
        if not runs:
            continue

        windows = []
        for r in runs:
            anchor = anchor_from_run(r)
            start = anchor - pre_frames
            end = anchor + post_frames + 1
            window = np.arange(start, end)
            windows.append(window)

        out_p = out_dir / model_type / "state_overlays" 
        out_p.mkdir(parents=True, exist_ok=True)
        out_path = out_p / f"overlay_state_{s}_{session_id}_{K}_states_{align_to}_{min_run_frames}.mp4"

        writer = cv2.VideoWriter(
            str(out_path), fourcc, fps, (tile_width, tile_height)
        )

        last_frames = [None] * len(windows)

        for t in range(total_window):
            frames, weights = [], []

            for i, w in enumerate(windows):
                if t < len(w):
                    fr = read_frame(w[t])
                    if fr is not None:
                        last_frames[i] = fr
                else:
                    fr = last_frames[i]

                if fr is None:
                    continue

                img, wgt = process(fr)
                frames.append(img)
                if use_fg_mask:
                    weights.append(
                        wgt if wgt is not None
                        else np.ones((*img.shape[:2], 1), np.float32)
                    )

            if not frames:
                out = np.zeros((tile_height, tile_width, 3), np.uint8)
            else:
                stack = np.stack(frames)
                if blend_mode == "mean":
                    if use_fg_mask:
                        W = np.stack(weights) * clip_alpha
                        out = (stack * W).sum(0) / np.clip(W.sum(0), 1e-6, None)
                    else:
                        out = stack.mean(0)
                elif blend_mode == "median":
                    out = np.median(stack, axis=0)
                elif blend_mode == "max":
                    out = np.max(stack, axis=0)
                else:
                    raise ValueError(blend_mode)

                out = np.clip(out, 0, 255).astype(np.uint8)

            if gamma:
                f = (out / 255.0) ** gamma
                out = np.clip(f * 255, 0, 255).astype(np.uint8)

            writer.write(out)

        writer.release()
        print(f"Saved overlay for state {s}: {out_path}")

    cap.release()


extract_state_overlay_videos(
    path=r"F:\social_sniffing\derivatives\1106077\social\CTRL",
    video_path=r"F:\social_sniffing\rawdata\1106077\social\CTRL\2025-09-23T12-31-23\Video\1106077.avi",
    session_id="session_2",
    tile_width=320,
    model_type='gaussian_raw',

    
    out_dir=r"F:\social_sniffing\derivatives\1106077\social\CTRL\model_predictions", 
    min_run_frames=50, 
    K=5
)
#add_states_to_df(Path(r"F:\social_sniffing\derivatives\1106077"))


# def extract_state_overlay_videos(
#     path,
#     video_path,
#     session_id,
#     out_dir,
#     tile_width=320,
#     min_run_frames=25,
#     pre_seconds=0.0,
#     state_duration_seconds=2.0,   # how long each occurrence contributes
#     blend_mode="mean",            # "mean", "median", "max"
#     clip_alpha=1.0,
#     gamma=0.5,
#     bg_mode="median",
#     bg_samples=150,
#     use_fg_mask=True,
#     mask_thresh=20,
#     invert=True,
#     K=5
# ):
#     """
#     Produce ONE overlay (crowd) video per state.
#     Each occurrence contributes a fixed temporal window aligned to state onset.
#     """

#     path = Path(path)
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     df = pd.read_csv(path / f"all_session_data_v3_with_{K}states_ARHMM.csv")
#     df = df[df["olfactory_stim"] == session_id].copy()
#     if df.empty:
#         raise RuntimeError(f"No data for session {session_id}")

#     df.reset_index(drop=True, inplace=True)

#     states = df["states"]
#     unique_states = sorted(states.unique())

#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise FileNotFoundError(video_path)

#     fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     drop_start = max(0, total_frames - len(df))
#     aligned_total = total_frames - drop_start

#     cap.set(cv2.CAP_PROP_POS_FRAMES, drop_start)
#     ok, fr0 = cap.read()
#     if not ok:
#         raise RuntimeError("Could not read video")

#     h0, w0 = fr0.shape[:2]
#     tile_height = int(h0 * tile_width / w0)

#     pre_frames = int(round(pre_seconds * fps))
#     state_frames = int(round(state_duration_seconds * fps))
#     total_window = pre_frames + state_frames

#     def read_frame(pos):
#         if pos < 0 or pos >= aligned_total:
#             return None
#         cap.set(cv2.CAP_PROP_POS_FRAMES, drop_start + pos)
#         ok, fr = cap.read()
#         if not ok:
#             return None
#         return cv2.resize(fr, (tile_width, tile_height))

#     # ---------- background ----------
#     def compute_bg():
#         if bg_mode == "none":
#             return None
#         idxs = np.linspace(0, aligned_total - 1,
#                            min(bg_samples, aligned_total)).astype(int)
#         frames = [read_frame(i).astype(np.float32)
#                   for i in idxs if read_frame(i) is not None]
#         return np.median(np.stack(frames), axis=0) if frames else None

#     bg = compute_bg()
#     kernel = np.ones((3, 3), np.uint8)

#     def process(fr):
#         img = fr.astype(np.float32)
#         if bg is not None:
#             img = cv2.absdiff(img, bg)

#         weights = None
#         if use_fg_mask:
#             gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
#             _, mask = cv2.threshold(gray, mask_thresh, 255, cv2.THRESH_BINARY)
#             mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#             weights = (mask / 255.0).astype(np.float32)[..., None]
#             img *= weights

#         if invert:
#             img = 255.0 - img

#         return img, weights

#     def find_runs(state):
#         idx = np.flatnonzero(states.values == state)
#         if idx.size == 0:
#             return []
#         splits = np.where(np.diff(idx) != 1)[0] + 1
#         runs = np.split(idx, splits)
#         return [r for r in runs if len(r) >= min_run_frames]

#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")

#     for s in unique_states:
#         runs = find_runs(s)
#         if not runs:
#             continue

#         windows = []
#         for r in runs:
#             start = max(0, r[0] - pre_frames)
#             end = min(start + total_window, aligned_total)
#             windows.append(np.arange(start, end))

#         out_path = out_dir / f"overlay_state_{s}_{session_id}_{K}_states_diagonal.mp4"
#         writer = cv2.VideoWriter(str(out_path), fourcc, fps,
#                                  (tile_width, tile_height))

#         last_frames = [None] * len(windows)

#         for t in range(max(len(w) for w in windows)):
#             frames, weights = [], []

#             for i, w in enumerate(windows):
#                 if t < len(w):
#                     fr = read_frame(w[t])
#                     if fr is not None:
#                         last_frames[i] = fr
#                 else:
#                     fr = last_frames[i]

#                 if fr is None:
#                     continue

#                 img, wgt = process(fr)
#                 frames.append(img)
#                 if use_fg_mask:
#                     weights.append(wgt if wgt is not None
#                                    else np.ones((*img.shape[:2], 1)))

#             if not frames:
#                 out = np.zeros((tile_height, tile_width, 3), np.uint8)
#             else:
#                 stack = np.stack(frames)
#                 if blend_mode == "mean":
#                     if use_fg_mask:
#                         W = np.stack(weights) * clip_alpha
#                         out = (stack * W).sum(0) / np.clip(W.sum(0), 1e-6, None)
#                     else:
#                         out = stack.mean(0)
#                 elif blend_mode == "median":
#                     out = np.median(stack, axis=0)
#                 elif blend_mode == "max":
#                     out = np.max(stack, axis=0)
#                 else:
#                     raise ValueError(blend_mode)

#                 out = np.clip(out, 0, 255).astype(np.uint8)

#             if gamma:
#                 f = (out / 255.0) ** gamma
#                 out = np.clip(f * 255, 0, 255).astype(np.uint8)

#             writer.write(out)

#         writer.release()
#         print(f"Saved overlay for state {s}: {out_path}")

#     cap.release()