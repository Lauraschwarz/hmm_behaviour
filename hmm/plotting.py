import pathlib
from turtle import up
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import pickle
import sys
from movement.roi import PolygonOfInterest
from edited_movement_func import plot_centroid_trajectory, plot_centroid_trajectory_by_states
import umap
import ultraplot as up 
from constants import ARENA, ARENA_VERTICES, inner_vertices, TRIAL_LENGTH
sys.path.append(r'C:\Users\Laura\social_sniffing\sniffies')
import session
from global_functions import get_barrier_open_time
from sklearn.preprocessing import StandardScaler
from ultra import get_exploration_and_signal_grid
from movement.utils.vector import compute_norm
from params import hmm_parameters_object
from hmm_utils import (
    load_pickle_file, extract_frame_range, 
    get_param_string, interpolate_circular_nan
)
from scipy.ndimage import label
from scipy.signal import find_peaks

cmp = sns.diverging_palette(220, 20, as_cmap=True)
cmp2 = sns.color_palette("tab20", as_cmap=True) 

MICE = ['1106077', '1106009', '1106010', '1106008', '1106078', '1106079', '1125563', '1125561','1125555','1125131', '1125132']
VGAT_MICE = ['1106077', '1106078', '1106079']
METHIMAZOLE_SESSIONS = {'1106077': '2025-10-02T14-11-03',
                        '1106009': '2025-10-02T13-33-33',
                        '1106010': '2025-09-26T12-42-23',
                        '1106078': '2025-09-26T13-21-09',
                        '1106079': '2025-10-02T14-45-26',
                        '1125555': '2025-08-27T14-43-52'
                        }
AVERSIVE_SESSIONS = {'1106079': '2025-09-29T13-38-32',
                     '1106078': '2025-09-22T12-52-28',
                     '1106077': '2025-09-29T13-07-16'}

GENOTYPES = {'1106077': 'VGAT',
                        '1106009': 'VGLUT2',
                        '1106008': 'VGLUT2',
                        '1106010': 'VGLUT2',
                        '1106078': 'VGAT',
                        '1106079': 'VGAT',
                        '1125563': 'DLIGHT',
                        '1125131': 'OT',
                    '1125132': 'OT',
                    '1125555': 'DLIGHT',
                    '1125561': 'DLIGHT'
                        }
REARING = {'1106077': r'F:\social_sniffing\derivatives\1106077\2025-09-23T12-31-23\Video\rearing.npy',
           '1106078': r'F:\social_sniffing\derivatives\1106078\2025-09-17T12-53-03\Video\rearing.npy',
              '1106079': r'F:\social_sniffing\derivatives\1106079\2025-09-23T13-06-24\Video\rearing.npy',
              '1125131': r'F:\social_sniffing\derivatives\1125131\2025-07-01T12-47-28\Video\rearing.npy',
              '1125132': r'F:\social_sniffing\derivatives\1125132\2025-07-01T13-25-31\Video\rearing.npy',
              '1125563': r'F:\social_sniffing\derivatives\1125563\2025-09-08T13-06-32\Video\rearing.npy',
              '1106008': r'F:\social_sniffing\derivatives\1106008\2025-09-23T13-41-34\Video\rearing.npy',
                '1106009': r'F:\social_sniffing\derivatives\1106009\2025-09-23T14-16-08\Video\rearing.npy',
                '1106010': r'F:\social_sniffing\derivatives\1106010\2025-09-17T12-23-11\Video\rearing.npy'
                       }
SAMPLING = {'1106077': r'F:\social_sniffing\derivatives\1106077\2025-09-23T12-31-23\Video\sampling.npy',
           '1106078': r'F:\social_sniffing\derivatives\1106078\2025-09-17T12-53-03\Video\sampling.npy',
              '1106079': r'F:\social_sniffing\derivatives\1106079\2025-09-23T13-06-24\Video\sampling.npy',
              '1125131': r'F:\social_sniffing\derivatives\1125131\2025-07-01T12-47-28\Video\sampling.npy',
                '1125132': r'F:\social_sniffing\derivatives\1125132\2025-07-01T13-25-31\Video\sampling.npy',
                '1125563': r'F:\social_sniffing\derivatives\1125563\2025-09-08T13-06-32\Video\sampling.npy',
                '1106008': r'F:\social_sniffing\derivatives\1106008\2025-09-23T13-41-34\Video\sampling.npy',
                '1106009': r'F:\social_sniffing\derivatives\1106009\2025-09-23T14-16-08\Video\sampling.npy',
                '1106010': r'F:\social_sniffing\derivatives\1106010\2025-09-17T12-23-11\Video\sampling.npy'
                       }
CHASE = {'1106077': r'F:\social_sniffing\derivatives\1106077\2025-09-23T12-31-23\Video\chase.npy',
           '1106078': r'F:\social_sniffing\derivatives\1106078\2025-09-17T12-53-03\Video\chase.npy',
              '1106079': r'F:\social_sniffing\derivatives\1106079\2025-09-23T13-06-24\Video\chase.npy',
              '1125131': r'F:\social_sniffing\derivatives\1125131\2025-07-01T12-47-28\Video\chase.npy',
                '1125132': r'F:\social_sniffing\derivatives\1125132\2025-07-01T13-25-31\Video\chase.npy',
                '1125563': r'F:\social_sniffing\derivatives\1125563\2025-09-08T13-06-32\Video\chase.npy',
                '1106008': r'F:\social_sniffing\derivatives\1106008\2025-09-23T13-41-34\Video\chase.npy',
                '1106009': r'F:\social_sniffing\derivatives\1106009\2025-09-23T14-16-08\Video\chase.npy',
                '1106010': r'F:\social_sniffing\derivatives\1106010\2025-09-17T12-23-11\Video\chase.npy'
                       }


def add_signal_peaks_to_df(df):
    peaks = find_peaks(df['deltaF_z'], distance=25, prominence=2.5)
    df['signal_peaks'] = np.full(len(df), np.nan)
    df['signal_peaks'][peaks[0]]=1
    return df





def extract_approach_epochs(
    df,
    far_threshold=100,       # mouse must start > this distance away
    near_threshold=50,      # episode ends if mouse is near port
    speed_threshold=50,     # minimum speed for movement
    accel_threshold=150,   # minimum acceleration for initiation
    min_delta_distance=50,  # minimum total distance reduction
    min_duration=50      # seconds (if df has a 'time' column)
):

    if "time" in df.columns:
        dt = float(df["time"].diff().median())
    else:
        dt = 1  # fall back to 1-sample units
        df = df.copy()
        df["time"] = np.arange(len(df))

    df = df.copy()
    df["dist0_vel"] = df["abdomen_port0"].diff()
    df["dist1_vel"] = df["abdomen_port1"].diff()

   
    def extract_for_port(dist, dist_vel, poke, port_id):

        candidate = (
            (dist > near_threshold) &
            (df.smoothed_speed > speed_threshold) &
            (df.smoothed_acceleration > accel_threshold) &
            (dist_vel < 0)  # moving toward port
        )

        labels, num = label(candidate)

        approach_list = []

        for label_id in range(1, num + 1):
            idxs = np.where(labels == label_id)[0]
            start_idx = idxs[0]
            end_idx   = idxs[-1]

            if (df.time[end_idx] - df.time[start_idx]) < min_duration:
                continue

            if dist.iloc[start_idx] < far_threshold:
                continue

            delta_dist = dist.iloc[start_idx] - dist.iloc[end_idx]
            if delta_dist < min_delta_distance:
                continue

            termination = "stop"
            if dist.iloc[end_idx] < near_threshold:
                termination = "near_port"
            end_segment = df.iloc[start_idx:end_idx+1]
            if end_segment[poke].any():
                termination = poke

            stim_before = df.loc[:start_idx, "stim_onset"]
            if stim_before.any():
                stim_idx = stim_before[stim_before != 0].index[-1]
            else:
                stim_idx = None

           
            approach_list.append({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_time": float(df.time[start_idx]),
                "end_time": float(df.time[end_idx]),
                "delta_distance": float(delta_dist),
                "terminated_by": termination,
                "stim_idx": stim_idx,
                "port": port_id
            })

        return approach_list


    approaches_port0 = extract_for_port(
        df["abdomen_port0"], df["dist0_vel"], "poke0", 0
    )

    approaches_port1 = extract_for_port(
        df["abdomen_port1"], df["dist1_vel"], "poke1", 1
    )

    return approaches_port0, approaches_port1



def plot_aligned_deltaF(
    df, 
    outpath,
    approaches_port0,
    approaches_port1,
    filter_by="any_poke",          # filtering option
    window_pre=50,             # frames before approach
    window_post=500,           # frames after approach
    deltaF_col="deltaF_z"
):
    """
    Plot deltaF_z traces aligned to approach start, with optional filtering.

    filter_by options:
        "none"       → no filtering (plot all)
        "poke0"      → only approaches terminated by poke0
        "poke1"      → only approaches terminated by poke1
        "any_poke"   → poke0 or poke1
        "reward"     → reward0 or reward1 occurs after approach
        custom_fn    → user-supplied function(ap) returning True/False
    """

    # Combine approaches from both ports
    all_approaches = approaches_port0 + approaches_port1

    # -------------------------------------------------------
    # Filtering logic
    # -------------------------------------------------------
    def passes_filter(ap):
        # User supplied function
        if callable(filter_by):
            return filter_by(ap)

        term = ap.get("terminated_by", "")

        if filter_by == "none":
            return True

        if filter_by == "poke0":
            return term == "poke0"

        if filter_by == "poke1":
            return term == "poke1"

        if filter_by == "any_poke":
            return ("poke0" in term) or ("poke1" in term)

        if filter_by == "reward":
            # reward after the approach start
            start = ap["start_idx"]
            seg = df.iloc[start : start + window_post]
            return seg["reward"].any()

        return True

    # Apply filtering
    selected_approaches = [ap for ap in all_approaches if passes_filter(ap)]

    if len(selected_approaches) == 0:
        print("No approaches satisfy the filtering condition.")
        return

    # -------------------------------------------------------
    # Extract aligned traces
    # -------------------------------------------------------
    traces = []
    for ap in selected_approaches:
        start = ap["start_idx"]

        start_idx = start - window_pre
        end_idx   = start + window_post

        # Check bounds
        if start_idx < 0 or end_idx >= len(df):
            continue

        trace = df.iloc[start_idx:end_idx][deltaF_col].values
        traces.append(trace)

    if len(traces) == 0:
        print("No traces within bounds for plotting.")
        return

    traces = np.array(traces)
    mean_trace = traces.mean(axis=0)
    time_axis = np.arange(-window_pre, window_post)

    # -------------------------------------------------------
    # Plotting
    # -------------------------------------------------------
    plt.figure(figsize=(10, 6))

    # gray individual traces
    for tr in traces:
        plt.plot(time_axis, tr, color="gray", alpha=0.3, linewidth=1)

    # mean trace in black
    plt.plot(time_axis, mean_trace, color="black", linewidth=2)

    # vertical line at approach start
    plt.axvline(0, linewidth=2)

    plt.xlabel("Frames relative to approach start")
    plt.ylabel(deltaF_col)
    plt.title(f"ΔF_z aligned to approach start\nFilter: {filter_by}")

    plt.savefig(outpath / f"aligned_deltaF_{filter_by}.svg", dpi=300)



def reindex_movement_data(track, harp_start):
    time_difference = (track.start - harp_start).total_seconds()
    n_neg_frames = int(np.ceil(time_difference * track.frame_rate))
    neg_times = np.linspace(-time_difference, 0, n_neg_frames, endpoint=False)
    full_times = np.concatenate([neg_times, track.ds.time.values])
    track_ds = track.ds.reindex(time=full_times, fill_value=np.nan)
    return track_ds

def load_roi(name):
    roi = PolygonOfInterest(ARENA[name], name='name')
    return roi

def plot_feature(ax, trace, trace_name=None):
    ax.plot(trace.index, trace)
    ax.scatter(trace.index, trace, s=1, zorder=100, cmap=cmp)
    ax.format(
        ylabel=f'{trace_name}',
        xlabel='Time (s)'
    )
def separate_df_by_states(state, df):
    mask = df['states'] == state
    run_id = mask.ne(mask.shift()).cumsum()

    df5 = df[mask].copy()
    df5['run'] = run_id[mask].values

    runs = [g for _, g in df5.groupby('run') ]
    runs = [g.reset_index(drop=True) for g in runs]

    return runs

def plot_stim_onset_reward_split(hmm_array, outpath=None, onset_col='stim_onset'):
    stim_times = hmm_array.index[hmm_array[onset_col] == 1]

    if len(stim_times) == 0:
        return

    reward_window = 500   # 10 seconds at 50 Hz

    rewarded = []
    unrewarded = []

    for t in stim_times:
        reward_segment = hmm_array.loc[t : t + reward_window, 'reward']

        start = t - 100
        end   = t + 100

        if start < 0:
            continue

        seg = hmm_array.loc[start:end, 'deltaF_z']

        if reward_segment.sum() > 0:
            rewarded.append(pd.Series(seg.values))
        else:
            unrewarded.append(pd.Series(seg.values))

    rewarded_df = pd.concat(rewarded, axis=1) if len(rewarded) > 0 else None
    unrewarded_df = pd.concat(unrewarded, axis=1) if len(unrewarded) > 0 else None

    fig, ax = plt.subplots(figsize=(10, 5))

    if unrewarded_df is not None:
        for col in unrewarded_df.columns:
            ax.plot(unrewarded_df.index,
                    unrewarded_df[col],
                    color=cmp(0.3),
                    alpha=0.1)
        ax.plot(unrewarded_df.index,
                unrewarded_df.mean(axis=1),
                color=cmp(0),
                linewidth=3,
                label='Unrewarded (mean)')

    if rewarded_df is not None:
        for col in rewarded_df.columns:
            ax.plot(rewarded_df.index,
                    rewarded_df[col],
                    color=cmp(1.8),
                    alpha=0.1)
        ax.plot(rewarded_df.index,
                rewarded_df.mean(axis=1),
                color=cmp(1.1),
                linewidth=3,
                label='Rewarded (mean)')

    ax.axvline(100, color='k', linestyle='--')

    ax.set_xlabel('Samples (relative to stim)')
    ax.set_ylabel('ΔF_z')
    ax.set_title('ΔF_z aligned to stim onset: Rewarded vs Unrewarded')
    ax.legend()

    plt.tight_layout()
    ax.set_ylim(-2, 5)
    ax.set_xlim(50, 200)


    plt.savefig(outpath / 'deltaF_z_rewarded_vs_unrewarded.svg', dpi=300)


def plot_delta_f_stim_onset_split_distance(hmm_array, outpath, distance='snout_groin', distance_threshold=50, column='stim_onset'):
    stim_times = hmm_array.index[hmm_array[column] == 1]

    aligned_above = []
    aligned_below = []

    if len(stim_times) == 0:
        return
    
    for t in stim_times:
        start = t - 100
        end = t + 400

        seg = hmm_array.loc[start:end, 'deltaF_z']
        # seg_reward = hmm_array.loc[start:end, 'reward']

        # if seg_reward.sum() < 1:
        #     continue
        dist_at_stim = hmm_array.loc[t, distance]

        if dist_at_stim > distance_threshold:
            aligned_above.append(pd.Series(seg.values))
        else:
            aligned_below.append(pd.Series(seg.values))
    
    aligned_above_df = pd.concat(aligned_above, axis=1) if len(aligned_above) > 0 else None
    aligned_below_df = pd.concat(aligned_below, axis=1) if len(aligned_below) > 0 else None
    print(f'Number of trials above threshold: {aligned_above_df.shape[1] if aligned_above_df is not None else 0}')
    print(f'Number of trials below threshold: {aligned_below_df.shape[1] if aligned_below_df is not None else 0}')

    fig, ax = plt.subplots(figsize=(8, 4))

    if aligned_below_df is not None:
        for i, col in enumerate(aligned_below_df.columns):
            ax.plot(aligned_below_df.index, aligned_below_df[col], color=cmp(0.3), alpha=0.2)
        aligned_below_df.mean(axis=1).plot(color=cmp(0), linewidth=3, ax=ax, label=f'Distance <= {distance_threshold}px')

    if aligned_above_df is not None:
        for i, col in enumerate(aligned_above_df.columns):
            ax.plot(aligned_above_df.index, aligned_above_df[col], color=cmp(1.8), alpha=0.2)
        aligned_above_df.mean(axis=1).plot(color=cmp(1.1), linewidth=3, ax=ax, label=f'Distance > {distance_threshold}px')

    ax.axvline(100, color='k', linestyle='--')

    ax.set_xlabel('Samples (relative to stim)')
    ax.set_ylabel('ΔF_z')
    ax.legend()
    ax.set_ylim(-2, 5)
    ax.set_xlim(50,200)

    plt.tight_layout()
    plt.savefig(outpath / f'deltaF_z_aligned_to_stim_onset_for_reward_distance_split_{distance_threshold}px.svg', dpi=300)


def plot_deltaF(hmm_array, outpath, column='stim_onset', signal_col='deltaF_z'):

    stim_times = hmm_array.index[hmm_array[column] == 1]

    aligned = []
    if len(stim_times) == 0:
        return
    
    for t in stim_times:
        start = t - 50
        end = t + 100

        seg = hmm_array.loc[start:end, signal_col]
        b = min(25, len(seg))
        base = seg.iloc[:b].mean() if b > 0 else 0.0
        y = seg.values - base
        aligned.append(pd.Series(y))
    aligned_df = pd.concat(aligned, axis=1)
    aligned_df.columns = [f'trial_{i}' for i in range(len(aligned_df.columns))]

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, col in enumerate(aligned_df.columns):
        ax.plot(aligned_df.index, aligned_df[col], color='gray', alpha=0.15)

    aligned_df.mean(axis=1).plot(color='black', linewidth=3, ax=ax)

    ax.axvline(50, color='red', linestyle='--')

    ax.set_xlabel('Samples (relative to stim)')
    ax.set_ylabel(f'{signal_col}')
    ax.set_ylim(-2, max(5, aligned_df.values.max() + 1))
    ax.grid(False)
    #ax.set_xlim(50,200)


    plt.tight_layout()
    plt.savefig(outpath / f'{signal_col}_aligned_to_{column}.svg', dpi=300)

def detect_interaction_bouts(df, distance_col='snout_groin', threshold=50, fps=50, merge_gap_seconds=0.5):
    merge_gap_frames = merge_gap_seconds * fps
    below = df[distance_col] < threshold

    bouts = []
    in_bout = False
    start = None

    for i, is_below in enumerate(below):
        if is_below and not in_bout:
            in_bout = True
            start = df.index[i]

        elif not is_below and in_bout:
            end = df.index[i-1]
            bouts.append((start, end))
            in_bout = False

    if in_bout:
        bouts.append((start, df.index[-1]))

    merged = []
    current_start, current_end = bouts[0]

    for next_start, next_end in bouts[1:]:
        # gap between bouts (in frames)
        gap = next_start - current_end

        if gap <= merge_gap_frames:
            # extend current bout
            current_end = next_end
        else:
            # push finished bout
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end))

    return merged


def return_interaction_approaches(hmm_array, min_neg_frames=50,
                                   search_back=200,
                                      search_forward=500,
                                        offset_before=100,
                                      distance_col='snout_groin'):
    stim_onsets = detect_interaction_bouts(hmm_array, distance_col=distance_col)
    approaches = []
    dist_vals_full = hmm_array[distance_col].values
    all_slopes = []
    for stim in stim_onsets:

        stim_idx = hmm_array.index.get_loc(stim[0])

        start_idx = max(0, stim_idx - search_back)

        distance_window = dist_vals_full[start_idx: stim_idx + 1]

        slope = np.diff(distance_window)
        all_slopes.append(slope)
        approach_start_local = start_idx  # fallback if no approach found
        if len(slope) < min_neg_frames:
            continue

        found = False
        for j in range(0, len(slope) - min_neg_frames + 1):
            if np.all(slope[j : j + min_neg_frames] < 0):
        
                approach_start_local = start_idx + j
                found = True
                break
        
        if not found:
            continue

        approach_start_global = max(0, approach_start_local - offset_before)

        end_idx = stim_idx + search_forward 

        seg = hmm_array.iloc[approach_start_global : end_idx].copy()
        seg['orig_index'] = seg.index
        seg = seg.reset_index(drop=True)

        approaches.append(seg)

    if len(approaches) == 0:
        print("No approaches found.")
        return [], []   
    max_len = max(len(seg) for seg in approaches)
    aligned = pd.DataFrame(index=np.arange(max_len))
    return approaches, aligned


def add_approaches_to_df(hmm_array, 
                                     
                                     deltaF_col='deltaF_z',
                                     reward_col='reward',
                                     stim_col='stim_onset',
                                     min_neg_frames=25,
                                     stim_buffer=50,
                                     search_forward=500,
                                     offset_before=100):

    distance = ['abdomen_port0', 'abdomen_port1']
    hmm_array['reward_approach_onset'] = np.nan

    #hmm_array['unrewarded_approach_onset'] = np.nan ##TODO figure out how to implement this

    stim_onsets = hmm_array.index[hmm_array[stim_col] == 1]

    dist_vals_full = []
    for i in distance:
        dist_vals_full.append(hmm_array[i].values)

    all_slopes = []
    for stim in stim_onsets:

        stim_idx = hmm_array.index.get_loc(stim)

        reward_window = hmm_array.loc[stim : stim + search_forward, reward_col]

        if reward_window.sum() == 0:

            continue

        reward_idx = int(reward_window.index[reward_window == 1][0])
        start_idx = max(0, stim_idx )
        for i in range(len(distance)):
            distance_window = dist_vals_full[i][start_idx: reward_idx + 1]

            slope = np.diff(distance_window)
            all_slopes.append(slope)
            #approach_start_local = start_idx  # fallback if no approach found
            if len(slope) < min_neg_frames:
                continue
            found = False

            j = len(slope) - 1
            while j >= 0 and slope[j] < 0:
                j -= 1
                approach_start_local = start_idx + (j + 1)
                hmm_array.loc[approach_start_local, 'reward_approach_onset'] = 1
                found = True
            if not found:
                continue
         
    return 


def plot_approach_trajectory(hmm_array, centroid_pos, outpath):
    approaches, _ = add_approaches_to_df(hmm_array)

    if approaches == []:
        return
    
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, seg in enumerate(approaches):
        start_time = seg['orig_index'].iloc[0]
        end_time = seg['orig_index'].iloc[-1]
        centroid_segment = centroid_pos.sel(time=slice(start_time/50, end_time/50))
        plot_centroid_trajectory(
                        centroid_segment.position,
                        keypoints='abdomen',
                        manual_color_var='deltaF_z',
                        linestyle="-",
                        marker=".",
                        s=3,
                        cmap=cmp,
                        suppress_colorbar=True,
                        ax=ax
                    )
    ax.set_title('Approach trajectories')
    ax.set_ylim(-2, 5)


    plt.tight_layout()
    #plt.show()
    #plt.savefig(outpath / f'approach_trajectories.svg', dpi=300)

    return

def plot_normalised_approach_reward(hmm_array, outpath, deltaF_col='deltaF_z', offset=100):
    # here the offset is to include some frames after reward collection
    #this does not really make much sense atm but leaving it for now
    approaches, aligned = add_approaches_to_df(hmm_array, deltaF_col=deltaF_col)

    if approaches == []:
        return
    new_approaches = []
    max_len = max(len(seg) for seg in approaches)
    for i, approach in enumerate(approaches):
        end_idx = np.where(approaches[i]['reward']==1)[0][0]
        start_idx = 0
        values = approach[deltaF_col].values[start_idx:end_idx+offset]
        new_approaches.append(values)
    resampled = np.vstack([resample_to_n(v, n=200) for v in new_approaches])  # shape (n_trials, n)
    time_norm = np.linspace(0, 1, 200)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(resampled.shape[0]):
        ax.plot(time_norm, resampled[i, :], color='gray', alpha=0.4)
    ax.plot(time_norm, np.nanmean(resampled, axis=0), color='black', linewidth=2)
    ax.set_xlabel('Normalized time (approach onset to reward)') 
    ax.set_ylabel(deltaF_col)
    ax.set_ylim(-2, 5)

    plt.tight_layout()

def resample_to_n(v, n=200):
    valid = ~np.isnan(v)
    if valid.sum() < 2:
        return np.full(n, np.nan)
    x_old = np.linspace(0, 1, len(v))
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old[valid], v[valid])


def extract_pokes(hmm_array, distance=['abdomen_port0', 'abdomen_port1'], column='poke',distance_threshold=25):
    pokes = hmm_array.index[hmm_array[column] == 1]
    separated_pokes = []
    all_pokes = []
    first_poke = pokes[0]
    for poke in pokes:
        if poke - first_poke <= 250:
            continue
        separated_pokes.append(poke)
        first_poke = poke

    for poke_time in separated_pokes:
        for i in distance:
            if hmm_array[i][poke_time] < distance_threshold:
                all_pokes.append(poke_time)
    
    return all_pokes
    
def plot_poke(hmm_array, outpath):
    stim_times = extract_pokes(hmm_array)
    aligned = []

    if len(stim_times) == 0:
        return
    
    for t in stim_times:
        start = t - 100
        end = t + 100

        seg = hmm_array.loc[start:end, 'deltaF_z']
        aligned.append(pd.Series(seg.values))
    aligned_df = pd.concat(aligned, axis=1)
    aligned_df.columns = [f'trial_{i}' for i in range(len(aligned_df.columns))]

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, col in enumerate(aligned_df.columns):
        ax.plot(aligned_df.index, aligned_df[col], color='gray', alpha=0.4)

    aligned_df.mean(axis=1).plot(color='black', linewidth=2, ax=ax)

    ax.axvline(100, color='red', linestyle='--')

    ax.set_xlabel('Samples (relative to stim)')
    ax.set_ylabel('ΔF_z')
    ax.set_ylim(-2, 5)

    plt.tight_layout()
    plt.savefig(outpath / f'deltaF_z_aligned_to_poke.svg', dpi=300)

def plot_deltaf_vs_distance(hmm_array, ax=None, nbins=80, x_col='abdomen_abdomen', delta_col='deltaF_z'):
    df = hmm_array[[x_col, delta_col]].dropna()

    bins = np.linspace(df[x_col].min(), df[x_col].max(), nbins + 1)
    inds = np.digitize(df[x_col].values, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:])

    means = np.full(nbins, np.nan)
    sems  = np.full(nbins, np.nan)
    for i in range(nbins):
        sel = inds == i
        if sel.sum() > 0:
            vals = df[delta_col].values[sel]
            means[i] = vals.mean()
            sems[i] = vals.std(ddof=1) / np.sqrt(sel.sum())

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(centers, means, color='k', lw=1.5)
    ax.fill_between(centers, means - sems, means + sems, alpha=0.25, color='k')
    ax.set_xlabel(f'{x_col} (px)')
    ax.set_ylabel(delta_col)
    ax.grid(True)
    plt.savefig(outpath / f'deltaF_z_vs_{x_col}.svg', dpi=300)
    return 

def create_mouse_df(mouse_directory):
        all_files = [
        p for p in pathlib.Path(mouse_directory).rglob('*hmm_features.csv')
        if 'olfactory_ctrls' in (part.lower() for part in p.parts) and  'Methimazole' not in (part for part in p.parts)
    ]
        hmm_array_full = []
        for i, file in enumerate(all_files):
            print('processing file:', file)
            session_name = str(file.parent).split('\\')[4]
            sesh = session.Session(mouse_id=str(file.parent.parent).split('\\')[3], session_path=str(file.parent.parent))
            centroid_pos = xr.open_dataset(file.parent / 'tracking_data.nc')
            centroid_pos=centroid_pos.sel(keypoints='abdomen')
            hmm_array=pd.read_csv(file)
            #add_approaches_to_df(hmm_array)
            outpath = Path(file.parent / 'plots')
            outpath.mkdir(parents=True, exist_ok=True)
            hmm_array_full.append(hmm_array)
            hmm_array['session_id'] = f'session_{i+1}' #because  0 indexing
            if session_name == METHIMAZOLE_SESSIONS.get(sesh.mouse_id):
                hmm_array['condition'] = 'Methimazole'
            else:
                hmm_array['condition'] = 'Control'
            rearing_path = file.parent / 'Video' / 'rearing.npy'
            sampling_path = file.parent / 'Video' / 'sampling.npy'
            chase_path = file.parent / 'Video' / 'chase.npy'
            # grooming_path = file.parent / 'Video' / 'grooming.npy'
            # rearing = np.load(rearing_path, allow_pickle=True) 
            sampling = np.load(sampling_path, allow_pickle=True) 
            # chase = np.load(chase_path, allow_pickle=True) 
            #grooming = np.load(grooming_path, allow_pickle=True) 
            hmm_array['genotype'] = GENOTYPES.get(sesh.mouse_id)
            hmm_array['olfactory_stim'] = 'Aversive' if session_name == AVERSIVE_SESSIONS.get(sesh.mouse_id) else 'Appetitive'
            # hmm_array['rearing'] = np.full(len(hmm_array), np.nan)
            # hmm_array['rearing'][rearing]=1
            hmm_array['sampling'] = np.full(len(hmm_array), np.nan)
            hmm_array['sampling'][sampling]=1
            # hmm_array['chase'] = np.full(len(hmm_array), np.nan)
            # hmm_array['chase'][chase]=1
            #hmm_array['grooming'] = np.full(len(hmm_array), np.nan)
            #hmm_array['grooming'][grooming]=1
    
        combined_hmm_array = pd.concat(hmm_array_full, ignore_index=True)
        combined_hmm_array.to_csv(rf'{mouse_directory}/all_sessions_df_smells.csv', index=False)

        return combined_hmm_array

def create_all_mice_df(MICE):
    all_mice_dfs = []
    for mouse in MICE:
        mouse_df = create_mouse_df(rf'F:\social_sniffing\derivatives\{mouse}')
        mouse_df['mouse_id'] = mouse
        mouse_df = add_signal_peaks_to_df(mouse_df)
        all_mice_dfs.append(mouse_df)
    combined_all_mice_df = pd.concat(all_mice_dfs, ignore_index=True)
    combined_all_mice_df.to_csv(r'F:\social_sniffing\derivatives\all_mice_df_with_genotypes_olfactory_ctrls.csv', index=False)
    return combined_all_mice_df
create_all_mice_df(VGAT_MICE)
exit()
def bout_duration(bout):
    start, end = bout
    return (end - start + 1) / 50.0   # seconds, since fps=50

def plot_deltaF_first5_sampling_all(df_all, outpath, sampling_col='sampling',
                                    signal_col='deltaF_z', window_pre=100,
                                    window_post=500, n_events=5,
                                    baseline_frames=25):
    """
    For each genotype:
        - Use only session_1
        - Extract first n_events and last n_events sampling events per mouse
        - Align signal_col around each event (window_pre/window_post)
        - Baseline subtract (mean of first baseline_frames pre-event samples)
        - Plot:
            * first-event traces (all mice) in blue shades
            * last-event traces (all mice) in red shades
            * mean of all first traces (dark blue) and mean of all last traces (dark red)
    Saves one SVG per genotype.

    Requires columns: mouse_id, genotype, session_id, sampling_col, signal_col.
    """
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    needed_cols = {sampling_col, signal_col, 'mouse_id', 'genotype', 'session_id'}
    missing = needed_cols - set(df_all.columns)
    if missing:
        print(f"Missing columns: {missing}")
        return

    df = df_all.copy()
    df[sampling_col] = (df[sampling_col] == 1).astype(int)
    df_session = df[df['session_id'] == 'session_1']
    if df_session.empty:
        print("No data for session_1.")
        return

    n_samples = window_pre + window_post + 1
    time_axis = np.arange(-window_pre, window_post + 1)

    genotypes = df_session['genotype'].dropna().unique()

    for genotype in genotypes:
        sub = df_session[df_session['genotype'] == genotype]
        if sub.empty:
            continue

        first_trials = []
        last_trials = []

        # Collect trials per mouse
        for mouse_id, mouse_df in sub.groupby('mouse_id'):
            sampling_pos = np.flatnonzero(mouse_df[sampling_col].values == 1)
            if sampling_pos.size < (2 * n_events):
                # Require enough events to have distinct first & last sets
                continue

            first_positions = sampling_pos[:n_events]
            last_positions = sampling_pos[-n_events:]

            def extract_trials(positions):
                trials_local = []
                for pos in positions:
                    start_pos = pos - window_pre
                    end_pos = pos + window_post
                    if start_pos < 0 or end_pos >= len(mouse_df):
                        continue
                    seg = mouse_df.iloc[start_pos:end_pos + 1][signal_col].values
                    if seg.shape[0] != n_samples:
                        continue
                    trials_local.append(seg)
                return trials_local

            first_seg = extract_trials(first_positions)
            last_seg = extract_trials(last_positions)

            if len(first_seg) == n_events:
                first_trials.extend(first_seg)
            if len(last_seg) == n_events:
                last_trials.extend(last_seg)

        if not first_trials and not last_trials:
            print(f"No valid trials for genotype {genotype}.")
            continue

        # Baseline subtraction
        def baseline_subtract(arr_list):
            if not arr_list:
                return np.empty((0, n_samples))
            arr = np.vstack(arr_list)
            b_frames = min(baseline_frames, window_pre)
            if b_frames > 0:
                baselines = np.nanmean(arr[:, :b_frames], axis=1, keepdims=True)
                return arr - baselines
            return arr

        first_arr_bs = baseline_subtract(first_trials)
        last_arr_bs  = baseline_subtract(last_trials)

        # Colors
        blue_shades = sns.color_palette("Blues", n_events + 3)[2:2 + n_events]
        red_shades  = sns.color_palette("Reds", n_events + 3)[2:2 + n_events]

        fig, ax = plt.subplots(figsize=(7, 5))

        # Plot first-event traces
        if first_arr_bs.size > 0:
            # If aggregated across mice, ordering lost; just cycle colors
            for i, row in enumerate(first_arr_bs):
                ax.plot(time_axis, row, color=blue_shades[i % n_events],
                        alpha=0.3, linewidth=0.8)
            ax.plot(time_axis, np.nanmean(first_arr_bs, axis=0),
                    color='#0b3d91', linewidth=2.2, label=f'{genotype} first {n_events} mean')

        # Plot last-event traces
        if last_arr_bs.size > 0:
            for i, row in enumerate(last_arr_bs):
                ax.plot(time_axis, row, color=red_shades[i % n_events],
                        alpha=0.3, linewidth=0.8)
            ax.plot(time_axis, np.nanmean(last_arr_bs, axis=0),
                    color='#8B0000', linewidth=2.2, label=f'{genotype} last {n_events} mean')

        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('Frames (relative to sampling event)')
        ax.set_ylabel(f'{signal_col} (baseline-subtracted)')
        ymax = np.nanmax([first_arr_bs.max() if first_arr_bs.size else -2,
                          last_arr_bs.max() if last_arr_bs.size else -2])
        ax.set_ylim(-2, max(5, ymax + 1))
        ax.set_title(f'{signal_col}: first vs last {n_events} sampling events (session_1)\nGenotype: {genotype}')
        ax.legend(fontsize='small', ncol=1)
        plt.tight_layout()
        plt.savefig(outpath / f'{signal_col}_first_vs_last_sampling_session1_{genotype}.svg', dpi=300)
        plt.close(fig)

def plot_behaviour_seaborn():
    df = pd.read_csv(r'F:\social_sniffing\derivatives\all_mice_df_with_genotypes.csv')
    df = df.copy()
    df = df[df['mouse_id']!= 1125555]
    df = df[df['mouse_id']!= 1106009]
    df = df[df['mouse_id']!= 1125561]

    behaviour_mice = df[df["session_id"]!="session_2"]

    # per-session aggregates
    inter_df = (
        behaviour_mice.groupby(["mouse_id","session_id"])
          .apply(lambda g: pd.Series({
              "total_frames": len(g),
              "interaction_bouts": detect_interaction_bouts(g, distance_col='abdomen_abdomen', threshold=100, fps=50, merge_gap_seconds=0.5)
          }))
          .reset_index()
    )
    inter_df["interaction_seconds"] = inter_df["interaction_bouts"].apply(lambda bouts: sum(bout_duration(b) for b in bouts))
    inter_df["total_seconds"] = inter_df["total_frames"] / 50.0
    inter_df["percent_time_interacting"] = 100 * inter_df["interaction_seconds"] / inter_df["total_seconds"]

    event_counts = (
        behaviour_mice.groupby(["mouse_id","session_id"])
                 .agg(stim_count=("stim_onset", lambda x: (x == 1).sum()),
                      reward_count=("reward", lambda x: (x == 1).sum()))
                 .reset_index()
    )
    event_counts["percent_reward"] = 100 * event_counts["reward_count"] / event_counts["stim_count"]

    speed_df = (
        behaviour_mice.groupby(["mouse_id","session_id"])["smoothed_speed"]
                 .mean().reset_index()
    )

    dist_df = (
        behaviour_mice.groupby(["mouse_id","session_id"])["forward_displacement_exp"]
                 .sum().reset_index()
    )
    dist_df["distance_m"] = dist_df["forward_displacement_exp"] * 0.05866 / 100

    rewards_in_bout = []
    for (mouse, session), sub in behaviour_mice.groupby(["mouse_id","session_id"]):
        bouts = detect_interaction_bouts(sub, distance_col='abdomen_abdomen', threshold=100, fps=50, merge_gap_seconds=0.5)
        stim_onsets = sub.index[sub["stim_onset"] == 1]
        rewarded = sub.index[sub["reward"] == 1]
        pairs = [
            (stim, reward)
            for stim in stim_onsets
            for reward in rewarded[(rewarded > stim) & (rewarded <= stim + 500)]
        ]
        in_bout = [
            (stim, reward)
            for stim, reward in pairs
            if any(start <= stim <= end for start, end in bouts)
        ]
        rewards_in_bout.append({"mouse_id": mouse, "session_id": session, "count": len(in_bout)})
    bout_df = pd.DataFrame(rewards_in_bout)

    # plotting
    fig, ax = up.subplots(nrows=5, ncols=1, figsize=(3, 12), share=False, includepanels=True)
    pal =sns.color_palette("crest", as_cmap=True)
    sns.lineplot(data=inter_df, x="session_id", y="percent_time_interacting",
                 hue="mouse_id",  palette=pal,
                 marker="o", markersize=6, lw=0.6, alpha=0.8, ax=ax[0], legend=False)
    sns.pointplot(data=inter_df, x="session_id", y="percent_time_interacting",
                  estimator=np.median, errorbar=("sd"), color="k", ax=ax[0], linewidth=1.5, markersize=6, legend=False)
    ax[0].set_title("% Time Interacting"); ax[0].set_xlabel("Session"); ax[0].set_ylabel("%"), ax[0].set_ylim(0,50)

    sns.lineplot(data=speed_df, x="session_id", y="smoothed_speed",
                 hue="mouse_id", palette=pal,
                 marker="o", markersize=6, lw=0.6, alpha=0.8, ax=ax[1], legend=False)
    sns.pointplot(data=speed_df, x="session_id", y="smoothed_speed",
                  estimator=np.median, errorbar=("sd"), color="k", ax=ax[1], linewidth=1.5, markersize=6, legend=False)
    ax[1].set_title("Mean Speed"); ax[1].set_xlabel("Session"); ax[1].set_ylabel("px/s")

    sns.lineplot(data=dist_df, x="session_id", y="distance_m",
                 hue="mouse_id", palette=pal, 
                 marker="o", markersize=6, lw=0.6, alpha=0.8, ax=ax[2], legend=False)
    sns.pointplot(data=dist_df, x="session_id", y="distance_m",
                  estimator=np.median, errorbar=("sd"), color="k", ax=ax[2], linewidth=1.5, markersize=6, legend=False)
    ax[2].set_title("Distance moved per session"); ax[2].set_xlabel("Session"); ax[2].set_ylabel("metres travelled")

    sns.lineplot(data=event_counts, x="session_id", y="percent_reward",
                 hue="mouse_id",palette=pal,
                 marker="o", markersize=6, lw=0.6, alpha=0.8, ax=ax[3], legend=False)
    sns.pointplot(data=event_counts, x="session_id", y="percent_reward",
                  estimator=np.median, errorbar=("sd"), color="k", ax=ax[3],linewidth=1.5, markersize=6, legend=False)
    ax[3].set_title("Reward %"); ax[3].set_xlabel("Session"); ax[3].set_ylabel("%")

    sns.lineplot(data=bout_df, x="session_id", y="count",
                 hue="mouse_id",  palette=pal,
                 marker="o", markersize=6, lw=0.6, alpha=0.8, ax=ax[4], legend=False)
    sns.pointplot(data=bout_df, x="session_id", y="count",
                  estimator=np.median, errorbar=("sd"), color="k", ax=ax[4],linewidth=1.5, markersize=6, legend=False)
    ax[4].set_title("Rewards During Bouts"); ax[4].set_xlabel("Session"); ax[4].set_ylabel("Count")

    ax[0].grid(False)
    ax[1].grid(False)
    ax[2].grid(False)
    ax[3].grid(False)
    ax[4].grid(False)
    for a in ax: a.tick_params(axis="x", direction="in")
    plt.savefig(r'F:\upgrade_figures\behaviour\mice_behavior_summary_methimazole.svg', dpi=300)

plot_behaviour_seaborn()
exit()

def plotting(directory):
    all_files = [
        p for p in pathlib.Path(directory).rglob('*hmm_features.csv')
        if 'olfactory_ctrls' not in (part.lower() for part in p.parts)
    ]
    for file in all_files:
        print('processing file:', file)
        sesh = session.Session(mouse_id=str(file.parent.parent).split('\\')[3], session_path=str(file.parent.parent))
        centroid_pos = xr.open_dataset(file.parent / 'tracking_data.nc')
        centroid_pos=centroid_pos.sel(keypoints='abdomen')
        states = pd.read_csv(file.parent / f'hmm_inferred_states_K=6.csv')['states'].values
        hmm_array=pd.read_csv(file)
        hmm_array['states'] = states
        outpath = Path(file.parent / 'plots')
        outpath.mkdir(parents=True, exist_ok=True)

        plot_deltaf_vs_distance(hmm_array, ax=None, nbins=80, x_col='abdomen_abdomen', delta_col='deltaF_z')
        #plot_approach_trajectory(hmm_array, centroid_pos, outpath)
        #plot_normalised_approach_reward(hmm_array, outpath, deltaF_col='deltaF_z', offset=100)
        plot_delta_f_stim_onset_split_distance(hmm_array, outpath, distance='abdomen_abdomen', distance_threshold=150)
        plot_stim_onset_reward_split(hmm_array, outpath)
        #plot_poke(hmm_array, outpath)
        plot_deltaF(hmm_array, outpath)
        plot_deltaF(hmm_array, outpath, column='reward_approach_onset')
        plot_deltaF(hmm_array, outpath, column='reward')
    return
# for mouse in MICE:
#     plotting(rf'F:\social_sniffing\derivatives\{mouse}')
        


