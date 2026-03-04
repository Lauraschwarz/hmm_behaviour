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
import umap
import ultraplot as up 
from constants import ARENA
sys.path.append(r'C:\Users\Laura\social_sniffing\sniffies')
import session
from os import mkdir
from sklearn.preprocessing import StandardScaler

from fancyimpute import IterativeSVD

from scipy.ndimage import label
from scipy.signal import find_peaks

cmp = sns.diverging_palette(220, 20, as_cmap=True)
cmp2 = sns.color_palette("tab20", as_cmap=True) 

MICE = [ '1106077','1106078','1106079']#,'1106010', '1106009']#, '1106010', '1106008', '1106078', '1106079']#, '1125563', '1125561','1125555','1125131', '1125132']
VGAT_MICE = ['1106077', '1106078', '1106079']
METHIMAZOLE_SESSIONS = {'1106077': ['2025-10-02T14-11-03','2025-10-02T10-18-45','2025-10-02T10-36-37'],
                        '1106009': ['2025-10-02T13-33-33', '2025-10-02T09-44-35','2025-10-02T09-59-09'],
                        '1106010': ['2025-09-26T12-42-23', '2025-09-26T14-34-16','2025-09-26T14-49-47'],
                        '1106078': ['2025-09-26T13-21-09', '2025-09-26T13-57-52','2025-09-26T14-14-14'],
                        '1106079': ['2025-10-02T14-45-26', '2025-10-02T15-14-47','2025-10-02T15-26-26'],
                   
                        }
AVERSIVE_SESSIONS = {'1106079': ['2025-09-29T13-38-32','2025-10-02T15-14-47'],
                     '1106078': ['2025-09-22T12-52-28', '2025-09-26T13-57-52'],
                     '1106077': ['2025-09-29T13-07-16', '2025-10-02T10-18-45'], 
                     '1106009':['2025-09-29T12-28-04', '2025-10-02T09-44-35'], 
                     '1106010':['2025-09-22T12-25-50', '2025-09-26T14-34-16'],
                     '1106008':['2025-09-29T10-08-53'], 
                     '1125140':['2025-06-05T12-57-31', '2025-06-06T12-44-52']
                  }

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




def add_signal_peaks_to_df(df):
    peaks = find_peaks(df['deltaF_z'], distance=25, prominence=2.5)
    df['signal_peaks'] = np.full(len(df), np.nan)
    df['signal_peaks'][peaks[0]]=1
    return df

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

def get_merged_sampling_runs(df, sampling_col="sampling", merge_gap=100):
    if sampling_col not in df.columns:
        raise ValueError(f"Missing column '{sampling_col}'")

    idx = np.flatnonzero(df[sampling_col].eq(1).to_numpy())
    if idx.size == 0:
        return []

    splits = np.where(np.diff(idx) != 1)[0] + 1
    runs = np.split(idx, splits)

    merged = []
    s0, e0 = runs[0][0], runs[0][-1]

    for r in runs[1:]:
        s, e = r[0], r[-1]
        if (s - e0 - 1) <= merge_gap:
            e0 = e
        else:
            merged.append((s0, e0))
            s0, e0 = s, e

    merged.append((s0, e0))
    return merged

def annotate_sampling_trials(
    df,
    sampling_col="sampling",
    merge_gap=100,
    buffer_frames=100,
    trial_col="trial",
):
    df = df.copy()
    df[trial_col] = np.nan

    merged_runs = get_merged_sampling_runs(
        df, sampling_col=sampling_col, merge_gap=merge_gap
    )

    if not merged_runs:
        return df

    n = len(df)
    for i, (s, e) in enumerate(merged_runs, start=1):
        bs = max(0, s - buffer_frames)
        be = min(n - 1, e + buffer_frames)
        df.loc[bs:be, trial_col] = i

    return df

def df_to_trial_sequences(df, columns=None, trial_col="trial", return_concat=True):
    sequences = []
    segments = []
    if columns is None:
        columns = [
            "smoothed_speed",
            "smoothed_acceleration",
            "snout_groin",
            "neckL_groin",
            "neckR_groin",
            "sniff_freq",
            "speed_con_smoothed", 
            "acceleration_con_smoothed",
            "snout_groin_con",
            "neckL_groin_con",
            "neckR_groin_con",

        ]
    for _, g in df.dropna(subset=[trial_col]).groupby(trial_col):
        X = g[columns].to_numpy()
        if len(X) > 1:
            sequences.append(X)
            segments.append(g.copy())

    if return_concat:
        if len(segments) == 0:
            return df.iloc[0:0].copy()
        return pd.concat(segments, axis=0).reset_index(drop=True)
    return sequences

def impute_df_features(df, columns=None, max_rank=5, max_iters=200):
    """
    Impute missing values in `columns` using IterativeSVD and z-score the result.
    Returns a copy of df with imputed & scaled columns.
    """
    if columns is None:
        columns = [
            "smoothed_speed",
            "smoothed_acceleration",
            "snout_groin",
            "neckL_groin",
            "neckR_groin",
            "sniff_freq",
            "speed_con_smoothed", 
            "acceleration_con_smoothed",
            "snout_groin_con",
            "neckL_groin_con",
            "neckR_groin_con",
            
        ]
    df_out = df.copy()
    featarr = df_out[columns].astype(float).to_numpy()
    rank_safe = max(1, min(max_rank, min(featarr.shape) - 1))
    featarr = np.where(np.isfinite(featarr), featarr, np.nan)
    imputed = IterativeSVD(rank=rank_safe, max_iters=max_iters).fit_transform(featarr)
    df_out.loc[:, columns] = StandardScaler().fit_transform(imputed)
    return df_out

def impute_and_separate_df(df, buffer_frames=100, merge_gap=100):
    """
    Merge bouts that are within `merge_gap` frames, then add ±`buffer_frames` around each merged bout.
    """
    selected_columns = [
        "smoothed_speed",
        "smoothed_acceleration",
        "snout_groin",
        "neckL_groin",
        "neckR_groin",
        "speed_con_smoothed", 
        "acceleration_con_smoothed",
        "snout_groin_con",
        "neckL_groin_con",
        "neckR_groin_con",
    
        "sniff_freq"
                ]
    featarr = df[selected_columns].astype(float).to_numpy()
    rank_safe = max(1, min(5, min(featarr.shape) - 1))
    featarr = np.where(np.isfinite(featarr), featarr, np.nan)
    imputed = IterativeSVD(rank=rank_safe, max_iters=200).fit_transform(featarr)
    df.loc[:, selected_columns] = StandardScaler().fit_transform(imputed)

    if 'sampling' not in df.columns:
        raise AttributeError("DataFrame must contain 'sampling' column.")

    mask = df['sampling'].eq(1).to_numpy()
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return df.iloc[0:0].copy()

    # contiguous runs where sampling==1
    splits = np.where(np.diff(idx) != 1)[0] + 1
    runs = np.split(idx, splits)

 # merge runs with small gaps
    merged = []
    s0, e0 = runs[0][0], runs[0][-1]
    for r in runs[1:]:
        s, e = r[0], r[-1]
        if (s - e0 - 1) <= merge_gap:
            e0 = e
        else:
            merged.append((s0, e0))
            s0, e0 = s, e
    merged.append((s0, e0))

    # build buffered segments and assign trial number across the whole segment
    n = len(df)
    segment_dfs = []
    for i, (s, e) in enumerate(merged, start=1):
        bs = max(0, s - buffer_frames)
        be = min(n - 1, e + buffer_frames)
        seg = df.iloc[bs:be + 1].copy()
        seg['orig_index'] = df.index.values[bs:be + 1]
        seg['trial'] = i  # extend trial id into buffers
        segment_dfs.append(seg)

    sampling_df = pd.concat(segment_dfs, axis=0).reset_index(drop=True)
    return sampling_df

def is_session(mapping, mouse_id, session_name):
    vals = mapping.get(mouse_id, [])
    if isinstance(vals, (list, tuple, set)):
        return session_name in vals
    return session_name == vals

def create_mouse_df(mouse_directory):
        all_files = [
        p for p in mouse_directory.rglob('*social_hmm_features.csv')]
        # Always get a Path and create the directory with pathlib
        outpath = Path(mouse_directory) / 'hmm_combined'
        outpath.mkdir(parents=True, exist_ok=True)
        # if (outpath / 'all_sessions_df_v3.csv').exists():
        #     print('Combined dataframe already exists, loading from file...')
        #     combined_hmm_array = pd.read_csv(outpath / 'all_sessions_df_v3.csv')
        #     sampling_df = pd.read_csv(outpath / 'imputed_split_sampling_bouts.csv')
        #     return combined_hmm_array, sampling_df
   
        hmm_array_full = []
        for i, file in enumerate(all_files):
            print('processing file:', file)
            session_name = str(file.parent.parent).split('\\')[-1]
            sesh = session.Session(mouse_id=str(file.parent.parent).split('\\')[3], session_path=str(file.parent.parent))
            centroid_pos = xr.open_dataset(file.parent / 'tracking_data_v3.nc')
            centroid_pos=centroid_pos.sel(keypoints='abdomen')
            hmm_array=pd.read_csv(file)
            outpath = Path(mouse_directory) / 'hmm_combined'
            outpath.mkdir(parents=True, exist_ok=True)
            hmm_array_full.append(hmm_array)
            hmm_array['session_id'] = f'session_{i+1}' #because  0 indexing
            mouse_id = str(file.parent.parent).split('\\')[3]
            hmm_array['mouse_id'] = mouse_id
            hmm_array['date'] = str(file.parent.parent).split('\\')[-1]
            if str(file.parent.parent).split('\\')[-1] == METHIMAZOLE_SESSIONS.get(sesh.mouse_id):
                hmm_array['condition'] = 'Methimazole'
            else:
                hmm_array['condition'] = 'Control'
            sampling_path = file.parent.parent / 'Video' / 'sampling_bouts.npy'
            
            sampling = np.load(sampling_path, allow_pickle=True) 
           
            hmm_array['genotype'] = GENOTYPES.get(sesh.mouse_id)
            hmm_array['olfactory_stim'] = 'Aversive' if is_session(AVERSIVE_SESSIONS, sesh.mouse_id, session_name) else 'Appetitive'
            print(hmm_array['olfactory_stim'].values[0])
          
            hmm_array['sampling'] = np.full(len(hmm_array), np.nan)
            for b in sampling:
                if isinstance(b, (list, tuple, np.ndarray)) and len(b) == 2:
                    start, end = int(b[0]), int(b[1])
                    hmm_array.loc[start:end, 'sampling'] = 1
                else:
                    hmm_array.loc[int(b), 'sampling'] = 1

       
        combined_hmm_array = pd.concat(hmm_array_full, ignore_index=True)

        
        combined_hmm_array = annotate_sampling_trials(
            combined_hmm_array
        )
        combined_hmm_array = impute_df_features(combined_hmm_array)
        combined_hmm_array.to_csv((outpath / 'all_sessions_df_v4.csv'), index=False)

        # Get a single concatenated DataFrame of all trial segments
        sampling_df = df_to_trial_sequences(combined_hmm_array, return_concat=True)
        sampling_df.to_csv((outpath / 'imputed_split_sampling_bouts_v2.csv'), index=False)

        return combined_hmm_array, sampling_df

def create_all_mice_df(MICE, experiment='social', condition='Methimazole'):
    all_mice_dfs = []
    all_sampling_dfs = []
    for mouse in MICE:
        path = Path(rf'F:\social_sniffing\derivatives\{mouse}\{experiment}\{condition}') 
        mouse_df, sampling_df = create_mouse_df(path)
        mouse_df['mouse_id'] = mouse
        mouse_df = add_signal_peaks_to_df(mouse_df)
        all_mice_dfs.append(mouse_df)
        all_sampling_dfs.append(sampling_df)
    
    combined_all_mice_df = pd.concat(all_mice_dfs, ignore_index=True)
    combined_all_sampling_df = pd.concat(all_sampling_dfs, ignore_index=True)
    combined_all_sampling_df.to_csv(rf'F:\social_sniffing\derivatives\test_concat_hmm_features_{experiment}_{condition}all_mice_doublefeature.csv', index=False)
    
    return combined_all_mice_df, combined_all_sampling_df
create_all_mice_df(MICE)


exit()


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

        plot_deltaF(hmm_array, outpath)
        plot_deltaF(hmm_array, outpath, column='reward_approach_onset')
        plot_deltaF(hmm_array, outpath, column='reward')
    return
# for mouse in MICE:
#     plotting(rf'F:\social_sniffing\derivatives\{mouse}')
        


