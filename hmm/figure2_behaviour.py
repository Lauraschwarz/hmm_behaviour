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
from plotting_hmm import plot_transition_circos
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
from scipy.integrate import cumulative_trapezoid
import numpy as np
import ultraplot as uplt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from plotting_hmm import  empirical_transition_matrix_boutwise_by_trial
#from plotting import plot_deltaF

cmp = sns.diverging_palette(220, 20, as_cmap=True)
colors = ["#112047", "#093885", "#2ca1db", 
            
           "#f9ac07", "#c77406", "#963b04", "#640303"]
cmp_discrete = sns.color_palette('spectral', as_cmap=True)
cmp2 = sns.color_palette("tab20", as_cmap=True) 


def make_state_colors_from_cmap(states, cmap):
    states = sorted(states)
    n = len(states)

    # Evenly sample the colormap
    colors = cmap(np.linspace(0, 1, n))

    return dict(zip(states, colors))

def classify_motion_relative_to_roi(
    df,
    roi_points,
    pos_cols=("x_pos","y_pos"),
    disp_cols=("forward_x","forward_y"),
    speed_thresh=1e-3,
    align_thresh=0.2,
    label_col="roi_motion"
):
    """
    Adds column classifying movement relative to ROI.

    Labels:
        1  -> toward ROI
       -1  -> away from ROI
        0  -> lateral / weak movement
    """

    df = df.copy()

    pts = np.array(roi_points)
    cx, cy = pts.mean(axis=0)

    px = df[pos_cols[0]].values
    py = df[pos_cols[1]].values

    dx = df[disp_cols[0]].values
    dy = df[disp_cols[1]].values

    # vector to ROI
    vx = cx - px
    vy = cy - py

    # magnitudes
    disp_mag = np.hypot(dx, dy)
    roi_mag = np.hypot(vx, vy)

    # avoid divide-by-zero
    valid = (disp_mag > speed_thresh) & (roi_mag > 1e-6)

    # cosine alignment
    cos_align = np.zeros_like(disp_mag)
    cos_align[valid] = (
        (dx[valid]*vx[valid] + dy[valid]*vy[valid]) /
        (disp_mag[valid]*roi_mag[valid])
    )

    labels = np.zeros(len(df), dtype=int)
    labels[cos_align > align_thresh] = 1
    labels[cos_align < -align_thresh] = -1

    df[label_col] = labels

    return df



def plot_position_by_state(
    hmm_array,
    outpath,
    state_col="states",
    movement_type=-1,
    condition="Appetitive",
    decimate=1,                 # keep every Nth point (big Illustrator speed-up)
    rasterize_scatter=False,    # rasterize heavy layers inside SVG
    rasterize_quiver=False,
    quiver_stride=5,            # keep every Nth arrow
    quiver_scale=0.4,
    scatter_size=10,
    scatter_alpha=1,
    quiver_alpha=0.35,
    trial_col="trial",          # NEW
    trials=None,                # NEW: list/iterable of trial ids to include (e.g. [1,2,3,4,5])
    first_n_trials=None,        # NEW: convenience (e.g. 5)
):
    states = hmm_array[state_col].dropna().unique()
    STATE_COLORS = make_state_colors_from_cmap(states, cmp_discrete)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot()

    df = hmm_array[hmm_array["olfactory_stim"] == condition].reset_index(drop=True)

    # ---- trial filtering (optional) ----
    if first_n_trials is not None:
        if trial_col not in df.columns:
            raise ValueError(f"'{trial_col}' column not found, cannot select first_n_trials.")
        # preserve dataframe order; take first N unique trial ids as they appear
        trial_ids = pd.Index(df[trial_col].dropna().unique()).tolist()
        keep = set(trial_ids[: int(first_n_trials)])
        df = df[df[trial_col].isin(keep)].copy()

    if trials is not None:
        if trial_col not in df.columns:
            raise ValueError(f"'{trial_col}' column not found, cannot filter by trials.")
        df = df[df[trial_col].isin(list(trials))].copy()
    # -----------------------------------

    for state in sorted(df[state_col].dropna().unique()):
        df_state = df[(df[state_col] == state) & (df["roi_motion"] == movement_type)].copy()
        if df_state.empty:
            continue

        # Downsample points (reduces SVG object count a lot)
        if decimate and decimate > 1:
            df_state_pts = df_state.iloc[::decimate]
        else:
            df_state_pts = df_state

        ax.scatter(
            df_state_pts["x_pos"],
            df_state_pts["y_pos"],
            s=scatter_size,
            color=STATE_COLORS[state],
            label=state,
            alpha=scatter_alpha,
            linewidths=0,
            rasterized=bool(rasterize_scatter),
        )

        # Quiver is extremely heavy in SVG: downsample strongly + optionally rasterize
        if quiver_stride and quiver_stride > 1:
            df_state_q = df_state.iloc[::quiver_stride]
        else:
            df_state_q = df_state

        qv = ax.quiver(
            df_state_q["x_pos"],
            df_state_q["y_pos"],
            df_state_q["forward_x"],
            df_state_q["forward_y"],
            angles="xy",
            scale=quiver_scale,
            scale_units="xy",
            headwidth=6,
            headlength=7,
            headaxislength=6,
            alpha=quiver_alpha,
        )
        if rasterize_quiver:
            qv.set_rasterized(True)

    ax.grid(False)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.legend(markerscale=2, frameon=False, handlelength=0.8, handletextpad=0.4)
    ax.set_title("Position by state")

    suffix = ""
    if first_n_trials is not None:
        suffix = f"_first{int(first_n_trials)}trials"
    elif trials is not None:
        suffix = "_trials" + "-".join(map(str, trials))

    out_file = Path(outpath) / f"position_by_state_movement_{movement_type}_all_quiver_type_{condition}{suffix}.svg"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)



def plot_deltaF_aligned_to_state_onsets(
    hmm_array,
    outpath,
    state_col="states",
    signal_col="deltaF_z",
    fps=50,
    window_seconds=2,
    baseline_frames=0,
    min_state_len=0,
    movement_type=-1,
    stim="Appetitive",
    time_col="Unnamed: 0",
    trial_col="trial",
    trials=None,
    first_n_trials=None,
    one_hit_per_motion_bout=True,   # NEW: avoid multiple hits from same retreat/toward bout
    motion_col="roi_motion",         # NEW
):
    # Build a time-contiguous base (do NOT filter before alignment)
    base = hmm_array.copy()

    if "__row_ix__" not in base.columns:
        base = base.reset_index(drop=False).rename(columns={"index": "__row_ix__"})

    if time_col in base.columns:
        base[time_col] = pd.to_datetime(base[time_col], errors="coerce")
        base = base.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    else:
        base = base.reset_index(drop=True)

    if state_col not in base.columns or signal_col not in base.columns:
        raise ValueError(f"Missing columns: {state_col} or {signal_col}")

    # Define motion bouts on the full time base (so bouts are time-contiguous)
    # Bout increments when we enter a movement_type segment.
    is_move = (base[motion_col].to_numpy() == movement_type)
    enter = np.r_[False, (is_move[1:] & ~is_move[:-1])]
    base["_motion_bout_id"] = np.cumsum(enter) * is_move  # 0 outside movement_type, 1..N within

    # subset for onset detection
    sub = base[(base["olfactory_stim"] == stim) & (base[motion_col] == movement_type)].copy()

    # ---- trial filtering (optional; same logic as plot_position_by_state) ----
    if (first_n_trials is not None) or (trials is not None):
        if trial_col not in sub.columns:
            raise ValueError(f"'{trial_col}' column not found, cannot filter by trials.")

    if first_n_trials is not None:
        trial_ids = pd.Index(sub[trial_col].dropna().unique()).tolist()
        keep = set(trial_ids[: int(first_n_trials)])
        sub = sub[sub[trial_col].isin(keep)].copy()

    if trials is not None:
        sub = sub[sub[trial_col].isin(list(trials))].copy()
    # -----------------------------------------------------------------------

    window_pre = int(window_seconds * fps)
    window_post = int(window_seconds * fps)
    n_samples = window_pre + window_post + 1
    x = np.arange(-window_pre, window_post + 1) / float(fps)

    ix_to_pos = pd.Series(np.arange(len(base)), index=base["__row_ix__"]).to_dict()
    base_signal = pd.to_numeric(base[signal_col], errors="coerce").to_numpy()

    for target_state in sorted(sub[state_col].dropna().unique()):
        s = sub[state_col].to_numpy()
        idx = np.flatnonzero(s == target_state)
        if idx.size == 0:
            continue

        splits = np.where(np.diff(idx) != 1)[0] + 1
        runs = np.split(idx, splits)

        # Build candidate onsets: onset row_ix + which motion bout they belong to
        candidates = []
        for r in runs:
            if len(r) < min_state_len:
                continue
            onset_row_ix = int(sub.iloc[r[0]]["__row_ix__"])
            motion_bout_id = int(sub.iloc[r[0]]["_motion_bout_id"])
            if motion_bout_id == 0:
                continue
            candidates.append((onset_row_ix, motion_bout_id))

        if not candidates:
            continue

        # De-duplicate: keep only one hit per motion bout (per target_state)
        if one_hit_per_motion_bout:
            seen = set()
            onset_row_ixs = []
            for onset_row_ix, mb in sorted(candidates, key=lambda x: x[0]):  # keep earliest in time
                if mb in seen:
                    continue
                seen.add(mb)
                onset_row_ixs.append(onset_row_ix)
        else:
            onset_row_ixs = [c[0] for c in candidates]

        aligned = []
        for onset_ix in onset_row_ixs:
            pos = ix_to_pos.get(onset_ix, None)
            if pos is None:
                continue

            lo, hi = pos - window_pre, pos + window_post
            if lo < 0 or hi >= len(base_signal):
                continue

            seg = base_signal[lo : hi + 1]
            if seg.shape[0] != n_samples:
                continue

            b = min(baseline_frames, window_pre)
            base0 = np.nanmean(seg[:b]) if b > 0 else 0.0
            aligned.append(seg - base0)

        if len(aligned) == 0:
            continue

        aligned = np.vstack(aligned)

        baseline_mask = x < 0
        baseline_vals = aligned[:, baseline_mask].ravel()
        mu = np.nanmean(baseline_vals)
        sd = np.nanstd(baseline_vals, ddof=1)
        aligned_z = (aligned - mu) / sd if (np.isfinite(sd) and sd > 0) else (aligned - mu)

        mean = np.nanmean(aligned_z, axis=0)
        sem = np.nanstd(aligned_z, axis=0, ddof=1) / np.sqrt(aligned_z.shape[0])

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax0.plot(x, aligned_z.T, alpha=0.25, linewidth=0.5, marker=".", color="gray")
        ax0.axvline(0, linestyle="--")
        ax0.set_ylabel(signal_col)
        ax0.set_title(f"{signal_col} aligned to onset of state {target_state} (n={aligned_z.shape[0]})")
        ax0.grid(False)
        ax0.set_axis_off()

        ax1.plot(x, mean, linewidth=2)
        ax1.fill_between(x, mean - sem, mean + sem, alpha=0.25)
        ax1.axvline(0, linestyle="--")
        ax1.set_xlabel("Time (s) relative to onset")
        ax1.set_ylabel(f"Mean {signal_col}")
        ax1.grid(False)
        ax1.set_axis_off()

        suffix = ""
        if first_n_trials is not None:
            suffix = f"_first{int(first_n_trials)}trials"
        elif trials is not None:
            suffix = "_trials" + "-".join(map(str, trials))

        if one_hit_per_motion_bout:
            suffix += "_oneHitPerBout"

        fig.tight_layout()
        fig.savefig(
            Path(outpath)
            / f"{signal_col}_aligned_to_state_{target_state}_onsets_{stim}_K5_mov_{movement_type}{suffix}_psthstyle.svg",
            dpi=300,
        )
        plt.close(fig)



file_path = pathlib.Path(r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\Methimazole\model_predictions\gaussian_raw\all_session_data_5states.csv')
outpath=r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\Methimazole\plots'
df = pd.read_csv(file_path)
port1 = ((400,500),(400,600),(500,600),(500,500))
port078 = ((200, 450), (200, 550),(300, 550),(300, 450))

df = classify_motion_relative_to_roi(df, port1)
# fig, ax = uplt.subplots(ncols=3, wratios=(3, 1, 1), ref=1, refwidth=4, share=False)
#     ax.format(abc=True)
for i in [-1,  1]:
    for condition in df['olfactory_stim'].unique():
        plot_position_by_state(df, outpath=r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\Methimazole\plots', state_col='states', movement_type=i, condition='Aversive', first_n_trials=10)

plot_deltaF_aligned_to_state_onsets(df, outpath=r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\Methimazole\plots', stim='Aversive', movement_type=-1, first_n_trials=10)
exit()
for movement_type in [-1, 1]:
    P, counts = empirical_transition_matrix_boutwise_by_trial(df[df['roi_motion'] == movement_type], "states", "trial", K=5)

    fig, ax = plot_transition_circos(
         P,
         min_prob=0.001,          # hide tiny transitions
         max_edges_per_node=10,   # declutter
         arrow=True
     )
    fig.savefig(Path(outpath) / f"transition_circos_{movement_type}.svg", dpi=300)
exit()

def find_bout_pattern_hits(df, state_col, pattern=(0, 4, 1)):
    """
    Finds occurrences of pattern in consecutive BOUTS (run-length compressed states).
    Returns a list of onset row indices (absolute row ids) for the onset of the first state in pattern.
    """
    g = df.copy()

    # bout segmentation
    g["_state_prev"] = g[state_col].shift(1)
    g["_new_bout"] = (g[state_col] != g["_state_prev"]).astype(int)
    g["_bout_id"] = g["_new_bout"].cumsum()

    # Robust aggregation (no named-agg tuple syntax)
    gb = g.groupby("_bout_id", sort=False)
    bouts = pd.DataFrame({
        "state": gb[state_col].first(),
        "start_ix": gb["__row_ix__"].first(),
        "end_ix": gb["__row_ix__"].last(),
    }).reset_index(drop=True)

    states = bouts["state"].to_numpy()
    p = np.array(pattern)

    hits = []
    if len(states) >= len(p):
        for i in range(len(states) - len(p) + 1):
            if np.array_equal(states[i:i + len(p)], p):
                hits.append(int(bouts.loc[i, "start_ix"]))  # onset of pattern[0] bout
    return hits



def extract_aligned_windows(base_df, onset_row_ixs, value_col, pre=30, post=90):
    """
    Extract fixed-length windows around each onset using *base_df* which must be
    sorted in time and contain a stable '__row_ix__' column.

    onset_row_ixs are '__row_ix__' values (stable ids), not positional indices.
    """
    win_len = pre + post + 1

    if "__row_ix__" not in base_df.columns:
        raise ValueError("base_df must contain '__row_ix__'")

    # map stable id -> positional index in base_df (time-contiguous)
    ix_to_pos = pd.Series(np.arange(len(base_df)), index=base_df["__row_ix__"]).to_dict()
    values = base_df[value_col].to_numpy()

    aligned = []
    kept = []

    for onset_ix in onset_row_ixs:
        pos = ix_to_pos.get(onset_ix, None)
        if pos is None:
            continue
        lo, hi = pos - pre, pos + post
        if lo < 0 or hi >= len(values):
            continue
        aligned.append(values[lo:hi + 1])
        kept.append(onset_ix)

    if not aligned:
        return np.empty((0, win_len)), kept

    return np.vstack(aligned), kept

def plot_psth_state_sequence_hits(
    hmm_df,
    outpath,
    condition_col="olfactory_stim",
    condition="Aversive",
    movement_col="roi_motion",
    movement_type=-1,
    state_col="states",
    value_col="deltaF",
    pattern=(0, 4, 1),
    pre=30,
    post=90,
    save_name=None,
):
    time_col = "Unnamed: 0"

    base = hmm_df.copy()

    # parse to datetime and sort by time
    base[time_col] = pd.to_datetime(base[time_col], errors="coerce")
    base = base.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)

    # stable row id for mapping onsets back to base
    base = base.reset_index(drop=False).rename(columns={"index": "__row_ix__"})
        # Filter once
    df = base[
        (base[condition_col] == condition) &
        (base[movement_col] == movement_type) &
        (~base[state_col].isna()) &
        (~base[value_col].isna())
    ].copy()

    # Find all hits (each hit = a "trial")
    onset_ixs = find_bout_pattern_hits(df, state_col=state_col, pattern=pattern)

    # Extract aligned deltaF_z windows
    aligned, kept = extract_aligned_windows(base, onset_ixs, value_col=value_col, pre=pre, post=post)

    if aligned.shape[0] == 0:
        raise ValueError("No usable hits found (or all hits were too close to edges for the chosen pre/post).")

    # Mean/SEM
    
    x = np.arange(-pre, post + 1)
    baseline_mask = x < 0            # x = np.arange(-pre, post+1)
    baseline_vals = aligned[:, baseline_mask].ravel()

    mu = baseline_vals.mean()
    sd = baseline_vals.std(ddof=1)
    aligned_z = (aligned - mu) / sd

    mean = aligned_z.mean(axis=0)
    sem = aligned_z.std(axis=0, ddof=1) / np.sqrt(aligned.shape[0])

    # Plot
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    # Single-hit traces
    ax0.plot(x, aligned_z.T, alpha=0.25, linewidth=0.5, marker='.', color='gray')
    ax0.axvline(0, linestyle="--")
    ax0.set_ylabel(value_col)
    ax0.set_title(f"{pattern} hits aligned to state {pattern[0]} onset (n={aligned.shape[0]})")
    ax0.grid(False)
    ax0.set_axis_off()

    # Mean PSTH
    ax1.plot(x, mean, linewidth=2)
    ax1.fill_between(x, mean - sem, mean + sem, alpha=0.25)
    ax1.axvline(0, linestyle="--")
    ax1.set_xlabel("Samples relative to onset")
    ax1.set_ylabel(f"Mean {value_col}")
    ax1.grid(False)
    ax1.set_axis_off()

    if save_name is None:
        save_name = f"psth_hits_{pattern[0]}_{pattern[1]}_{condition}_mov_{movement_type}.svg"

    fig.tight_layout()
    fig.savefig(Path(outpath) / save_name, dpi=300)
    plt.close(fig)

    return aligned, kept

def plot_example_trace(df, x_col="x_pos", y_col="y_pos", cmap="mako", time_col=None):
    d = df[[x_col, y_col]].copy().dropna()
    t = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(
        d[x_col].to_numpy(),
        d[y_col].to_numpy(),
        c=t/50,
        s=3,
        cmap=cmap,
        linewidths=0,
        rasterized=True,   # key: rasterize the points
    )

    ax.grid(False)
    ax.set_axis_off()
    ax.set_aspect("equal")

    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Time (seconds)")

    out = Path(r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\CTRL\plots') / f'example_trace_{x_col}_{y_col}_app.svg'
    fig.savefig(out, dpi=300)  # dpi matters for the rasterized parts only
    plt.show()


# plot_position_by_state(df, outpath=r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\CTRL\plots', state_col='states', movement_type=1, condition='Aversive')

# plot_position_by_state(df, outpath=r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\CTRL\plots', state_col='states', movement_type=-1, condition='Aversive')
# exit()
# plot_example_trace(df[df['olfactory_stim'] == 'Appetitive'].reset_index(drop=True)[10000:], x_col="x_pos", y_col="y_pos", cmap="mako")
# exit()

aligned, kept_onsets = plot_psth_state_sequence_hits(
    hmm_df=df,
    outpath=r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\Methimazole\plots',
    pattern=(0,1),
    pre=50,
    post=100,
    movement_type=-1
)
aligned, kept_onsets = plot_psth_state_sequence_hits(
    hmm_df=df,
    outpath=r'F:\social_sniffing\derivatives\1106077\olfactory_ctrls\Methimazole\plots',
    pattern=(4,1),
    pre=50,
    post=100,
    movement_type=1
)
