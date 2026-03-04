
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from panel import state
import seaborn as sns

# Local imports (keep only what is used in this file)
from constants import ARENA, ARENA_VERTICES, inner_vertices, TRIAL_LENGTH  # noqa: F401
from movement.roi import PolygonOfInterest  # noqa: F401
from movement.utils.vector import compute_norm  # noqa: F401
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.append(r"C:\Users\Laura\social_sniffing\sniffies")
from global_functions import get_barrier_open_time  # noqa: F401,E402

from plotting_hmm import plot_transition_circos, empirical_transition_matrix_boutwise_by_trial, plot_transition_matrix_no_self  # noqa: E402
from scipy.ndimage import gaussian_filter1d  

# ---- palettes ----
cmp = sns.diverging_palette(220, 20, as_cmap=True)
cmp_discrete = sns.color_palette("Spectral", as_cmap=True)
cmp2 = sns.color_palette("tab20", as_cmap=True)  # noqa: F401


def make_state_colors_from_cmap(states, cmap):
    states = sorted(states)
    colors = cmap(np.linspace(0, 1, len(states)))
    return dict(zip(states, colors))

def keep_longest_bout_per_trial(bouts_df, trial_col="trial", len_col="bout_len"):
    if bouts_df.empty:
        return bouts_df
    if trial_col not in bouts_df.columns:
        raise ValueError(f"'{trial_col}' missing in bouts_df")
    if len_col not in bouts_df.columns:
        raise ValueError(f"'{len_col}' missing in bouts_df")

    # keep first if ties (stable)
    return (
        bouts_df.sort_values([trial_col, len_col], ascending=[True, False])
                .groupby(trial_col, sort=False, as_index=False)
                .head(1)
                .reset_index(drop=True)
    )

def mean_sem_across_mice(per_mouse_traces):
    """
    per_mouse_traces: list of 1D arrays (same length), one per mouse
    Returns mean and SEM across mice.
    """
    X = np.vstack(per_mouse_traces)
    mean = np.nanmean(X, axis=0)
    sem = np.nanstd(X, axis=0, ddof=1) / np.sqrt(X.shape[0])
    return mean, sem, X

def plot_group_mean_sem_per_state_subplots(
    traces_by_mouse,   # dict[mouse] -> dict[state] -> 1D mean trace
    t,
    *,
    states=(0, 1, 4),
    ncols=3,
    figsize_per_ax=(2.5, 3.0),
    title="Group mean ΔF_z aligned to state onset",
    ylabel="deltaF_z (baseline-corrected)",
    line_kw=None,
    band_alpha=0.25,
):
    if line_kw is None:
        line_kw = dict(color="k", linewidth=2.2)

    n_states = len(states)
    nrows = int(np.ceil(n_states / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_ax[0] * ncols, figsize_per_ax[1] * nrows),
        sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n_states:]:
        ax.axis("off")

    for i, state in enumerate(states):
        ax = axes[i]

        per_mouse = []
        for mouse, per_state in traces_by_mouse.items():
            if state in per_state:
                per_mouse.append(per_state[state])

        if len(per_mouse) == 0:
            ax.set_title(f"State {state} (no data)")
            ax.axis("off")
            continue

        X = np.vstack(per_mouse)  # (n_mice, n_time)
        mu = np.nanmean(X, axis=0)
        sem = np.nanstd(X, axis=0, ddof=1) / np.sqrt(X.shape[0]) if X.shape[0] > 1 else np.zeros_like(mu)

        ax.plot(t, mu, **line_kw)
        ax.fill_between(t, mu - sem, mu + sem, alpha=band_alpha, color=line_kw.get("color", "k"))

        ax.axvline(0, linestyle="--", linewidth=1, color="k", alpha=0.5)

        ax.set_title(f"State {state} (n={X.shape[0]} mice)", fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    # labels
    for ax in axes[:n_states]:
        ax.set_xlabel("Time (s)")
    for ax in axes[::ncols]:
        ax.set_ylabel(ylabel)

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes



def sort_bouts(
    M, t,
    baseline=(-1.0, 0.0),
    response=(0.0, 1.0),
    sort_mode="mean_signed",   # see options below
    direction="auto",          # "auto", "pos", "neg" (only used for some modes)
):
    """
    M: (n_bouts, n_timepoints) aligned matrix
    t: (n_timepoints,) time vector in seconds
    baseline/response: tuples in seconds defining windows relative to alignment (t=0)
    sort_mode options:
      - "peak_pos"      : max(post) - mean(pre)
      - "peak_neg"      : mean(pre) - min(post)   (stronger dips higher)
      - "mean_signed"   : mean(post) - mean(pre)  (can be +/-)
      - "mean_abs"      : abs(mean(post) - mean(pre))
      - "auc_signed"    : sum(post - mean(pre))   (signed)
      - "auc_abs"       : abs(sum(post - mean(pre)))
      - "effect_size"   : (mean(post)-mean(pre)) / std(pre)
      - "latency_peak"  : time of max(post) (earlier first)
      - "latency_trough": time of min(post) (earlier first)
    direction:
      - "auto": if sort_mode is "mean_signed"/"auc_signed"/"effect_size" choose pos/neg ordering based on median
      - "pos": strongest positive first
      - "neg": strongest negative first
    Returns: order (indices), score (per bout)
    """
    t = np.asarray(t)
    bmask = (t >= baseline[0]) & (t < baseline[1])
    rmask = (t >= response[0]) & (t <= response[1])

    if not bmask.any() or not rmask.any():
        raise ValueError("Baseline/response window masks are empty. Check your t axis and window values.")

    pre = M[:, bmask]
    post = M[:, rmask]

    pre_mean = np.nanmean(pre, axis=1)
    post_mean = np.nanmean(post, axis=1)

    if sort_mode == "peak_pos":
        score = np.nanmax(post, axis=1) - pre_mean
        order = np.argsort(score)[::-1]

    elif sort_mode == "peak_neg":
        score = pre_mean - np.nanmin(post, axis=1)
        order = np.argsort(score)[::-1]

    elif sort_mode == "mean_signed":
        score = post_mean - pre_mean
        if direction == "pos":
            order = np.argsort(score)[::-1]
        elif direction == "neg":
            order = np.argsort(score)
        else:  # auto
            order = np.argsort(score) if np.nanmedian(score) < 0 else np.argsort(score)[::-1]

    elif sort_mode == "mean_abs":
        score = np.abs(post_mean - pre_mean)
        order = np.argsort(score)[::-1]

    elif sort_mode == "auc_signed":
        score = np.nansum(post - pre_mean[:, None], axis=1)
        if direction == "pos":
            order = np.argsort(score)[::-1]
        elif direction == "neg":
            order = np.argsort(score)
        else:
            order = np.argsort(score) if np.nanmedian(score) < 0 else np.argsort(score)[::-1]

    elif sort_mode == "auc_abs":
        score = np.abs(np.nansum(post - pre_mean[:, None], axis=1))
        order = np.argsort(score)[::-1]

    elif sort_mode == "effect_size":
        pre_std = np.nanstd(pre, axis=1, ddof=1)
        pre_std[pre_std == 0] = np.nan
        score = (post_mean - pre_mean) / pre_std
        if direction == "pos":
            order = np.argsort(score)[::-1]
        elif direction == "neg":
            order = np.argsort(score)
        else:
            order = np.argsort(score) if np.nanmedian(score) < 0 else np.argsort(score)[::-1]

    elif sort_mode == "latency_peak":
        idx = np.nanargmax(post, axis=1)
        score = t[rmask][idx]   # latency in seconds
        order = np.argsort(score)  # earlier first

    elif sort_mode == "latency_trough":
        idx = np.nanargmin(post, axis=1)
        score = t[rmask][idx]
        order = np.argsort(score)

    else:
        raise ValueError(f"Unknown sort_mode: {sort_mode}")

    return order, score



def _require_row_ix(df: pd.DataFrame, col="__row_ix__") -> None:
    if col not in df.columns:
        raise ValueError(f"Missing '{col}'. Call _ensure_row_ix(df) once at the start of your pipeline.")
    
def _ensure_row_ix(df: pd.DataFrame, col="__row_ix__") -> pd.DataFrame:
    """Ensure stable row-id column exists (based on current row order)."""
    if col in df.columns:
        return df
    return df.reset_index(drop=False).rename(columns={"index": col})


def _sort_time(df: pd.DataFrame, time_col="Unnamed: 0") -> pd.DataFrame:
    """Sort by time if present; otherwise preserve order. (Does NOT reset index.)"""
    out = df.copy()
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
        out = out.dropna(subset=[time_col]).sort_values(time_col)
    return out

def _filter_trials(df: pd.DataFrame, trial_col="trial", trials=None, first_n_trials=None) -> pd.DataFrame:
    """Filter to first N trials or a specified set of trials."""
    out = df.copy()
    if trials is None and first_n_trials is None:
        return out

    if trial_col not in out.columns:
        raise ValueError(f"'{trial_col}' column not found, cannot filter by trials.")

    if first_n_trials is not None:
        trial_ids = pd.Index(out[trial_col].dropna().unique()).tolist()
        keep = set(trial_ids[: int(first_n_trials)])
        out = out[out[trial_col].isin(keep)].copy()

    if trials is not None:
        out = out[out[trial_col].isin(list(trials))].copy()

    return out


def _add_motion_bout_id(base: pd.DataFrame, movement_type, motion_col="roi_motion", out_col="_motion_bout_id") -> pd.DataFrame:
    """Add bout ids for contiguous segments where motion_col == movement_type."""
    out = base.copy()
    is_move = (out[motion_col].to_numpy() == movement_type)
    enter = np.r_[False, (is_move[1:] & ~is_move[:-1])]
    out[out_col] = np.cumsum(enter) * is_move  # 0 outside; 1..N inside bouts
    return out


def _style_psth_axes(ax0, ax1):
    for ax in (ax0, ax1):
        ax.grid(False)
        ax.set_axis_off()


# ----------------------------
# Core analysis functions
# ----------------------------


def find_state_bouts(df, state_col="states_smoothed", target_state=0,
                    trial_col="trial", time_col="Unnamed: 0"):
    """
    Returns a DataFrame of bouts with:
      trial, onset_row (index in df), offset_row, bout_len
    Respects trial boundaries if trial_col exists.
    """
    base = df.copy()
    base[time_col] = pd.to_datetime(base[time_col], errors="coerce")
    base = base.dropna(subset=[time_col]).sort_values(time_col)

    bouts = []

    group_iter = base.groupby(trial_col, sort=False) if trial_col in base.columns else [(None, base)]

    for trial_id, g in group_iter:
        s = g[state_col].to_numpy()
        is_state = (s == target_state)

        # run-length segmentation
        starts = np.where(is_state & ~np.r_[False, is_state[:-1]])[0]
        ends   = np.where(is_state & ~np.r_[is_state[1:], False])[0]

        for st, en in zip(starts, ends):
            onset_row_ix  = int(g.iloc[st]["__row_ix__"])
            offset_row_ix = int(g.iloc[en]["__row_ix__"])
            bouts.append({
                "trial": trial_id,
                "start_row_ix": onset_row_ix,
                "end_row_ix": offset_row_ix,
                "bout_len": int(en - st + 1)
            })

    return pd.DataFrame(bouts)

def make_bout_aligned_state_matrix(
    df,
    bouts_df,
    *,
    state_col="states",
    align="onset",
    pre=120,
    post=180,
    sort_by="bout_len_desc",
    time_col="Unnamed: 0",
    trial_col="trial",
):
    base = df.copy()
    base[time_col] = pd.to_datetime(base[time_col], errors="coerce")
    base = base.dropna(subset=[time_col]).sort_values(time_col)

    idx_to_pos = pd.Series(np.arange(len(base)), index=base.index).to_dict()
    states = pd.to_numeric(base[state_col], errors="coerce").to_numpy()
    trials = base[trial_col].to_numpy() if trial_col in base.columns else None

    win_len = pre + post + 1
    S = []
    meta = []

    anchor_col = "onset_idx" if align == "onset" else "offset_idx"

    for _, r in bouts_df.iterrows():
        anchor_idx = r[anchor_col]
        pos = idx_to_pos.get(anchor_idx, None)
        if pos is None:
            continue

        lo, hi = pos - pre, pos + post
        if lo < 0 or hi >= len(states):
            continue

        w = states[lo:hi+1].astype(float)

        # NEW: enforce trial boundary by masking outside trial_id
        if trials is not None and trial_col in r:
            trial_id = r[trial_col]
            tmask = (trials[lo:hi+1] == trial_id)
            w[~tmask] = np.nan

        S.append(w)
        meta.append(r.to_dict())

    if not S:
        return np.empty((0, win_len)), pd.DataFrame(meta), np.linspace(-pre, post, win_len)

    S = np.vstack(S)
    meta = pd.DataFrame(meta)

    if sort_by == "bout_len_desc" and "bout_len" in meta.columns:
        order = np.argsort(-meta["bout_len"].to_numpy())
        S = S[order]
        meta = meta.iloc[order].reset_index(drop=True)

    dt = base[time_col].diff().median()
    dt_s = pd.to_timedelta(dt).total_seconds() if pd.notna(dt) else 1.0
    t = np.arange(-pre, post + 1) * dt_s

    return S, meta, t

def make_bout_aligned_matrix(
    df,
    bouts_df,
    *,
    value_col="deltaF_z",
    align="onset",              # "onset" -> start_row_ix, "offset" -> end_row_ix
    pre=60,
    post=60,
    sort_by="bout_len_desc",
    time_col="Unnamed: 0",
    row_ix_col="__row_ix__",
    trial_col="trial",
    enforce_trial="mask",       # "mask" (recommended), "drop", or "none"
    min_finite_frac=0.8,        # only used when enforce_trial="mask"
):
    """
    Build an aligned matrix M around bout anchors using stable row ids.

    Expected bouts_df columns:
      - start_row_ix, end_row_ix (these should be values from df[row_ix_col])
      - optional: trial, bout_len, etc.

    enforce_trial:
      - "mask": set samples outside the bout's trial to NaN, keep window if >= min_finite_frac finite
      - "drop": drop windows that cross trial boundaries
      - "none": ignore trial boundaries
    """
    base = df.copy()

    # sort by time (but do NOT rely on base.index for alignment)
    if time_col in base.columns:
        base[time_col] = pd.to_datetime(base[time_col], errors="coerce")
        base = base.dropna(subset=[time_col]).sort_values(time_col)

    if row_ix_col not in base.columns:
        raise ValueError(f"df must contain '{row_ix_col}'. Call _ensure_row_ix(df) first.")

    if value_col not in base.columns:
        raise ValueError(f"df missing '{value_col}'")

    if len(bouts_df) == 0:
        win_len = pre + post + 1
        dt = base[time_col].diff().median() if time_col in base.columns else pd.Timedelta(seconds=1)
        dt_s = pd.to_timedelta(dt).total_seconds() if pd.notna(dt) else 1.0
        t = np.arange(-pre, post + 1) * dt_s
        return np.empty((0, win_len)), pd.DataFrame(), t

    # Map stable row id -> row position in the time-sorted base
    ix_to_pos = pd.Series(np.arange(len(base)), index=base[row_ix_col]).to_dict()

    values = pd.to_numeric(base[value_col], errors="coerce").to_numpy(dtype=float)
    trials = base[trial_col].to_numpy() if (trial_col in base.columns) else None

    win_len = pre + post + 1
    M = []
    meta = []

    anchor_col = "start_row_ix" if align == "onset" else "end_row_ix"
    if anchor_col not in bouts_df.columns:
        raise KeyError(f"bouts_df missing '{anchor_col}'. Available: {list(bouts_df.columns)}")

    for _, r in bouts_df.iterrows():
        anchor_ix = r[anchor_col]
        if pd.isna(anchor_ix):
            continue
        anchor_ix = int(anchor_ix)  # this is a stable row id (same domain as df[row_ix_col])

        pos = ix_to_pos.get(anchor_ix, None)
        if pos is None:
            continue

        lo, hi = pos - pre, pos + post
        if lo < 0 or hi >= len(values):
            continue

        w = values[lo : hi + 1].copy()
        if w.shape[0] != win_len:
            continue

        # Handle trial boundaries
        if enforce_trial != "none" and (trials is not None) and (trial_col in bouts_df.columns):
            trial_id = r[trial_col]

            if enforce_trial == "drop":
                # drop if any sample in window is not this trial
                if not np.all(trials[lo : hi + 1] == trial_id):
                    continue

            elif enforce_trial == "mask":
                # mask samples outside trial to NaN, require enough finite samples
                tmask = (trials[lo : hi + 1] == trial_id)
                w[~tmask] = np.nan
                if np.mean(np.isfinite(w)) < float(min_finite_frac):
                    continue

            else:
                raise ValueError("enforce_trial must be one of: 'mask', 'drop', 'none'")

        # Optional: reject NaNs in the signal (outside trial masking)
        # If you want to be strict, change this to: if np.any(~np.isfinite(w)): continue
        if not np.any(np.isfinite(w)):
            continue

        M.append(w)
        meta.append(r.to_dict())

    # time axis in seconds (based on median dt)
    if time_col in base.columns:
        dt = base[time_col].diff().median()
        dt_s = pd.to_timedelta(dt).total_seconds() if pd.notna(dt) else 1.0
    else:
        dt_s = 1.0
    t = np.arange(-pre, post + 1) * dt_s

    if not M:
        return np.empty((0, win_len)), pd.DataFrame(meta), t

    M = np.vstack(M)
    meta = pd.DataFrame(meta)

    if sort_by == "bout_len_desc" and "bout_len" in meta.columns:
        order = np.argsort(-pd.to_numeric(meta["bout_len"], errors="coerce").to_numpy())
        M = M[order]
        meta = meta.iloc[order].reset_index(drop=True)

    return M, meta, t

def classify_approach_retreat_with_ddist(
    df,
    *,
    exp_pos=("x_pos","y_pos"),
    cons_pos=("x_pos_con","y_pos_con"),
    disp_cols=("forward_x","forward_y"),
    time_col="Unnamed: 0",
    speed_thresh=1e-3,
    align_thresh=0.5,
    ddist_thresh=80,    # px/s  <-- THIS is where "100 px" belongs
    out_col="cons_motion_dd",
):
    out = df.copy()

    # time deltas (seconds)
    t = pd.to_datetime(out[time_col], errors="coerce")
    dt = t.diff().dt.total_seconds().to_numpy()
    dt[dt <= 0] = np.nan

    ex = out[exp_pos[0]].to_numpy(float)
    ey = out[exp_pos[1]].to_numpy(float)
    cx = out[cons_pos[0]].to_numpy(float)
    cy = out[cons_pos[1]].to_numpy(float)

    # distance and its derivative (px/s)
    dist = np.hypot(cx - ex, cy - ey)
    ddist_dt = np.r_[np.nan, np.diff(dist)] / dt

    # velocity of experimental mouse (arena coords)
    dx = out[disp_cols[0]].to_numpy(float)
    dy = out[disp_cols[1]].to_numpy(float)

    # vector to conspecific
    rx = cx - ex
    ry = cy - ey

    vmag = np.hypot(dx, dy)
    rmag = np.hypot(rx, ry)

    valid = np.isfinite(vmag) & np.isfinite(rmag) & np.isfinite(ddist_dt)
    valid &= (vmag > speed_thresh) & (rmag > 1e-6)

    cos = np.full(len(out), np.nan, dtype=float)
    cos[valid] = (dx[valid]*rx[valid] + dy[valid]*ry[valid]) / (vmag[valid]*rmag[valid])

    labels = np.zeros(len(out), dtype=int)

    # approach: toward + closing
    labels[valid & (cos > align_thresh) & (ddist_dt < -ddist_thresh)] = 1
    # retreat: away + opening
    labels[valid & (cos < -align_thresh) & (ddist_dt >  ddist_thresh)] = -1

    out["cons_cos"] = cos
    out["dist_cons"] = dist
    out["ddist_dt"] = ddist_dt
    out[out_col] = labels
    return out

def classify_motion_relative_to_roi(
    df,
    roi_points,
    pos_cols=("x_pos", "y_pos"),
    disp_cols=("forward_x", "forward_y"),
    speed_thresh=1e-2,
    align_thresh=0.2,
    label_col="roi_motion",
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

    disp_mag = np.hypot(dx, dy)
    roi_mag = np.hypot(vx, vy)

    valid = (disp_mag > speed_thresh) & (roi_mag > 1e-6)

    cos_align = np.zeros_like(disp_mag)
    cos_align[valid] = (
        (dx[valid] * vx[valid] + dy[valid] * vy[valid])
        / (disp_mag[valid] * roi_mag[valid])
    )

    labels = np.zeros(len(df), dtype=int)
    labels[cos_align > align_thresh] = 1
    labels[cos_align < -align_thresh] = -1
    df[label_col] = labels
    return df

def label_bouts_by_overlap(
    bouts,
    motion_bouts,
    *,
    bouts_start_col="onset_idx",
    bouts_end_col="offset_idx",
    motion_start_col="start_row_ix",
    motion_end_col="end_row_ix",
    motion_type_col="bout_type",
    min_overlap_frac=0.55,
):
    """
    Label each bout in `bouts` by which motion bout overlaps it the most.
    Works with arbitrary start/end column names.
    """
    if len(bouts) == 0:
        out = bouts.copy()
        out["cons_bout_label"] = []
        return out

    if len(motion_bouts) == 0:
        out = bouts.copy()
        out["cons_bout_label"] = "none"
        out["cons_overlap_frac"] = 0.0
        return out

    for c in (bouts_start_col, bouts_end_col):
        if c not in bouts.columns:
            raise KeyError(f"`bouts` missing column '{c}'. Available: {list(bouts.columns)}")
    for c in (motion_start_col, motion_end_col):
        if c not in motion_bouts.columns:
            raise KeyError(f"`motion_bouts` missing column '{c}'. Available: {list(motion_bouts.columns)}")
    if motion_type_col not in motion_bouts.columns:
        raise KeyError(f"`motion_bouts` missing column '{motion_type_col}'.")

    mb = motion_bouts[[motion_start_col, motion_end_col, motion_type_col]].copy()

    def overlap_frac(a0, a1, b0, b1):
        inter = max(0, min(a1, b1) - max(a0, b0) + 1)
        denom = (a1 - a0 + 1)
        return inter / denom if denom > 0 else 0.0

    labels = []
    best_fracs = []

    for _, r in bouts.iterrows():
        a0 = int(r[bouts_start_col])
        a1 = int(r[bouts_end_col])

        best_label = "none"
        best_frac = 0.0

        for _, m in mb.iterrows():
            b0 = int(m[motion_start_col])
            b1 = int(m[motion_end_col])
            frac = overlap_frac(a0, a1, b0, b1)
            if frac > best_frac:
                best_frac = frac
                best_label = m[motion_type_col]

        if best_frac < min_overlap_frac:
            best_label = "none"

        labels.append(best_label)
        best_fracs.append(best_frac)

    out = bouts.copy()
    out["cons_bout_label"] = labels
    out["cons_overlap_frac"] = best_fracs
    return out

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

import numpy as np
import pandas as pd

def extract_approach_bouts(
    df,
    *,
    row_ix_col="__row_ix__",
    dist_col="dist_cons",
    ddist_col="ddist_dt",
    cos_col="cons_cos",
    align_start_app=0.75,
    align_stop_app=0.45,
    ddist_start_app=60.0,   # start if ddist_dt <= -ddist_start_app
    ddist_stop_app=30.0,    # continue while ddist_dt <= -ddist_stop_app
    max_gap_frames=10,
    min_len_frames=6,
    min_peak_close=20.0,    # <-- require max closure vs start (px)
):
    """
    Extract approach bouts from continuous ddist_dt + alignment (cos), with hysteresis.
    Uses peak closure criterion:
        peak_close = dist_start - min(dist during bout)
    to avoid missing bouts that close then partially re-open before ending.
    """
    if row_ix_col not in df.columns:
        raise ValueError(f"df must contain {row_ix_col}. Call _ensure_row_ix(df) first.")
    for c in (dist_col, ddist_col, cos_col):
        if c not in df.columns:
            raise ValueError(f"Missing column {c}.")

    row_ix = df[row_ix_col].to_numpy()
    dist  = df[dist_col].to_numpy(float)
    ddist = df[ddist_col].to_numpy(float)
    cos   = df[cos_col].to_numpy(float)

    finite = np.isfinite(dist) & np.isfinite(ddist) & np.isfinite(cos)

    # approach evidence: toward (cos positive) AND closing fast (ddist negative)
    start_app = finite & (cos >= align_start_app) & (ddist <= -ddist_start_app)
    keep_app  = finite & (cos >= align_stop_app)  & (ddist <= -ddist_stop_app)

    # hysteresis segmentation (same structure as retreat)
    in_bout = np.zeros(len(df), dtype=bool)
    i = 0
    while i < len(df):
        if start_app[i]:
            j = i + 1
            gaps = 0
            while j < len(df):
                if keep_app[j]:
                    gaps = 0
                    j += 1
                    continue
                gaps += 1
                if gaps <= max_gap_frames:
                    j += 1
                    continue
                break

            end = j - 1
            while end > i and (not keep_app[end]):
                end -= 1

            in_bout[i:end+1] = True
            i = end + 1
        else:
            i += 1

    idx = np.where(in_bout)[0]
    if idx.size == 0:
        return pd.DataFrame(columns=[
            "bout_type","start_i","end_i","start_row_ix","end_row_ix",
            "len_frames","dist_start","dist_end","peak_close"
        ])

    # run-length encode contiguous True segments
    starts = [idx[0]]
    ends = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            ends.append(a)
            starts.append(b)
    ends.append(idx[-1])

    bouts = []
    for s, e in zip(starts, ends):
        L = e - s + 1
        if L < min_len_frames:
            continue

        d0 = dist[s]
        peak_close = d0 - np.nanmin(dist[s:e+1])   # positive = got closer
        if peak_close < min_peak_close:
            continue

        bouts.append({
            "bout_type": "approach",
            "start_i": int(s),
            "end_i": int(e),
            "start_row_ix": int(row_ix[s]),
            "end_row_ix": int(row_ix[e]),
            "len_frames": int(L),
            "dist_start": float(d0),
            "dist_end": float(dist[e]),
            "peak_close": float(peak_close),
        })

    return pd.DataFrame(bouts)
def extract_retreat_bouts(
    df,
    *,
    row_ix_col="__row_ix__",
    dist_col="dist_cons",
    ddist_col="ddist_dt",
    cos_col="cons_cos",
    align_start_ret=0.75,
    align_stop_ret=0.45,
    ddist_start_ret=60.0,
    ddist_stop_ret=30.0,
    max_gap_frames=10,
    min_len_frames=6,
    min_peak_open=20.0,   # <-- retreat requirement (px)
):
    if row_ix_col not in df.columns:
        raise ValueError(f"df must contain {row_ix_col}. Call _ensure_row_ix(df) first.")
    for c in (dist_col, ddist_col, cos_col):
        if c not in df.columns:
            raise ValueError(f"Missing column {c}.")

    row_ix = df[row_ix_col].to_numpy()
    dist  = df[dist_col].to_numpy(float)
    ddist = df[ddist_col].to_numpy(float)
    cos   = df[cos_col].to_numpy(float)

    finite = np.isfinite(dist) & np.isfinite(ddist) & np.isfinite(cos)

    start_ret = finite & (cos <= -align_start_ret) & (ddist >= ddist_start_ret)
    keep_ret  = finite & (cos <= -align_stop_ret)  & (ddist >= ddist_stop_ret)

    in_bout = np.zeros(len(df), dtype=bool)
    i = 0
    while i < len(df):
        if start_ret[i]:
            j = i + 1
            gaps = 0
            while j < len(df):
                if keep_ret[j]:
                    gaps = 0
                    j += 1
                    continue
                gaps += 1
                if gaps <= max_gap_frames:
                    j += 1
                    continue
                break

            end = j - 1
            while end > i and (not keep_ret[end]):
                end -= 1

            in_bout[i:end+1] = True
            i = end + 1
        else:
            i += 1

    idx = np.where(in_bout)[0]
    if idx.size == 0:
        return pd.DataFrame(columns=[
            "bout_type","start_i","end_i","start_row_ix","end_row_ix",
            "len_frames","dist_start","dist_end","peak_open"
        ])

    # run-length encode
    starts = [idx[0]]
    ends = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            ends.append(a)
            starts.append(b)
    ends.append(idx[-1])

    bouts = []
    for s, e in zip(starts, ends):
        L = e - s + 1
        if L < min_len_frames:
            continue

        d0 = dist[s]
        peak_open = np.nanmax(dist[s:e+1]) - d0
        if peak_open < min_peak_open:
            continue

        bouts.append({
            "bout_type": "retreat",
            "start_i": int(s),
            "end_i": int(e),
            "start_row_ix": int(row_ix[s]),
            "end_row_ix": int(row_ix[e]),
            "len_frames": int(L),
            "dist_start": float(d0),
            "dist_end": float(dist[e]),
            "peak_open": float(peak_open),
        })

    return pd.DataFrame(bouts)

def plot_position_by_state(
    hmm_array,
    outpath,
    state_col="states",
    movement_type=-1,
    condition="Appetitive",
    decimate=2,
    rasterize_scatter=False,
    rasterize_quiver=False,
    quiver_stride=5,
    quiver_scale=0.4,
    scatter_size=10,
    scatter_alpha=1,
    quiver_alpha=0.35,
    trial_col="trial",
    trials=None,
    first_n_trials=None,
):
    df = hmm_array[hmm_array["olfactory_stim"] == condition].copy()
    df = _filter_trials(df, trial_col=trial_col, trials=trials, first_n_trials=first_n_trials)

    states = df[state_col].dropna().unique()
    STATE_COLORS = make_state_colors_from_cmap(states, cmp_discrete)

    fig, ax = plt.subplots(figsize=(8, 8))

    for state in sorted(df[state_col].dropna().unique()):
        if movement_type is not None:
            df_state = df[(df[state_col] == state) & (df["roi_motion"] == movement_type)].copy()
        else:
            df_state = df[df[state_col] == state].copy()
        if df_state.empty:
            continue

        pts = df_state.iloc[::decimate] if (decimate and decimate > 1) else df_state
        ax.scatter(
            pts["x_pos"],
            pts["y_pos"],
            s=scatter_size,
            color=STATE_COLORS[state],
            label=state,
            alpha=scatter_alpha,
            linewidths=0,
            rasterized=bool(rasterize_scatter),
        )

        # qdf = df_state.iloc[::quiver_stride] if (quiver_stride and quiver_stride > 1) else df_state
        # qv = ax.quiver(
        #     qdf["x_pos"],
        #     qdf["y_pos"],
        #     qdf["forward_x"],
        #     qdf["forward_y"],
        #     angles="xy",
        #     scale=quiver_scale,
        #     scale_units="xy",
        #     headwidth=6,
        #     headlength=7,
        #     headaxislength=6,
        #     alpha=quiver_alpha,
        # )
        # if rasterize_quiver:
        #     qv.set_rasterized(True)

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

    out_file = Path(outpath) / f"position_by_state_movement_{movement_type}_type_{condition}{suffix}.svg"
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
    one_hit_per_motion_bout=True,
    motion_col="roi_motion",
    require_col=None,            # NEW
    require_value=True,          # NEW
):
    # base: time-contiguous
    _require_row_ix(hmm_array)
    base = _sort_time(hmm_array, time_col=time_col)
    if state_col not in base.columns or signal_col not in base.columns:
        raise ValueError(f"Missing columns: {state_col} or {signal_col}")

    base = _add_motion_bout_id(base, movement_type=movement_type, motion_col=motion_col, out_col="_motion_bout_id")

    # sub: for onset detection only
    sub = base[(base["olfactory_stim"] == stim) & (base[motion_col] == movement_type)].copy()

    if require_col is not None:
        if require_col not in sub.columns:
            raise ValueError(f"'{require_col}' not found in dataframe.")
        sub = sub[sub[require_col] == require_value].copy()

    sub = _filter_trials(sub, trial_col=trial_col, trials=trials, first_n_trials=first_n_trials)

    pre = int(window_seconds * fps)
    post = int(window_seconds * fps)
    win_len = pre + post + 1
    x = np.arange(-pre, post + 1) / float(fps)

    ix_to_pos = pd.Series(np.arange(len(base)), index=base["__row_ix__"]).to_dict()
    base_signal = pd.to_numeric(base[signal_col], errors="coerce").to_numpy()

    for target_state in sorted(sub[state_col].dropna().unique()):
        s = sub[state_col].to_numpy()
        idx = np.flatnonzero(s == target_state)
        if idx.size == 0:
            continue

        splits = np.where(np.diff(idx) != 1)[0] + 1
        runs = np.split(idx, splits)

        # collect (onset_row_ix, motion_bout_id)
        candidates = []
        for r in runs:
            if len(r) < min_state_len:
                continue
            onset_row_ix = int(sub.iloc[r[0]]["__row_ix__"])
            mb = int(sub.iloc[r[0]]["_motion_bout_id"])
            if mb == 0:
                continue
            candidates.append((onset_row_ix, mb))

        if not candidates:
            continue

        # optionally de-duplicate per motion bout
        if one_hit_per_motion_bout:
            seen = set()
            onset_row_ixs = []
            for onset_row_ix, mb in sorted(candidates, key=lambda t: t[0]):
                if mb in seen:
                    continue
                seen.add(mb)
                onset_row_ixs.append(onset_row_ix)
        else:
            onset_row_ixs = [c[0] for c in candidates]

        aligned = []
        for onset_ix in onset_row_ixs:
            pos = ix_to_pos.get(onset_ix)
            if pos is None:
                continue

            lo, hi = pos - pre, pos + post
            if lo < 0 or hi >= len(base_signal):
                continue

            seg = base_signal[lo : hi + 1]
            if seg.shape[0] != win_len:
                continue

            b = min(baseline_frames, pre)
            base0 = np.nanmean(seg[:b]) if b > 0 else 0.0
            aligned.append(seg - base0)

        if not aligned:
            continue

        aligned = np.vstack(aligned)

        baseline_vals = aligned[:, x < 0].ravel()
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

        ax1.plot(x, mean, linewidth=2)
        ax1.fill_between(x, mean - sem, mean + sem, alpha=0.25)
        ax1.axvline(0, linestyle="--")
        ax1.set_xlabel("Time (s) relative to onset")
        ax1.set_ylabel(f"Mean {signal_col}")

        _style_psth_axes(ax0, ax1)

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

def plot_state_raster_with_occupancy(
    S, t,
    *,
    title="",
    state_order=None,          # e.g. [0,1,2,3,4]
    state_colors=None,         # dict {state: color} or None
    figsize=(3.0, 3.8),
    show_vline_at_zero=True,
    raster_alpha=1.0,
    legend=False,
):
    """
    S: (n_trials, n_time) state labels (int or float with NaNs)
    t: (n_time,) time (seconds)

    Top: raster (imshow) of states over time (rows=trials)
    Bottom: occupancy fraction per state vs time
    """
    S = np.asarray(S)
    t = np.asarray(t)

    if S.ndim != 2:
        raise ValueError("S must be 2D (n_trials, n_time)")
    if t.ndim != 1 or t.shape[0] != S.shape[1]:
        raise ValueError("t must be 1D and match S.shape[1]")

    finite = S[np.isfinite(S)].astype(int)
    if finite.size == 0:
        raise ValueError("S contains no finite state labels.")

    if state_order is None:
        state_order = sorted(np.unique(finite).tolist())
    else:
        state_order = list(state_order)

    # map raw state -> 0..K-1
    lut = {s: i for i, s in enumerate(state_order)}
    Sm = np.full(S.shape, np.nan, dtype=float)
    for s, i in lut.items():
        mask = np.isfinite(S) & (S.astype(int) == s)
        Sm[mask] = i

    K = len(state_order)

    # colors
    if state_colors is None:
        base = plt.get_cmap("Spectral", K)
        
        cmap = ListedColormap(base(np.arange(K)))
        cmap = cmap.copy()
        cmap.set_bad(color="black")
    else:
        cmap = ListedColormap([state_colors[s] for s in state_order])
        cmap = cmap.copy()
        cmap.set_bad(color="black")

    bounds = np.arange(-0.5, K + 0.5, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.2, 1.2], hspace=0.15)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_o = fig.add_subplot(gs[1, 0], sharex=ax_r)

    # --- raster ---
    extent = [t[0], t[-1], S.shape[0], 0]  # row 0 at top
    im = ax_r.imshow(
        Sm,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        extent=extent,
        alpha=raster_alpha,
    )

    if show_vline_at_zero:
        ax_r.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.6)
        ax_o.axvline(0, color="k", linestyle="--", linewidth=1, alpha=0.6)

    ax_r.set_title(title)
    ax_r.set_ylabel("Trials")
    ax_r.set_xticklabels([])

   
    # clean
    for ax in (ax_r, ax_o):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # optional colorbar (small)
    cbar = fig.colorbar(im, ax=ax_r, fraction=0.05, pad=0.02, ticks=np.arange(K))
    cbar.ax.set_yticklabels([str(s) for s in state_order])
    cbar.set_label("State")

    if legend:
        ax_o.legend(frameon=False, ncol=min(K, 5), fontsize=8)

    fig.tight_layout()
    return fig, (ax_r, ax_o)

def plot_state_heatmap_with_endlines_and_mean(
    M,
    meta,
    t,
    *,
    bout_len_col="bout_len",
    title="",
    cmap="plasma",
    vclip=(2, 98),
    figsize=(4.2, 6.5),
    endline_color="w",
    endline_alpha=0.9,
    endline_lw=0.8,
    show_vline_at_zero=True,
    mean_color="k",
    sem_alpha=0.25,
):
    """
    M: (n_bouts, n_timepoints) aligned matrix (e.g., deltaF_z)
    meta: DataFrame with at least a 'bout_len' column for each row in M
    t: (n_timepoints,) time in seconds (same window as M)

    Draws:
      - heatmap
      - a short vertical line per row at state end (based on bout_len)
      - mean ± SEM subplot for the same window
    """

    if M.shape[0] == 0:
        raise ValueError("No bouts/windows to plot.")
    if meta is None or bout_len_col not in meta.columns:
        raise ValueError(f"meta must include '{bout_len_col}' to draw end lines.")
    if len(meta) != M.shape[0]:
        raise ValueError("meta rows must match M rows (same bout ordering).")

    t = np.asarray(t)
    n_bouts, n_t = M.shape

    # color scaling
    vmin, vmax = np.percentile(M[np.isfinite(M)], vclip)

    # mean / SEM across bouts
    alpha = 1.0
    mean = np.nanmean(M, axis=0)
    mean_smooth = gaussian_filter1d(mean, sigma=1.25)  # smooth mean for better visualization
    sem = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(M), axis=0))
    sem_smooth = gaussian_filter1d(sem, sigma=1.25)

    # Layout: heatmap + mean subplot
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1.2], hspace=0.15)
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_mu = fig.add_subplot(gs[1, 0], sharex=ax_hm)

    # heatmap
    extent = [t[0], t[-1], n_bouts, 0]  # top row = 0
    ax_hm.imshow(
        M,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )

    if show_vline_at_zero:
        ax_hm.axvline(0, color="w", linewidth=1)
        ax_mu.axvline(0, color="k", linewidth=1, linestyle="--", alpha=0.6)

    ax_hm.set_title(title)
    ax_hm.set_ylabel("Bouts")

    # --- 1) per-bout end marker for the aligned state ---
    # infer dt from t (assumes uniform spacing)
    if len(t) >= 2:
        dt = np.median(np.diff(t))
    else:
        dt = 1.0

    bout_len = meta[bout_len_col].to_numpy()

    # End time relative to onset:
    # if bout_len is in samples, end is (len-1)*dt after onset.
    t_end = (bout_len - 1) * dt

    # draw a short vertical tick within each row
    # row i occupies y in [i, i+1] in extent coordinates (top is 0)
    for i, te in enumerate(t_end):
        if te < t[0] or te > t[-1]:
            continue
        # draw a small segment centered in that row
        y0 = i + 0.15
        y1 = i + 0.85
        ax_hm.plot([te, te], [y0, y1],
                   color=endline_color, alpha=endline_alpha, linewidth=endline_lw)

    # --- 2) mean ± SEM subplot ---
    ax_mu.plot(t, mean_smooth, linewidth=2)
    ax_mu.fill_between(t,
                       mean_smooth - sem_smooth,
                       mean_smooth + sem_smooth,
                       alpha=sem_alpha,
                       color=mean_color)
    ax_mu.set_xlabel("Time (s)")
    ax_mu.set_ylabel("Mean")
    ax_mu.grid(False)
    ax_mu.spines["top"].set_visible(False)
    ax_mu.spines["right"].set_visible(False)


    # cleaner look
    plt.setp(ax_hm.get_xticklabels(), visible=False)

    fig.tight_layout()
    return fig, (ax_hm, ax_mu)


def plot_aligned_heatmap_and_mean(
    M,
    t,
    *,
    title="",
    cmap="plasma",
    vclip=(2, 98),
    figsize=(4.2, 6.5),
    show_vline_at_zero=True,
    mean_color="k",
    sem_alpha=0.25,
    smooth_sigma=1.25,
):
    if M.shape[0] == 0:
        raise ValueError("No events/windows to plot.")

    t = np.asarray(t)

    finite = M[np.isfinite(M)]
    if finite.size == 0:
        raise ValueError("M contains no finite values.")

    vmin, vmax = np.percentile(finite, vclip)

    mean = np.nanmean(M, axis=0)
    sem = np.nanstd(M, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(M), axis=0))

    mean_s = gaussian_filter1d(mean, sigma=smooth_sigma) if smooth_sigma else mean
    sem_s = gaussian_filter1d(sem, sigma=smooth_sigma) if smooth_sigma else sem

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1.2], hspace=0.15)
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_mu = fig.add_subplot(gs[1, 0], sharex=ax_hm)

    extent = [t[0], t[-1], M.shape[0], 0]
    ax_hm.imshow(
        M,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )

    if show_vline_at_zero:
        ax_hm.axvline(0, color="w", linewidth=1)
        ax_mu.axvline(0, color="k", linewidth=1, linestyle="--", alpha=0.6)

    ax_hm.set_title(title)
    ax_hm.set_ylabel("Events")

    ax_mu.plot(t, mean_s, linewidth=2)
    ax_mu.fill_between(t, mean_s - sem_s, mean_s + sem_s, alpha=sem_alpha, color=mean_color)
    ax_mu.set_xlabel("Time (s)")
    ax_mu.set_ylabel("Mean")
    ax_mu.grid(False)
    ax_mu.spines["top"].set_visible(False)
    ax_mu.spines["right"].set_visible(False)

    plt.setp(ax_hm.get_xticklabels(), visible=False)
    fig.tight_layout()
    return fig, (ax_hm, ax_mu)
# ----------------------------
# Pattern-hit PSTH utilities
# ----------------------------

def find_bout_pattern_hits(df, state_col, pattern=(0, 4, 1)):
    """
    Finds occurrences of pattern in consecutive BOUTS (run-length compressed states).
    Returns onset '__row_ix__' of the first state in the pattern.
    """
    if "__row_ix__" not in df.columns:
        raise ValueError("df must contain '__row_ix__' for pattern-hit functions.")

    g = df.copy()
    g["_state_prev"] = g[state_col].shift(1)
    g["_new_bout"] = (g[state_col] != g["_state_prev"]).astype(int)
    g["_bout_id"] = g["_new_bout"].cumsum()

    gb = g.groupby("_bout_id", sort=False)
    bouts = pd.DataFrame(
        {
            "state": gb[state_col].first(),
            "start_ix": gb["__row_ix__"].first(),
        }
    ).reset_index(drop=True)

    states = bouts["state"].to_numpy()
    p = np.array(pattern)

    hits = []
    if len(states) >= len(p):
        for i in range(len(states) - len(p) + 1):
            if np.array_equal(states[i : i + len(p)], p):
                hits.append(int(bouts.loc[i, "start_ix"]))
    return hits

def compute_speed(df):
    dx = df["x_pos"].diff()
    dy = df["y_pos"].diff()
    dt = pd.to_datetime(df["Unnamed: 0"]).diff().dt.total_seconds()
    speed = np.sqrt(dx**2 + dy**2) / dt
    df["speed_raw"] = speed
    return df

def extract_aligned_windows(base_df, onset_row_ixs, value_col, pre=30, post=90):
    """Extract windows from a time-sorted base_df using stable '__row_ix__' ids."""
    if "__row_ix__" not in base_df.columns:
        raise ValueError("base_df must contain '__row_ix__'")

    win_len = pre + post + 1
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
        aligned.append(values[lo : hi + 1])
        kept.append(onset_ix)

    if not aligned:
        return np.empty((0, win_len)), kept
    return np.vstack(aligned), kept


def plot_bout_xy_trajectories(
    df,
    meta,
    *,
    x_col="x_pos",
    y_col="y_pos",
    time_col="Unnamed: 0",
    trial_col="trial",
    onset_col="onset_idx",
    offset_col="offset_idx",
    fx_col="forward_x",
    fy_col="forward_y",
    color_by="bout_len",  # "bout_len" or None
    alpha=0.35,
    lw=2.0,
    figsize=(6, 6),
    title="Bout trajectories (x,y)",
    quiver_stride=3,
):
    base = df.copy()
    base[time_col] = pd.to_datetime(base[time_col], errors="coerce")
    base = base.dropna(subset=[time_col]).sort_values(time_col)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")

    # colormap if desired
    if color_by is not None and color_by in meta.columns:
        vals = meta[color_by].to_numpy()
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("mako")
    else:
        norm = cmap = None

    for _, r in meta.iterrows():
        lo = r[onset_col]
        hi = r[offset_col]

        # slice by index label range
        seg = base.loc[lo:hi] if lo in base.index and hi in base.index else None
        if seg is None or len(seg) < 2:
            continue

        x = seg[x_col].to_numpy()
        y = seg[y_col].to_numpy()
        if np.any(pd.isna(x)) or np.any(pd.isna(y)):
            continue

        c = cmap(norm(r[color_by])) if cmap is not None else "k"
        ax.plot(x, y, color=c, alpha=alpha, linewidth=lw)

        # mark onset point
        ax.scatter([x[0]], [y[0]], s=15, color=c, alpha=min(alpha + 0.2, 1.0))
        
        # segqv = seg.iloc[::quiver_stride] if (quiver_stride and quiver_stride > 1) else seg
        # x = segqv[x_col].to_numpy()
        # y = segqv[y_col].to_numpy()
        # fx = segqv[fx_col].to_numpy()
        # fy = segqv[fy_col].to_numpy()

        # ax.quiver(
        #     x,
        #     y,
        #     fx,
        #     fy,
        #     angles="xy",
        #     scale=0.4,
        #     scale_units="xy",
        #     headwidth=12,
        #     headlength=10,
        #     headaxislength=10,
        #     alpha=0.35,
        # )

    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(False)
    ax.set_axis_off()


    # optional colorbar
    if cmap is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_by)

    plt.tight_layout()
    return fig, ax
def plot_bout_xy_quiver_clean(
    df,
    meta,
    *,
    x_col="x_pos",
    y_col="y_pos",
    fx_col="forward_x",
    fy_col="forward_y",
    onset_col="onset_idx",
    offset_col="offset_idx",
    step=10,              # strong subsampling
    arrow_scale=25,       # larger number = shorter arrows
    normalize=True,       # show direction only
    color="black",
    alpha=0.6,
    figsize=(6, 6),
    title="State 4 bout trajectories",
):
    base = df.copy()
    idx_to_pos = pd.Series(np.arange(len(base)), index=base.index).to_dict()

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")

    for _, r in meta.iterrows():
        on = r[onset_col]
        off = r[offset_col]

        if on not in idx_to_pos or off not in idx_to_pos:
            continue

        p0, p1 = idx_to_pos[on], idx_to_pos[off]
        if p1 <= p0:
            continue

        seg = base.iloc[p0:p1+1:step]

        x = seg[x_col].to_numpy()
        y = seg[y_col].to_numpy()
        fx = seg[fx_col].to_numpy()
        fy = seg[fy_col].to_numpy()

        if normalize:
            mag = np.sqrt(fx**2 + fy**2)
            fx = fx / (mag + 1e-8)
            fy = fy / (mag + 1e-8)

        ax.quiver(
            x,
            y,
            fx,
            fy,
            angles="xy",
            scale_units="xy",
            scale=arrow_scale,
            color=color,
            alpha=alpha,
            width=0.0025,
            headwidth=3,
            headlength=4,
            headaxislength=3,
        )

    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig, ax

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
    time_col="Unnamed: 0",
):
    _require_row_ix(hmm_df)
    base = _sort_time(hmm_df, time_col=time_col)

    df = base[
        (base[condition_col] == condition)
        & (base[movement_col] == movement_type)
        & (~base[state_col].isna())
        & (~base[value_col].isna())
    ].copy()

    onset_ixs = find_bout_pattern_hits(df, state_col=state_col, pattern=pattern)
    aligned, kept = extract_aligned_windows(base, onset_ixs, value_col=value_col, pre=pre, post=post)

    if aligned.shape[0] == 0:
        raise ValueError("No usable hits found (or all hits near edges for chosen pre/post).")

    x = np.arange(-pre, post + 1)
    baseline_vals = aligned[:, x < 0].ravel()
    mu = baseline_vals.mean()
    sd = baseline_vals.std(ddof=1)
    aligned_z = (aligned - mu) / sd if sd > 0 else (aligned - mu)

    mean = aligned_z.mean(axis=0)
    sem = aligned_z.std(axis=0, ddof=1) / np.sqrt(aligned_z.shape[0])

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax0.plot(x, aligned_z.T, alpha=0.25, linewidth=0.5, marker=".", color="gray")
    ax0.axvline(0, linestyle="--")
    ax0.set_ylabel(value_col)
    ax0.set_title(f"{pattern} hits aligned to state {pattern[0]} onset (n={aligned.shape[0]})")

    ax1.plot(x, mean, linewidth=2)
    ax1.fill_between(x, mean - sem, mean + sem, alpha=0.25)
    ax1.axvline(0, linestyle="--")
    ax1.set_xlabel("Samples relative to onset")
    ax1.set_ylabel(f"Mean {value_col}")

    _style_psth_axes(ax0, ax1)

    if save_name is None:
        save_name = f"psth_hits_{pattern[0]}_{pattern[1]}_{condition}_mov_{movement_type}.svg"

    fig.tight_layout()
    fig.savefig(Path(outpath) / save_name, dpi=300)
    plt.close(fig)
    return aligned, kept

def plot_example_trace(df, out,
                        cmap="mako", 
                       time_col=None,
                        movement_type=-1,
                        decimate=2,
                        rasterize_scatter=False,
                        rasterize_quiver=False,
                        quiver_stride=10,
                        quiver_scale=0.4,
                        scatter_size=10,
                        scatter_alpha=1,
                        quiver_alpha=0.35,
                        trial_col="trial",
                        trials=None,
                        first_n_trials=None,
                    ):
    if movement_type is not None:
        df = df[df["roi_motion"] == movement_type].copy()
    df = _filter_trials(df, trial_col=trial_col, trials=trials, first_n_trials=first_n_trials)

 
    fig, ax = plt.subplots(figsize=(8, 8))

    pts = df.iloc[::decimate] if (decimate and decimate > 1) else df
    t=np.arange(len(pts)) 
    sc =ax.scatter(
        pts["x_pos"],
        pts["y_pos"],
        s=scatter_size,
        c=t,
        cmap=cmap,
        label=state,
        alpha=scatter_alpha,
        linewidths=0,
        rasterized=bool(rasterize_scatter),
    )

    qdf = df.iloc[::quiver_stride] if (quiver_stride and quiver_stride > 1) else df
    qv = ax.quiver(
        qdf["x_pos"],
        qdf["y_pos"],
        qdf["forward_x"],
        qdf["forward_y"],
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
    ax.set_title("Example trace")
    cbar = fig.colorbar(sc, ax=ax, pad=0.01)
    cbar.set_label("Frame")

    suffix = ""
    if first_n_trials is not None:
        suffix = f"_first{int(first_n_trials)}trials"
    elif trials is not None:
        suffix = "_trials" + "-".join(map(str, trials))

    out_file = Path(out) / f"example_trace_{movement_type}_{suffix}.svg"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_stim_onset_aligned_matrix(
    df,
    *,
    stim_col="stim_onset",
    value_col="deltaF_z",
    time_col="Unnamed: 0",
    pre=120,
    post=180,
):
    base = df.copy()
    base[time_col] = pd.to_datetime(base[time_col], errors="coerce")
    base = base.dropna(subset=[time_col]).sort_values(time_col)

    if stim_col not in base.columns:
        raise ValueError(f"{stim_col} not found")
    if value_col not in base.columns:
        raise ValueError(f"{value_col} not found")

    values = pd.to_numeric(base[value_col], errors="coerce").to_numpy()

    # anchor positions: exactly where stim_onset == 1
    event_pos = np.flatnonzero(base[stim_col].to_numpy() == 1)

    win_len = pre + post + 1
    M = []
    meta = []

    for p in event_pos:
        lo, hi = p - pre, p + post
        if lo < 0 or hi >= len(values):
            continue

        w = values[lo:hi+1]
        if w.shape[0] != win_len:
            continue

        M.append(w)
        meta.append({
            "event_pos": int(p),
            "event_time": base.iloc[p][time_col],
        })

    # build time axis (seconds)
    dt = base[time_col].diff().median()
    dt_s = pd.to_timedelta(dt).total_seconds() if pd.notna(dt) else 1.0
    t = np.arange(-pre, post + 1) * dt_s

    if not M:
        return np.empty((0, win_len)), pd.DataFrame(meta), t

    return np.vstack(M), pd.DataFrame(meta), t


def plot_relative_position_by_state_with_quiver(
    df,
    outpath,
    *,
    state_col="states",
    bouts=None,
    row_ix_col="__row_ix__",
    exp_pos=("x_pos", "y_pos"),
    cons_pos=("x_pos_con", "y_pos_con"),
    fx_col="forward_x",
    fy_col="forward_y",
    decimate=2,
    scatter_size=10,
    scatter_alpha=0.9,
    quiver=True,
    quiver_stride=6,
    quiver_scale=0.1,
    quiver_alpha=0.4,
    normalize_quiver=True,
    rasterize_quiver=True,
    rasterize_scatter=False,
    figsize=(6.5, 6.5),
    title=None,
    save_name=None,
):
    """
    Plot experimental mouse position relative to conspecific, colored by HMM state.

    If `bouts` is provided (DataFrame with start_row_ix / end_row_ix),
    only data inside those bouts is plotted (all overlaid on one figure).
    Otherwise the full df is used.
    """
    outpath = Path(outpath)
    outpath.mkdir(parents=True, exist_ok=True)

    # ---- select data ----
    if bouts is not None:
        if row_ix_col not in df.columns:
            raise ValueError(f"df must contain '{row_ix_col}' when using bouts=")

        bouts_df = bouts.copy()
        if "bout_type" not in bouts_df.columns:
            bouts_df["bout_type"] = "bout"

        # build positional lookup once
        d0 = df.sort_values(row_ix_col).copy()
        ix_to_pos = pd.Series(
            np.arange(len(d0)), index=d0[row_ix_col].values
        ).to_dict()

        parts = []
        for r in bouts_df.itertuples(index=False):
            s = int(getattr(r, "start_row_ix"))
            e = int(getattr(r, "end_row_ix"))
            p0 = ix_to_pos.get(s)
            p1 = ix_to_pos.get(e)
            if p0 is None or p1 is None or p1 <= p0:
                continue
            seg = d0.iloc[p0 : p1 + 1]
            if len(seg) >= 2:
                parts.append(seg)

        if not parts:
            print("No valid bout segments found — nothing to plot.")
            return

        print(f"Plotting {len(parts)} / {len(bouts_df)} bouts")
        d = pd.concat(parts, axis=0)
        tag = bouts_df["bout_type"].iloc[0] if len(bouts_df) else "bouts"
    else:
        d = df.copy()
        tag = "all"

    # ---- relative coordinates ----
    d = d.copy()
    d["x_rel"] = d[exp_pos[0]] - d[cons_pos[0]]
    d["y_rel"] = d[exp_pos[1]] - d[cons_pos[1]]

    states = sorted(d[state_col].dropna().unique())
    STATE_COLORS = make_state_colors_from_cmap(states, cmp_discrete)

    # ---- plot ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal", adjustable="box")

    for state in states:
        ds = d[d[state_col] == state]
        if ds.empty:
            continue

        pts = ds.iloc[::decimate] if (decimate and decimate > 1) else ds
        ax.scatter(
            pts["x_rel"],
            pts["y_rel"],
            s=scatter_size,
            color=STATE_COLORS[state],
            alpha=scatter_alpha,
            linewidths=0,
            rasterized=bool(rasterize_scatter),
            label=f"state {state}",
        )

        if quiver and (fx_col in ds.columns) and (fy_col in ds.columns):
            qdf = ds.iloc[::quiver_stride] if (quiver_stride and quiver_stride > 1) else ds

            x = qdf["x_rel"].to_numpy(dtype=float)
            y = qdf["y_rel"].to_numpy(dtype=float)
            u = qdf[fx_col].to_numpy(dtype=float)
            v = qdf[fy_col].to_numpy(dtype=float)

            ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(u) & np.isfinite(v)
            x, y, u, v = x[ok], y[ok], u[ok], v[ok]

            if normalize_quiver and len(x):
                mag = np.hypot(u, v)
                keep = mag > 1e-8
                x, y, u, v = x[keep], y[keep], u[keep] / mag[keep], v[keep] / mag[keep]

            if len(x):
                qv = ax.quiver(
                    x, y, u, v,
                    angles="xy",
                    scale=quiver_scale,
                    scale_units="xy",
                    headwidth=12,
                    headlength=14,
                    headaxislength=12,
                    alpha=quiver_alpha,
                    color="k",
                    linewidth=0.5,
                )
                if rasterize_quiver:
                    qv.set_rasterized(True)

    # conspecific at origin
    ax.scatter([0], [0], s=70, color="k", alpha=0.8, linewidths=0, label="conspecific")

    ax.grid(False)
    ax.set_axis_off()
    ax.set_title(title or f"Exp relative to conspecific ({tag})")
    ax.legend(frameon=False, markerscale=1.6, handlelength=0.8, handletextpad=0.4)

    out_file = outpath / (save_name or f"relative_pos_by_state_{tag}_quiver.svg")
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def extract_state_events_within_motion_bouts(
    df,
    motion_bouts,
    *,
    row_ix_col="__row_ix__",
    trial_col="trial",
    state_col="states",
    motion_start_col="start_row_ix",
    motion_end_col="end_row_ix",
    motion_type_col="bout_type",          # "approach"/"retreat"
    condition_col="olfactory_stim",
    condition=None,                       # e.g. "Appetitive" or None
    min_state_len_frames=1,               # require at least this many frames in the state run
):
    """
    Returns events with columns:
      - bout_type
      - trial
      - state
      - anchor_row_ix    (state onset inside the motion bout)
      - state_end_row_ix (state offset inside the motion bout)
      - state_len_frames
      - motion_start_row_ix, motion_end_row_ix
    """
    if row_ix_col not in df.columns:
        raise ValueError(f"df missing {row_ix_col}")
    if state_col not in df.columns:
        raise ValueError(f"df missing {state_col}")
    for c in (motion_start_col, motion_end_col, motion_type_col):
        if c not in motion_bouts.columns:
            raise ValueError(f"motion_bouts missing {c}")

    base = df.copy()
    if condition is not None:
        if condition_col not in base.columns:
            raise ValueError(f"df missing {condition_col}")
        base = base[base[condition_col] == condition].copy()

    # Sort by row_ix so slicing is well-defined
    base = base.sort_values(row_ix_col).copy()
    base_ix = base.set_index(row_ix_col, drop=False)

    events = []

    for r in motion_bouts.itertuples(index=False):
        ms = int(getattr(r, motion_start_col))
        me = int(getattr(r, motion_end_col))
        btype = getattr(r, motion_type_col)

        seg = base_ix.loc[ms:me].copy()
        if seg.empty:
            continue

        # Optional: enforce single trial
        trial_id = seg[trial_col].iloc[0] if trial_col in seg.columns else None

        # run-length encode state within this motion segment
        s = seg[state_col].to_numpy()
        row_ix = seg[row_ix_col].to_numpy()

        # treat NaNs as "no state"
        valid = pd.notna(s)
        if not valid.any():
            continue

        # run boundaries (only where state changes)
        # We'll do it on the raw array, but skip runs where state is NaN
        change = np.r_[True, s[1:] != s[:-1]]
        run_id = np.cumsum(change)

        seg2 = seg.assign(_run=run_id)

        for _, g in seg2.groupby("_run", sort=False):
            st = g[state_col].iloc[0]
            if pd.isna(st):
                continue
            L = len(g)
            if L < min_state_len_frames:
                continue

            events.append({
                "bout_type": btype,
                trial_col: trial_id,
                "state": int(st),
                "anchor_row_ix": int(g[row_ix_col].iloc[0]),        # state onset inside motion bout
                "state_end_row_ix": int(g[row_ix_col].iloc[-1]),    # state offset inside motion bout
                "state_len_frames": int(L),
                "motion_start_row_ix": ms,
                "motion_end_row_ix": me,
            })

    return pd.DataFrame(events)

def make_event_aligned_matrix(
    df,
    events_df,
    *,
    value_col="deltaF_z",
    anchor_col="anchor_row_ix",
    pre=120,
    post=180,
    time_col="Unnamed: 0",
    row_ix_col="__row_ix__",
    trial_col="trial",
    enforce_trial=None,     # "mask" / "drop" / "none"
    min_finite_frac=0.8,
):
    base = df.copy()
    base[time_col] = pd.to_datetime(base[time_col], errors="coerce")
    base = base.dropna(subset=[time_col]).sort_values(time_col)

    if row_ix_col not in base.columns:
        raise ValueError(f"df missing {row_ix_col}")
    if anchor_col not in events_df.columns:
        raise KeyError(f"events_df missing {anchor_col}")

    ix_to_pos = pd.Series(np.arange(len(base)), index=base[row_ix_col]).to_dict()
    values = pd.to_numeric(base[value_col], errors="coerce").to_numpy(dtype=float)
    trials = base[trial_col].to_numpy() if trial_col in base.columns else None

    win_len = pre + post + 1
    M, meta = [], []

    for _, r in events_df.iterrows():
        anchor_ix = int(r[anchor_col])
        pos = ix_to_pos.get(anchor_ix, None)
        if pos is None:
            continue

        lo, hi = pos - pre, pos + post
        if lo < 0 or hi >= len(values):
            continue

        w = values[lo:hi+1].copy()
        if w.shape[0] != win_len:
            continue

        # trial handling (optional)
        if enforce_trial != "none" and (trials is not None) and (trial_col in events_df.columns):
            trial_id = r[trial_col]
            if enforce_trial == "drop":
                if not np.all(trials[lo:hi+1] == trial_id):
                    continue
            elif enforce_trial == "mask":
                tmask = (trials[lo:hi+1] == trial_id)
                w[~tmask] = np.nan
                if np.mean(np.isfinite(w)) < float(min_finite_frac):
                    continue

        M.append(w)
        meta.append(r.to_dict())

    dt = base[time_col].diff().median()
    dt_s = pd.to_timedelta(dt).total_seconds() if pd.notna(dt) else 1.0
    t = np.arange(-pre, post + 1) * dt_s

    if not M:
        return np.empty((0, win_len)), pd.DataFrame(meta), t

    return np.vstack(M), pd.DataFrame(meta), t


# ----------------------------
# Script entrypoint
# ----------------------------
def main():
    for mouse in [1106077]:
        file_path = Path(rf"F:\social_sniffing\derivatives\{mouse}\social\Methimazole\model_predictions\gaussian_raw\all_session_data_5states.csv")
        outpath = Path(rf"F:\social_sniffing\derivatives\{mouse}\social\Methimazole\plots_new")
        outpath.mkdir(parents=True, exist_ok=True)

        df = pd.read_csv(file_path)
        df = _ensure_row_ix(df)

        df_full = _sort_time(df.copy(), time_col="Unnamed: 0")

        df2 = classify_approach_retreat_with_ddist(
            df_full,
            exp_pos=("x_pos", "y_pos"),
            cons_pos=("x_pos_con", "y_pos_con"),
            disp_cols=("forward_x", "forward_y"),
            out_col="cons_motion",
            align_thresh=0.75, 
            ddist_thresh=80
        )
        approach_bouts = extract_approach_bouts(df2, min_peak_close=20, max_gap_frames=10)
        retreat_bouts  = extract_retreat_bouts(df2,  min_peak_open=10,  max_gap_frames=10)

        motion_bouts = pd.concat([approach_bouts, retreat_bouts], ignore_index=True)\
                         .sort_values("start_row_ix")\
                         .reset_index(drop=True)
        print(motion_bouts["bout_type"].value_counts())

        plot_relative_position_by_state_with_quiver(
            df2, outpath / "approach_bouts",
            bouts=approach_bouts,
                     # optional cap
        )

        plot_relative_position_by_state_with_quiver(
            df2, outpath / "retreat_bouts",
            bouts=retreat_bouts,
            
    )
        #plot_relative_position_by_state_with_quiver(df2, outpath, movement_value=-1)  # retreat

        events = extract_state_events_within_motion_bouts(
        df2,
        motion_bouts,
        row_ix_col="__row_ix__",
        trial_col="trial",
        state_col="states",
        condition=None,          # or None
        min_state_len_frames=3,          # optional
    )

    for btype in ["approach", "retreat"]:
        for state in sorted(events["state"].unique()):
            ev = events[(events["bout_type"] == btype) & (events["state"] == state)].copy()
            if ev.empty:
                continue

            M, meta, t = make_event_aligned_matrix(
                df2, ev,
                value_col="sniff_freq",
                anchor_col="anchor_row_ix",   # state onset inside motion bout
                pre=60, post=180,
                enforce_trial=None,
                min_finite_frac=0.8,
            )

            if M.shape[0] == 0:
                continue

            fig, _ = plot_state_heatmap_with_endlines_and_mean(
                M, meta, t,
                bout_len_col="state_len_frames",   # endline = end of STATE run, not motion bout
                title=f"State {state} within {btype} bouts (aligned to state onset)",
            )
            fig.savefig(outpath / f"state_{state}_within_{btype}_bouts_aligned_state_onset_sniff_freq.svg",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
   
# def main():
#     for mouse in [1106077]:#, 1106078,1106079]:

#         file_path = Path(
#         rf"F:\social_sniffing\derivatives\{mouse}\social\CTRL\model_predictions\gaussian_raw\all_session_data_5states.csv"
#     )
#         outpath = Path(rf"F:\social_sniffing\derivatives\{mouse}\social\CTRL\plots_new")
#         outpath.mkdir(parents=True, exist_ok=True)

#         df = pd.read_csv(file_path)
#         df = _ensure_row_ix(df)

#         port1 = ((400, 500), (400, 600), (500, 600), (500, 500))
#         port078 = ((200, 450), (200, 550), (300, 550), (300, 450))
#         port10 = ((150, 300), (150, 400), (250, 400), (250, 400))
#         port563 = ((400, 450), (400, 550), (500, 550), (400, 550))

#         if mouse == 1106078:
#             df = classify_motion_relative_to_roi(df, port078)
#         elif mouse == 1106010:
#             df = classify_motion_relative_to_roi(df, port10)
#         elif mouse == 1125563:
#             df = classify_motion_relative_to_roi(df, port563)
#         else:
#             df = classify_motion_relative_to_roi(df, port1)

#         df_full = df.copy()  # time-contiguous base
#         df_full = _sort_time(df_full, time_col="Unnamed: 0")  # ensure consistent ordering

#         # Optional: keep only one condition everywhere
#         # df_full = df_full[df_full["olfactory_stim"] == "Appetitive"].copy()
#         M, meta, t = make_stim_onset_aligned_matrix(
#         df_full,
#         value_col="deltaF_z",
#         stim_col="stim_onset",
#         pre=120,
#         post=180,
#     )

#         fig, _ = plot_aligned_heatmap_and_mean(
#                                                 M,
#                                                 t,
#                                                 title="deltaF_z aligned to stim onset",
#                                                 cmap="plasma",
#                                             )
#         fig.savefig(outpath / "deltaF_z_aligned_to_stim_onset.svg",
#                     dpi=300, bbox_inches="tight")
#         plt.close(fig)

#         for state in [0, 1, 2, 3, 4]:
#             # # detect bouts in subset
#             # if state in (1, 4):
#             #     df_sub = df_full[df_full["roi_motion"] == -1].copy()
#             # else:
#             #     df_sub = df_full.copy()
#             #df_sub = df_full[df_full["roi_motion"] == 1].copy()
#             df_sub = df_full.copy()
#             bouts = find_state_bouts(df_sub, state_col="states", target_state=state, trial_col="trial")

#             # NEW: one bout per trial (the longest)
#             # bouts = keep_longest_bout_per_trial(bouts, trial_col="trial", len_col="bout_len")

#             M, meta, t = make_bout_aligned_matrix(df_full, bouts, value_col="deltaF_z",
#                                       align="onset", pre=120, post=180, sort_by=None)
#             print(f"Mouse {mouse}, State {state}: Found {len(bouts)} bouts")
#             if len(bouts) == 0:
#                 print(f"Mouse {mouse}, State {state}: No bouts found, skipping.")
#                 continue

#                        # --- state-sequence heatmap aligned to the same anchors ---
#             # S, meta_s, t_s = make_bout_aligned_state_matrix(
#             #     df_sub,
#             #     bouts,
#             #     state_col="states",
#             #     align="offset",     # or "offset"
#             #     pre=180,
#             #     post=180,
                
#             # )

#             # order = np.argsort(-meta_s["bout_len"].to_numpy())
#             # S_sorted = S[order]
#             # meta_sorted = meta_s.iloc[order].reset_index(drop=True)

#             # fig, _ = plot_state_raster_with_occupancy(S_sorted, t_s, title="...", state_order=[0,1,2,3,4])
#             #plt.show()
#             # fig.savefig(outpath / f"state_{state}_aligned_state_sequence_roi_1.svg", dpi=300, bbox_inches="tight")
#             # plt.close(fig)

#             fig, _ = plot_state_heatmap_with_endlines_and_mean(
#                 M, meta, t, title=f"State {state} aligned (deltaF_z)"
#             )
#             fig.savefig(outpath / f"state_{state}_aligned_heatmap_deltaF.svg", dpi=300, bbox_inches="tight")
#             plt.close(fig)

#             fig, _ = plot_bout_xy_trajectories(df_sub, meta, title=f"State {state} bout trajectories")
#             fig.savefig(outpath / f"state_{state}_bout_trajectories.svg", dpi=300, bbox_inches="tight")
#             plt.close(fig)

# def main():
#     mice = [1106077, 1106078, 1106079]  # <-- set your mice here

#     out_root = Path(r"F:\social_sniffing\derivatives\GROUP\olfactory_ctrls\CTRL\plots_new")
#     out_root.mkdir(parents=True, exist_ok=True)

#     traces_by_mouse = {}   # mouse -> {state: 1D mean trace}
#     t_ref = None

#     port1 = ((400, 500), (400, 600), (500, 600), (500, 500))
#     port078 = ((200, 450), (200, 550), (300, 550), (300, 450))
#     port10 = ((150, 300), (150, 400), (250, 400), (250, 400))

#     for mouse in mice:
#         file_path = Path(
#             rf"F:\social_sniffing\derivatives\{mouse}\olfactory_ctrls\CTRL\model_predictions\gaussian_raw\all_session_data_5states.csv"
#         )

#         df = pd.read_csv(file_path)
#         df = _ensure_row_ix(df)

#         # classify motion (ROI differs for a couple mice)
#         if mouse == 1106078:
#             df = classify_motion_relative_to_roi(df, port078)
#         elif mouse == 1106010:
#             df = classify_motion_relative_to_roi(df, port10)
#         else:
#             df = classify_motion_relative_to_roi(df, port1)

#         # time-contiguous base
#         df_full = _sort_time(df.copy(), time_col="Unnamed: 0")

#         # optional: enforce one condition consistently across mice
#         # df_full = df_full[df_full["olfactory_stim"] == "Appetitive"].copy()

#         per_state_traces = {}

#         for state in [0, 1, 2, 3, 4]:
#             # detect bouts on subset (but ALWAYS extract from df_full)
#             if state in (1, 4):
#                 df_sub = df_full[df_full["roi_motion"] == -1].copy()
#             else:
#                 df_sub = df_full.copy()

#             bouts = find_state_bouts(
#                 df_sub, state_col="states", target_state=state, trial_col="trial"
#             )
#             if len(bouts) == 0:
#                 continue

#             M, meta, t = make_bout_aligned_matrix(
#                 df_full,
#                 bouts,
#                 value_col="deltaF_z",
#                 align="onset",   # change to "offset" if desired
#                 pre=120,
#                 post=180,
#             )
#             if M.shape[0] == 0:
#                 continue

#             # baseline-correct each bout (recommended)
#             bmask = (t >= -1.0) & (t < 0.0)
#             M0 = M - np.nanmean(M[:, bmask], axis=1, keepdims=True)

#             # per-mouse mean trace for this state
#             per_state_traces[state] = np.nanmean(M0, axis=0)

#             # store reference time axis
#             if t_ref is None:
#                 t_ref = t
#             else:
#                 if len(t_ref) != len(t) or not np.allclose(t_ref, t):
#                     raise ValueError("Time axis differs across mice. Check pre/post and dt computation.")

#         traces_by_mouse[mouse] = per_state_traces
#         print(f"Mouse {mouse}: collected states {sorted(per_state_traces.keys())}")

#     # ---- group plot: one subplot per state (0,1,4), mean ± SEM across mice ----
#     fig, _ = plot_group_mean_sem_per_state_subplots(
#         traces_by_mouse,
#         t_ref,
#         states=(0, 1, 4),
#         ncols=3,
#         title="Group mean ± SEM deltaF_z aligned to state onset",
#         ylabel="deltaF_z",
#     )
#     #plt.show()
#     fig.savefig(out_root / "group_mean_sem_deltaF_z_states_0_1_4_retreat.svg", dpi=300, bbox_inches="tight")
#     plt.close(fig)




# bouts = find_state_bouts(df, state_col="states", target_state=4, trial_col="trial")



# M, meta, t = make_bout_aligned_matrix(df, bouts, value_col="deltaF_z",
#                                     align="onset", pre=120, post=180)

# fig, (ax_hm, ax_mu) = plot_state_heatmap_with_endlines_and_mean(
# M,
# meta,
# t,
# title="State 4 aligned (deltaF_z)",
# )
# plt.show()

# fig, ax = plot_bout_xy_trajectories(df, meta, title="State 4 bout trajectories")
# plt.show()
# order, score = sort_bouts(
#     M, t,
#     baseline=(-1.0, 0.0),
#     response=(0.0, 1.0),
#     sort_mode="mean_signed",   # good for dips
#     direction="auto"
# )

# M_sorted = M[order]
# meta_sorted = meta.iloc[order].reset_index(drop=True)
# meta_sorted["sort_score"] = score[order]

# # plot ONLY true retreats
# plot_position_by_state(
#     df_ann[df_ann["true_retreat"]].copy(),
#     outpath=outpath,
#     movement_type=-1,
#     condition="Aversive",
#     decimate=5,
#     first_n_trials=None,
# )
# plot_deltaF_aligned_to_state_onsets(
#     df_ann,
#     outpath=outpath,
#     stim="Aversive",
#     movement_type=-1,
#     first_n_trials=8,
#     require_col="true_retreat",
#     require_value=True,
# )

# plot_psth_state_sequence_hits(
#     hmm_df=df,
#     outpath=outpath,
#     pattern=(0, 1),
#     pre=50,
#     post=100,
#     movement_type=-1,
# )
# plot_psth_state_sequence_hits(
#     hmm_df=df,
#     outpath=outpath,
#     pattern=(4, 1),
#     pre=50,
#     post=100,
#     movement_type=1,
# )

# # Circos transitions (optional)
# # for movement_type in [-1, 1]:
# #     P, counts = empirical_transition_matrix_boutwise_by_trial(
# #         df[df["roi_motion"] == movement_type], "states", "trial", K=5
# #     )
# #     fig, ax = plot_transition_circos(P, min_prob=0.001, max_edges_per_node=10, arrow=True)
# #     fig.savefig(outpath / f"transition_circos_{movement_type}.svg", dpi=300)
# #     plt.close(fig)


if __name__ == "__main__":
    main()