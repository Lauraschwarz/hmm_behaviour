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
from scipy.integrate import cumulative_trapezoid
import numpy as np
import ultraplot as uplt
#from plotting import plot_deltaF

cmp = sns.diverging_palette(220, 20, as_cmap=True)
cmp2 = sns.color_palette("tab20", as_cmap=True) 

MICE = ['1106077', '1106009', '1106010', '1106008', '1106078', '1106079', '1125563', '1125131', '1125132']
VGAT_MICE = ['1106077', '1106078', '1106079', '1125131', '1125132', '1125563']
METHIMAZOLE_SESSIONS = {'1106077': '2025-10-02T14-11-03',
                        '1106009': '2025-10-02T13-33-33',
                        '1106010': '2025-09-26T12-42-23',
                        '1106078': '2025-09-26T13-21-09',
                        '1106079': '2025-10-02T14-45-26',
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
                    '1125132': 'OT'
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


file_path = pathlib.Path(r'F:\social_sniffing\derivatives\all_mice_df_with_genotypes_olfactory_ctrls.csv')

df = pd.read_csv(file_path)


def panel_D(df,  mouse_id=1106079, column='sampling', signal_col='deltaF_z', condition='Control'):
    df_g = df[(df["mouse_id"] == mouse_id)]
    df_c = df_g[df_g["condition"] == condition]
    array = [[1,1,1],[2,2,2],[3,3,3]]

    for session in df_c["session_id"].unique():
        df1 = df_c[df_c["session_id"] == session]
        stim_times = df1.index[df1[column] == 1]
        first3 = stim_times[3:6]
        last3 = stim_times[-3:]
        fig, ax = uplt.subplots(array, figsize=(6, 5), share=False, includepanels=True)

        for i, t in enumerate(first3):
            start, end = t - 50, t + 100
            seg = df_g.loc[start:end, signal_col]
            b = min(25, len(seg))
            base = seg.iloc[:b].mean() if b > 0 else 0.0
            y = seg.values - base
            sns.lineplot(x=seg.index - t, y=y, ax=ax[i], color='steelblue', alpha=0.5, legend=False)
            ax[i].axvline(0, color='k', ls='--', lw=0.8)
            ax[i].set_title(f"Mouse {mouse_id} — Sampling event trial {i+1}")
            ax[i].set_ylabel("dF/F z-score (baseline-subtracted)")
            ax[i].set_ylim(-3,10)

        for i, t in enumerate(last3):
            start, end = t - 50, t + 100
            seg = df_g.loc[start:end, signal_col]
            b = min(25, len(seg))
            base = seg.iloc[:b].mean() if b > 0 else 0.0
            y = seg.values - base
            sns.lineplot(x=seg.index - t, y=y, ax=ax[i], color='indianred', alpha=0.5, legend=False)
            ax[i].axvline(0, color='k', ls='--', lw=0.8)

        fig.savefig(rf'F:\upgrade_figures\panel_D\panel_D_mouse_{mouse_id}_all_{condition}_{session}_{column}.svg')


def panel_EF(df, column='sampling', mouse_id=1106079, signal_col='deltaF_z', condition='Control'):

    df_g = df[(df["mouse_id"] == mouse_id)]
    df_g = df_g[df_g["condition"] == condition]
    
    
    array = [  
        [1, 1, 1]]
     
    for session in df_g["session_id"].unique():
        df1 = df_g[df_g["session_id"] == session]
    
        stim_times = df1.index[df1[column] == 1]
        first3 = stim_times[:3]
        last3 = stim_times[-3:]
        fig, ax = uplt.subplots(array, figsize=(4, 5), share=False, includepanels=True)
        cum_first = []
        time_rel = None

        for i, t in enumerate(first3):
            start, end = t - 50, t + 100
            seg = df1.loc[start:end, signal_col]
            cum = cumulative_trapezoid(seg.values - seg.iloc[:25].mean(), dx=1/25, initial=0)
            ax.plot(seg.index - t, cum, color='steelblue', alpha=0.5)
            ax.axvline(0, color='k', ls='--', lw=0.8)
        
            ax.set_ylim(-8, 8)
            cum_first.append(cum)
            if time_rel is None:
                time_rel = seg.index - t
        if cum_first:
            mean_first = np.nanmean(np.vstack(cum_first), axis=0)
            for i in range(len(first3)):
                ax.plot(time_rel, mean_first, color='steelblue', lw=2, label='mean')
        # fig.savefig(rf'F:\upgrade_figures\panel_E\panel_E_first3_{mouse_id}.svg')
        # plt.close(fig)

        # fig, ax = uplt.subplots(array, figsize=(4, 5), share=False, includepanels=True)
        cum_last = []
        time_rel = None
        for i, t in enumerate(last3):
            start, end = t - 50, t + 100
            seg = df1.loc[start:end, signal_col]
            cum = cumulative_trapezoid(seg.values - seg.iloc[:25].mean() , dx=1/25, initial=0)
            ax.plot(seg.index - t, cum, color='indianred', alpha=0.5)
            ax.axvline(0, color='k', ls='--', lw=0.8)
            ax.set_title(f"{mouse_id} — comparison of sampling events")
            ax.set_ylabel("integral of dF/F ")
            ax.set_xlabel("Time (frames relative to sampling event)")
            ax.set_ylim(-8, 8)
            cum_last.append(cum)
            if time_rel is None:
                time_rel = seg.index - t
        if cum_last: 
            mean_last = np.nanmean(np.vstack(cum_last), axis=0)
            for i in range(len(last3)):
                ax.plot(time_rel, mean_last, color='indianred', lw=2, label='mean')
        fig.savefig(rf'F:\upgrade_figures\panel_E\panel_E_summary_integral_{mouse_id}_{session}_{column}_{condition}.svg')
    #plt.close(fig)

def panel_E_genotype(df, column='sampling', signal_col='deltaF_z', genotype='VGAT', condition='Aversive'):
    df_g = df[(df["genotype"] == genotype)]
    df_g = df_g[df_g["session_id"] == 'session_1']

    cum_first_all, cum_last_all = [], []
    time_rel1 = time_rel2 = None
    records = []
    # per-session accumulators
    session_first = {}
    session_last = {}
    mouse_means = []
    fig, ax = uplt.subplots(ncols=3, wratios=(3, 1, 1), ref=1, refwidth=4, share=False)
    ax.format(abc=True)


    for mouse_id in df_g["mouse_id"].unique():
        mouse_df = df_g[df_g["mouse_id"] == mouse_id]
        for session in mouse_df["session_id"].unique():
            session_df = mouse_df[mouse_df["session_id"] == session].reset_index(drop=True)
            onsets = np.flatnonzero(session_df[column].values == 1)
            first3_idx = onsets[:3]
            last3_idx  = onsets[-3:]

            first_cums, last_cums = [], []

            for onset in first3_idx:
                start, end = onset - 50, onset + 100
                if start < 0 or end >= len(session_df): continue
                seg = session_df.iloc[start:end+1][signal_col]
                cum = cumulative_trapezoid(seg.values - seg.iloc[:25].mean(), dx=1/25, initial=0)
                first_cums.append(cum); cum_first_all.append(cum)
                session_first.setdefault(session, []).append(cum)
                if time_rel1 is None: 
                    time_rel1 = np.arange(start, end+1) - onset

            for onset in last3_idx:
                start, end = onset - 50, onset + 100
                if start < 0 or end >= len(session_df): continue
                seg = session_df.iloc[start:end+1][signal_col]
                cum = cumulative_trapezoid(seg.values - seg.iloc[:25].mean(), dx=1/25, initial=0)
                last_cums.append(cum); cum_last_all.append(cum)
                session_last.setdefault(session, []).append(cum)
                if time_rel2 is None: 
                    time_rel2 = np.arange(start, end+1) - onset

            if first_cums and last_cums:
                records.append({
                    "mouse_id": mouse_id, "group": "First",
                    "max_integral": float(max(np.nanmax(c) for c in first_cums)),
                    "mean_integral": float(np.nanmedian([c for c in first_cums]))
                    
                })
                records.append({
                    "mouse_id": mouse_id, "group": "Last",
                    "max_integral": float(max(np.nanmax(c) for c in last_cums)),
                    "mean_integral": float(np.nanmedian([c for c in last_cums]))
                })
            sns.lineplot(x=time_rel1, y=np.nanmean(np.vstack(first_cums), axis=0), color='steelblue', alpha=0.3, ax=ax[0], legend=False)
            sns.lineplot(x=time_rel2, y=np.nanmean(np.vstack(last_cums), axis=0), color='indianred', alpha=0.3, ax=ax[0], legend=False)
    

   
    # overall means
    if cum_first_all:
        mean_first = np.nanmean(np.vstack(cum_first_all), axis=0)
        ax[0].plot(time_rel1, mean_first, color='steelblue', lw=2.5, label="First mean")
    if cum_last_all:
        mean_last = np.nanmean(np.vstack(cum_last_all), axis=0)
        ax[0].plot(time_rel2, mean_last, color='indianred', lw=2.5, label="Last mean")

    # per-session means (thin overlays)
    blue_sess = sns.color_palette("blend:#7AB,#EDA", 6)
    blue_mean = sns.color_palette("Blues", 9)[-2]
    red_mean  = sns.color_palette("Reds", 9)[-2]
    red_sess  = sns.color_palette("blend:#cd5c5c,#8B0000", 6)
    # for si, (sess, cums) in enumerate(session_first.items()):
    #     m = np.nanmean(np.vstack(cums), axis=0)
    #     ax[0].plot(time_rel1, m, color=blue_sess[min(si, len(blue_sess)-1)], lw=1.5, alpha=0.5, label=f"{sess} First")
    # for si, (sess, cums) in enumerate(session_last.items()):
    #     m = np.nanmean(np.vstack(cums), axis=0)
    #     ax[0].plot(time_rel2, m, color=red_sess[min(si, len(red_sess)-1)], lw=1.5, alpha=0.5, label=f"{sess} Last")
    
     # per-mouse mean overlays
    
    ax[0].axvline(0, color='k', ls='--', lw=0.8)
    ax[0].set_title(f"{genotype}: First vs Last sampling events")
    ax[0].set_ylabel("integral of dF/F")
    ax[0].set_xlabel("Frames (relative to sampling)")
    ax[0].set_ylim(-5, 5)
    ax[0].grid(False)
    #ax[0].legend(fontsize='x-small', ncol=2, frameon=False)

    if records:
        df_pts = pd.DataFrame(records)
        pal = {'First': 'steelblue', 'Last': 'indianred'}
      
        sns.scatterplot(data=df_pts, x="group", y="max_integral", hue="group", palette=pal,
                        s=30, alpha=0.5, ax=ax[1], legend=False)
        sns.pointplot(data=df_pts, x="group", y="max_integral",
                      estimator=np.median, errorbar=("ci",95), join=True, palette=pal,
                      ax=ax[1], markers="o", lw=0.5, markersize=7)
        ax[1].set_title(f"{genotype}: First vs Last")
        ax[1].set_ylabel("Max integral of dF/F")
        ax[1].set_ylim(-1, 25)
        ax[1].grid(False)
        ax[1].tick_params(axis='x', rotation=45)

        sns.scatterplot(data=df_pts, x="group", y="mean_integral", hue="group", palette=pal,
                        s=30, alpha=0.5, ax=ax[2], legend=False)
        sns.pointplot(data=df_pts, x="group", y="mean_integral",
                      estimator=np.median, errorbar=("ci",95), join=True, palette=pal,
                      ax=ax[2], markers="o", lw=0.5, markersize=7)
        ax[2].set_title(f"{genotype}: First vs Last")
        ax[2].set_ylabel("Mean integral of dF/F")
        ax[2].set_ylim(-1, 1.5)
        ax[2].tick_params(axis='x', rotation=45)
        ax[2].grid(False)

    fig.savefig(rf'F:\upgrade_figures\panel_E\panel_E_summary_integral_session_1_{genotype}_{column}_{condition}.svg')
    plt.close(fig)

def Methimazole_comp(df, column='sampling', signal_col='deltaF_z', genotype='VGAT'):
    df_g = df[df["genotype"] == genotype]

    ctrl_label = 'session_1'
    meth_label = 'Methimazole'

    cum_ctrl_all, cum_meth_all = [], []
    time_ctrl = time_meth = None
    ctrl_session = {}
    meth_session = {}
    # per-mouse mean curves
    ctrl_mouse_means = {}
    meth_mouse_means = {}

    records = []

    fig, ax = uplt.subplots(ncols=3, wratios=(3, 1, 1), share=False)

    for mouse_id in df_g["mouse_id"].unique():
        mdf = df_g[df_g["mouse_id"] == mouse_id]

        mdf_ctrl = mdf[mdf["session_id"] == ctrl_label].reset_index(drop=True)
        mdf_meth = mdf[mdf["condition"] == meth_label].reset_index(drop=True)
        
        pos_ctrl = np.flatnonzero(mdf_ctrl[column].values == 1)
        pos_meth = np.flatnonzero(mdf_meth[column].values == 1)
  

        first3_ctrl = pos_ctrl[:3]
        first3_meth = pos_meth[:3]

        ctrl_cums, meth_cums = [], []

        for onset in first3_ctrl:
            start, end = onset - 50, onset + 100
     
            seg = mdf_ctrl.iloc[start:end+1][signal_col]
            base = seg.iloc[:50].mean()
            cum = cumulative_trapezoid(seg.values - base, dx=1/25, initial=0)
            ctrl_cums.append(cum); cum_ctrl_all.append(cum)
            ctrl_session.setdefault(ctrl_label, []).append(cum)
            if time_ctrl is None:
                time_ctrl = np.arange(start, end+1) - onset

        for onset in first3_meth:
            start, end = onset - 50, onset + 100
            seg = mdf_meth.iloc[start:end+1][signal_col]
            base = seg.iloc[:50].mean()
            cum = cumulative_trapezoid(seg.values - base, dx=1/25, initial=0)
            meth_cums.append(cum); cum_meth_all.append(cum)
            meth_session.setdefault(meth_label, []).append(cum)
            if time_meth is None:
                time_meth = np.arange(start, end+1) - onset

        if ctrl_cums:
            ctrl_mouse_means[mouse_id] = np.nanmean(np.vstack(ctrl_cums), axis=0)
        if meth_cums:
            meth_mouse_means[mouse_id] = np.nanmean(np.vstack(meth_cums), axis=0)

        if ctrl_cums and meth_cums:
            records.append({"mouse_id": mouse_id, "group": "Control",
                            "max_integral": float(max(np.nanmax(c) for c in ctrl_cums)), 
                            "mean_integral": float(np.nanmedian([c for c in ctrl_cums]))})
            records.append({"mouse_id": mouse_id, "group": "Methimazole",
                            "max_integral": float(max(np.nanmax(c) for c in meth_cums)),
                            "mean_integral": float(np.nanmedian([c for c in meth_cums]))})

    # Panel 1: means (overall + per-session + per-mouse overlays)
    blue_mean = sns.color_palette("Blues", 9)[-2]
    red_mean  = sns.color_palette("Reds", 9)[-2]
    blue_sess = sns.color_palette("blend:#7AB,#EDA", 6)
    red_sess  = sns.color_palette("blend:#cd5c5c,#8B0000", 6)

    if cum_ctrl_all:
        mean_ctrl = np.nanmean(np.vstack(cum_ctrl_all), axis=0)
        ax[0].plot(time_ctrl, mean_ctrl, color=blue_mean, lw=2.0, label="Control mean")
    if cum_meth_all:
        mean_meth = np.nanmean(np.vstack(cum_meth_all), axis=0)
        ax[0].plot(time_meth, mean_meth, color=red_mean, lw=2.0, label="Meth mean")

    for si, (sess, cums) in enumerate(ctrl_session.items()):
        m = np.nanmean(np.vstack(cums), axis=0)
        ax[0].plot(time_ctrl, m, color=blue_sess[min(si, len(blue_sess)-1)], lw=1.0, alpha=0.6, label=f"{sess}")
    for si, (sess, cums) in enumerate(meth_session.items()):
        m = np.nanmean(np.vstack(cums), axis=0)
        ax[0].plot(time_meth, m, color=red_sess[min(si, len(red_sess)-1)], lw=1.0, alpha=0.6, label=f"{sess}")

    # per-mouse mean overlays
    for mid, mcurve in ctrl_mouse_means.items():
        ax[0].plot(time_ctrl, mcurve, color=blue_mean, lw=0.8, alpha=0.4)
    for mid, mcurve in meth_mouse_means.items():
        ax[0].plot(time_meth, mcurve, color=red_mean, lw=0.8, alpha=0.4)

    ax[0].axvline(0, color='k', ls='--', lw=0.8)
    ax[0].set_title(f"{genotype}: Control vs Meth (first 3 sampling)")
    ax[0].set_ylabel("Cumulative integral")
    ax[0].set_xlabel("Frames (rel. to sampling)")
    ax[0].set_ylim(-5, 5)
    ax[0].legend(fontsize='x-small', ncol=2, frameon=False)

    if records:
        df_pts = pd.DataFrame(records)
        pal = {'Control': 'steelblue', 'Methimazole': 'indianred'}
        sns.scatterplot(data=df_pts, x="group", y="max_integral", hue="group", palette=pal,
                        s=30, alpha=0.5, ax=ax[2], legend=False)
        sns.pointplot(data=df_pts, x="group", y="max_integral",
                      estimator=np.median, errorbar=("ci",95), join=True, palette=pal,
                      ax=ax[2], markers="o", lw=0.5, markersize=7)
        ax[2].set_title(f"{genotype}: Control vs Methimazole")
        ax[2].set_ylabel("Max integral of dF/F")
        ax[2].set_ylim(-1, 25)
        ax[2].tick_params(axis='x', rotation=45)

        sns.scatterplot(data=df_pts, x="group", y="mean_integral", hue="group", palette=pal,
                        s=30, alpha=0.5, ax=ax[1], legend=False)
        sns.pointplot(data=df_pts, x="group", y="mean_integral",
                      estimator=np.median, errorbar=("ci",95), join=True, palette=pal,
                      ax=ax[1], markers="o", lw=0.5, markersize=7)
        ax[1].set_title(f"{genotype}: Control vs Methimazole")
        ax[1].set_ylabel("Mean integral of dF/F")
        ax[1].set_ylim(-1, 1.5)
        ax[1].tick_params(axis='x', rotation=45)

    fig.savefig(rf'F:\upgrade_figures\panel_A\panel_A_{genotype}_Meth_comparison_{column}.svg')
    plt.close(fig)

def comp_session(df, genotype, column='sampling', signal_col='deltaF_z'):
    df_g = df[df["genotype"] == genotype]

    label_1 = 'session_1'
    label_2 = 'session_2'

    ses_1_all, ses_2_all = [], []
    time_ses_1 = time_ses_2 = None
    session1 = {}
    session_2 = {}
    # per-mouse mean curves
    ses1_means = {}
    ses2_means = {}

    records = []

    fig, ax = uplt.subplots(ncols=3, wratios=(3, 1, 1), share=False)

    for mouse_id in df_g["mouse_id"].unique():
        mdf = df_g[df_g["mouse_id"] == mouse_id]

        ses1_df = mdf[mdf["session_id"] == label_1].reset_index(drop=True)
        ses2_df = mdf[mdf["session_id"] == label_2].reset_index(drop=True)
        
        stim_ses1 = np.flatnonzero(ses1_df[column].values == 1)
        stim_ses2 = np.flatnonzero(ses2_df[column].values == 1)
  

        first3_ses1 = stim_ses1[:3]
        first3_ses2 = stim_ses2[:3]

        ses1_cums, ses2_cums = [], []

        for onset in first3_ses1:
            start, end = onset - 50, onset + 100
     
            seg = ses1_df.iloc[start:end+1][signal_col]
            base = seg.iloc[:50].mean()
            cum = cumulative_trapezoid(seg.values - base, dx=1/25, initial=0)
            ses1_cums.append(cum); ses_1_all.append(cum)
            session1.setdefault(label_1, []).append(cum)
            if time_ses_1 is None:
                time_ses_1 = np.arange(start, end+1) - onset

        for onset in first3_ses2:
            start, end = onset - 50, onset + 100
            seg = ses2_df.iloc[start:end+1][signal_col]
            base = seg.iloc[:50].mean()
            cum = cumulative_trapezoid(seg.values - base, dx=1/25, initial=0)
            ses2_cums.append(cum); ses_2_all.append(cum)
            session_2.setdefault(label_2, []).append(cum)
            if time_ses_2 is None:
                time_ses_2 = np.arange(start, end+1) - onset

        if ses1_cums:
            ses1_means[mouse_id] = np.nanmean(np.vstack(ses1_cums), axis=0)
        if ses2_cums:
            ses2_means[mouse_id] = np.nanmean(np.vstack(ses2_cums), axis=0)

        if ses1_cums and ses2_cums:
            records.append({"mouse_id": mouse_id, "group": "Control",
                            "max_integral": float(max(np.nanmax(c) for c in ses1_cums)), 
                            "mean_integral": float(np.nanmedian([c for c in ses1_cums]))})
            records.append({"mouse_id": mouse_id, "group": "Methimazole",
                            "max_integral": float(max(np.nanmax(c) for c in ses2_cums)),
                            "mean_integral": float(np.nanmedian([c for c in ses2_cums]))})

    # Panel 1: means (overall + per-session + per-mouse overlays)
    blue_mean = sns.color_palette("Blues", 9)[-2]
    red_mean  = sns.color_palette("Reds", 9)[-2]
    blue_sess = sns.color_palette("blend:#7AB,#EDA", 6)
    red_sess  = sns.color_palette("blend:#cd5c5c,#8B0000", 6)

    if ses_1_all:
        mean_ctrl = np.nanmean(np.vstack(ses_1_all), axis=0)
        ax[0].plot(time_ses_1, mean_ctrl, color=blue_mean, lw=2.0, label="Control mean")
    if ses_2_all:
        mean_meth = np.nanmean(np.vstack(ses_2_all), axis=0)
        ax[0].plot(time_ses_2, mean_meth, color=red_mean, lw=2.0, label="Meth mean")

    for si, (sess, cums) in enumerate(session1.items()):
        m = np.nanmean(np.vstack(cums), axis=0)
        ax[0].plot(time_ses_1, m, color=blue_sess[min(si, len(blue_sess)-1)], lw=1.0, alpha=0.6, label=f"{sess}")
    for si, (sess, cums) in enumerate(session_2.items()):
        m = np.nanmean(np.vstack(cums), axis=0)
        ax[0].plot(time_ses_2, m, color=red_sess[min(si, len(red_sess)-1)], lw=1.0, alpha=0.6, label=f"{sess}")

    # per-mouse mean overlays
    for mid, mcurve in ses1_means.items():
        ax[0].plot(time_ses_1, mcurve, color=blue_mean, lw=0.8, alpha=0.4)
    for mid, mcurve in ses2_means.items():
        ax[0].plot(time_ses_2, mcurve, color=red_mean, lw=0.8, alpha=0.4)

    ax[0].axvline(0, color='k', ls='--', lw=0.8)
    ax[0].set_title(f"{genotype}: Control vs Meth (first 3 sampling)")
    ax[0].set_ylabel("Cumulative integral")
    ax[0].set_xlabel("Frames (rel. to sampling)")
    ax[0].set_ylim(-5, 5)
    ax[0].legend(fontsize='x-small', ncol=2, frameon=False)

    if records:
        df_pts = pd.DataFrame(records)
        pal = {'Control': 'steelblue', 'Methimazole': 'indianred'}
        sns.scatterplot(data=df_pts, x="group", y="max_integral", hue="group", palette=pal,
                        s=30, alpha=0.5, ax=ax[2], legend=False)
        sns.pointplot(data=df_pts, x="group", y="max_integral",
                      estimator=np.median, errorbar=("ci",95), join=True, palette=pal,
                      ax=ax[2], markers="o", lw=0.5, markersize=7)
        ax[2].set_title(f"{genotype}: Control vs Methimazole")
        ax[2].set_ylabel("Max integral of dF/F")
        ax[2].set_ylim(-1, 25)
        ax[2].tick_params(axis='x', rotation=45)

        sns.scatterplot(data=df_pts, x="group", y="mean_integral", hue="group", palette=pal,
                        s=30, alpha=0.5, ax=ax[1], legend=False)
        sns.pointplot(data=df_pts, x="group", y="mean_integral",
                      estimator=np.median, errorbar=("ci",95), join=True, palette=pal,
                      ax=ax[1], markers="o", lw=0.5, markersize=7)
        ax[1].set_title(f"{genotype}: Control vs Methimazole")
        ax[1].set_ylabel("Mean integral of dF/F")
        ax[1].set_ylim(-1, 1.5)
        ax[1].tick_params(axis='x', rotation=45)

    fig.savefig(rf'F:\upgrade_figures\panel_B\panel_B_{genotype}_sesh_comparison_{column}.svg')
    plt.close(fig)


def sniff_trace_example(mouse_id=1106077):
    session_path = r'F:\social_sniffing\derivatives\1106077\2025-09-23T12-31-23'
    ses = session.Session(mouse_id=mouse_id, session_path=session_path)
    sniff_raw = ses.sniffing.analog
    sniff_inst_hz, peak_times = ses.sniffing.extract_sniff_freq()
    valve1 = ses.sniffing.get_valve_open( 'SupplyPort1')

    # pick a window start (e.g., last 5 seconds)
    t_start = sniff_raw.index[400000]
    t_end = t_start + pd.Timedelta(seconds=10)

    # slice and plot
    win_raw = sniff_raw.loc[t_start:t_end]
    win_hz  = sniff_inst_hz.loc[t_start:t_end]

    fig, ax = uplt.subplots(nrows=2, figsize=(12, 6), sharex=True)
    ax[0].plot(win_raw.index, win_raw.values, color='gray', alpha=0.7, label='Raw sniff signal')
    ax[1].plot(win_hz.index, win_hz.values, color='blue', alpha=0.8, label='Instantaneous sniff frequency (Hz)')
    ax[0].grid(False)
    ax[1].grid(False)
    plt.axis('off')
    fig.savefig(rf'F:\upgrade_figures\sniff_trace_example_mouse_{mouse_id}.svg', dpi)
    plt.show()
 

for genotype in ['VGAT']:
    #panel_E_genotype(df, genotype=genotype, column='stim_onset', signal_col='deltaF_z', condition='Methimazole')
    panel_E_genotype(df, genotype=genotype, column='sampling', signal_col='deltaF_z', condition='Aversive')
    #comp_session(df, genotype=genotype, column='sampling', signal_col='deltaF_z')
    #Methimazole_comp(df, column='sampling', signal_col='deltaF_z', genotype=genotype)
exit()
def plot_deltaF_all(df, column='stim_onset', signal_col='deltaF_z', outpath=Path(r'F:\upgrade_figures')):
    array = [[1,1,1],[2,2,2],[3,3,3],[4,4,4]]
    fig, ax = uplt.subplots(array, figsize=(3, 8), share=False, includepanels=True)

    genotypes = list(df["genotype"].unique())
    for i, genotype in enumerate(genotypes):
        df_g = df[df["genotype"] == genotype].copy()
        stim_times = df_g.index[df_g[column] == 1]
        if len(stim_times) == 0:
            ax[i].set_title(f"{genotype} — no {column}")
            continue

        aligned = []
        rel_idx = None
        for t in stim_times:
            start, end = t - 50, t + 50
            if start < df_g.index.min() or end > df_g.index.max():
                continue
            seg = df_g.loc[start:end, signal_col]
            b = min(25, len(seg))
            base = seg.iloc[:b].mean() if b > 0 else 0.0
            y = seg.values - base
            aligned.append(pd.Series(y))
            if rel_idx is None:
                rel_idx = np.arange(start, end+1) - t

        if not aligned:
            ax[i].set_title(f"{genotype} — insufficient window")
            continue

        aligned_df = pd.concat(aligned, axis=1)
        aligned_df.index = rel_idx
        aligned_df.columns = [f'trial_{j+1}' for j in range(aligned_df.shape[1])]

        for col in aligned_df.columns:
            ax[i].plot(aligned_df.index, aligned_df[col], color='gray', alpha=0.15)
        ax[i].plot(aligned_df.index, aligned_df.mean(axis=1), color='black', linewidth=2, label='mean')

        ax[i].axvline(0, color='red', linestyle='--')
        ax[i].set_title(genotype)
        ax[i].set_xlabel('Frames (relative to stim)')
        ax[i].set_ylabel(signal_col)
        ax[i].set_ylim(-2, 6)
        ax[i].set_xlim(-25, 50)
        ax[i].grid(False)

    fig.savefig(outpath / f'{signal_col}_aligned_to_{column}.svg', dpi=300)
plot_deltaF_all(df, column='stim_onset', signal_col='deltaF_z', outpath=Path(r'F:\upgrade_figures'))
plot_deltaF_all(df, column='reward', signal_col='deltaF_z', outpath=Path(r'F:\upgrade_figures'))

exit()
plot_deltaF(df_dlight, column='stim_onset', signal_col='deltaF_z', outpath=Path(r'F:\upgrade_figures'))
plot_deltaF(df_dlight, column='reward', signal_col='deltaF_z', outpath=Path(r'F:\upgrade_figures'))
for mouse in [1106077,1106078,1106079,1125563,1125131,1125132, 1106008, 1106009, 1106010,1125555, 1125561]:
    #panel_D(df, mouse_id=mouse, column='reward', signal_col='deltaF_z', condition='Control')
    panel_EF(df, mouse_id=mouse, column='sampling', signal_col='deltaF_z', condition='Control')

exit()
