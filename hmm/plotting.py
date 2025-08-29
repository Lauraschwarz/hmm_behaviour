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

from constants import ARENA, ARENA_VERTICES, inner_vertices, TRIAL_LENGTH
sys.path.append(r'C:\Users\Laura\social_sniffing\sniffies')
import session
from fancyimpute import IterativeSVD
from sklearn.preprocessing import StandardScaler
from ultra import get_exploration_and_signal_grid
from movement.utils.vector import compute_norm
from params import hmm_parameters_object
from hmm_utils import (
    load_pickle_file, extract_frame_range, 
    get_param_string, interpolate_circular_nan
)
cmp = sns.diverging_palette(220, 20, as_cmap=True)
cmp2 = sns.color_palette("tab20", as_cmap=True)

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


states = np.load(r'F:\social_sniffing\behaviour_model\hmm\output\1125132\block0_N=6_pairwise_speed_accel_discon_disbum_disport0_disport1\K=11\states_K=11_block0_N=6_pairwise_speed_accel_discon_disbum_disport0_disport1.npy')
file_path = r'F:\social_sniffing\derivatives\1125132\2025-07-02T14-37-53\hmm_features.pkl'
with open(file_path, "rb") as file:
    featarr_combined = pickle.load(file)
feature_names = list(featarr_combined.columns)
featarr_combined = featarr_combined.iloc[1550:]
session = session.Session(
    mouse_id='1125140',
    session_path=r'F:\social_sniffing\derivatives\1125132\2025-07-02T14-37-53')
video_path = Path(r'F:\social_sniffing\rawdata\1125132\2025-07-02T14-37-53\Video\1125132.avi')

track = session.track
track_reindexed = reindex_movement_data(track, session.sniffing.start)
absolute_times = pd.to_datetime(track.start) + pd.to_timedelta(track_reindexed.time.values, unit='s')
absolute_times = absolute_times[1550:]
centroid_pos = track_reindexed.position.sel(individuals='2')
centroid_pos = centroid_pos.isel(time=slice(1550, None))
centroid_pos = centroid_pos.assign_coords(time=absolute_times)
centroid_pos = centroid_pos.assign_coords(states=('time', states))
centroid_pos.attrs['hmm'] = states


sniff_inst_hz = session.sniffing.extract_sniff_freq()
port0_poke_timestamps = session.sniffing.get_pokes('DIPort0')
port1_poke_timestamps = session.sniffing.get_pokes( 'DIPort1')
valve0_open_ts = session.sniffing.get_valve_open( 'SupplyPort0')
valve1_open_ts = session.sniffing.get_valve_open( 'SupplyPort1')
stim_onsets = session.sniffing.get_stim_onsets()

batch = 3
n_trials = len(stim_onsets)

for batch_start in range(0, n_trials, batch):
    batch_end = min(batch_start + batch, n_trials)
    features = featarr_combined.loc[stim_onsets[batch_start]:stim_onsets[batch_end]]
  
    
    stimulus_batch = stim_onsets[batch_start:batch_end]

    arena_fig, arena_axes = get_exploration_and_signal_grid(
            n_conditions=batch,
            add_row=True,
            n_add_rows=4,
            figsize=(14, 11)
        )
    for i, onset in enumerate(stimulus_batch):

        position_by_time_ax = arena_axes[i]
        position_by_deltaf_ax = arena_axes[i + len(stimulus_batch)]
        speed_trace_ax = arena_axes[i + (len(stimulus_batch) * 2)]
        acceleration_trace_ax = arena_axes[i + (len(stimulus_batch) * 3)]
        distance_trace_ax = arena_axes[i + (len(stimulus_batch) * 4)]




        end = onset +pd.Timedelta(seconds=45)
        features = featarr_combined.loc[onset:end]
  
        positions = centroid_pos.sel(time=slice(features.index[0], features.index[-1]))
        suppress_colorbar = i not in [len(stimulus_batch) - 1,  # plot colorbar only for last plot (assuming cbar same for all columns)
                                    int(len(stimulus_batch) * 2) - 1]
        roi1 = load_roi('port0')
        
        roi2 = load_roi('port1')
        roi3 = load_roi('Arena')
        roi3.plot(position_by_time_ax, facecolor='lightgrey', linewidth=0, alpha=1)
        roi3.plot(position_by_deltaf_ax, facecolor='lightgrey', linewidth=0, alpha=1)
        roi2.plot(position_by_time_ax, facecolor='Grey', linewidth=0, alpha=1)
        roi2.plot(position_by_deltaf_ax, facecolor='Grey', linewidth=0, alpha=1)
        roi1.plot(position_by_time_ax, facecolor='Grey', linewidth=0, alpha=1)
        roi1.plot(position_by_deltaf_ax, facecolor='Grey', linewidth=0, alpha=1)

        plot_feature(ax=speed_trace_ax, trace=features['smoothed_speed'], trace_name='smoothed_speed')
        plot_feature(ax=acceleration_trace_ax, trace=features['smoothed_acceleration'], trace_name='smoothed_acceleration')
        plot_feature(ax=distance_trace_ax, trace=features['abdomen_abdomen'], trace_name='abdomen_abdomen')

        plot_centroid_trajectory(
                        positions,
                        keypoints='abdomen',
                        manual_color_var='states',
                        ax=position_by_deltaf_ax,
                        linestyle="-",
                        marker=".",
                        s=3,
                        cmap=cmp2,
                        suppress_colorbar=suppress_colorbar,
                    )

        plot_centroid_trajectory_by_states(
                        positions,
                        keypoints='abdomen',
                        ax=position_by_time_ax,
                        linestyle="-",
                        marker=".",
                        s=3,
                        cmap=cmp,
                        suppress_colorbar=suppress_colorbar,
                    )

    plt.tight_layout()

    plt.show()

# featarr_combined = featarr_combined.iloc[1550:]

plot_centroid_trajectory_by_states(
                centroid_pos,
                c=states,
                speed=featarr_combined['smoothed_speed'],
                keypoints='abdomen',
                linestyle="-",
                marker=".",
                s=3,
                cmap=cmp2,
                suppress_colorbar=suppress_colorbar,
            )
# for column in enumerate(featarr_combined.columns):
#     col_idx, feature_name = column
#     ax = arena_axes[col_idx]
#     feature_data = featarr_combined[feature_name].values
#     ax1 = ax.twinx()    
#     ax.plot(feature_data, lw=0.5, marker='.', markersize=0.5, color='lightgray', alpha=1)
#     ax1.plot(states, lw=0.1, marker='.', color='k', alpha=0.2, markersize=0.2)
#     ax.set_title(feature_name)
#     ax.set_ylabel('Value')
#     ax1.set_ylabel('State', color='k')
#     ax.set_xlabel('Frame')
#     ax.grid(True)

#     plot_centroid_trajectory(
#                 centroid_pos,
#                 keypoints='abdomen',
#                 manual_color_var='states',
#                 ax=arena_axes[10],
#                 linestyle="-",
#                 marker=".",
#                 s=3,
#                 cmap=cmp,
#                 suppress_colorbar=True,
#             )
   