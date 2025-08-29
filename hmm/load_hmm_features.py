import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import pickle
from fancyimpute import IterativeSVD
from sklearn.preprocessing import StandardScaler

from movement.utils.vector import compute_norm
from params import hmm_parameters_object
from hmm_utils import (
    load_pickle_file, extract_frame_range, 
    get_param_string, interpolate_circular_nan
)


# -------------------------
# Setup & Parameters
# -------------------------
if hmm_parameters_object.seed is not None:
    np.random.seed(hmm_parameters_object.seed)

root_path = Path(hmm_parameters_object.root_path)
hmm_base_outpath = Path(hmm_parameters_object.hmm_base_outpath)
mouse_id = hmm_parameters_object.mouse_id
session_id = hmm_parameters_object.session_id
block_id = hmm_parameters_object.block_id
param_string = get_param_string(hmm_parameters_object)
fps = 50

print(f"Param string: {param_string}")

mouse_dir = root_path / 'derivatives' / mouse_id / session_id / 'behav'
output_path = hmm_base_outpath / mouse_id / param_string
output_path.mkdir(parents=True, exist_ok=True)




# -------------------------
# Feature Extraction
# -------------------------
if hmm_parameters_object.seed is not None:
    # Set random seed for reproducibility
    seed_num = hmm_parameters_object.seed
    np.random.seed(seed_num)

#load dataframe

file_path = r'F:\social_sniffing\derivatives\1125132\2025-07-02T14-37-53\hmm_features.pkl'
with open(file_path, "rb") as file:
    featarr_combined = pickle.load(file)
feature_names = list(featarr_combined.columns)


# -------------------------
# Feature Selection
# -------------------------
N = hmm_parameters_object.N_features


# -------------------------
# Imputation + Normalization
# -------------------------
print("NaNs before imputation:", np.isnan(featarr_combined).sum())
rank_safe = min(5, min(featarr_combined.shape) - 1)

A_completed = IterativeSVD(rank=rank_safe, max_iters=200).fit_transform(featarr_combined)
A_completed_normalized = StandardScaler().fit_transform(A_completed)


# -------------------------
# PCA Analysis
# -------------------------
U, S, Vt = np.linalg.svd(A_completed_normalized, full_matrices=False)
explained_var = (S ** 2) / np.sum(S ** 2)

plt.plot(np.cumsum(explained_var))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(True)
plt.savefig(output_path / f"explained_variance_{param_string}.png")
plt.show()

# Identify top features contributing to PC1
abs_loadings = np.abs(Vt[0])
top_feature_indices = np.argsort(abs_loadings)[::-1]
top_features = [(feature_names[i], Vt[0][i]) for i in top_feature_indices]

print("Top 5 features contributing to the 1st principal component:")
idx_best_features = []
for name, loading in top_features[:N]:
    print(f"{name}: loading = {loading:.4f}")
    idx_best_features.append(feature_names.index(name))

# Save results
top_features_dict = {name: loading for name, loading in top_features[:N]}
with open(output_path / f"top_features_{param_string}.txt", "w") as f:
    for name, loading in top_features_dict.items():
        f.write(f"{name}: {loading:.4f}\n")

np.save(output_path / f"completed_distances_{param_string}.npy", A_completed_normalized)
np.save(output_path / f"feature_array_{param_string}.npy", featarr_combined)


# -------------------------
# Visualization
# -------------------------
plt.figure(figsize=(12, 4))
plt.imshow(A_completed_normalized.T, aspect="auto", cmap="viridis")
plt.title("Imputed Features Over Time")
plt.xlabel("Frame")
plt.ylabel("Feature Index")
plt.colorbar(label="Value")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.bar(range(len(Vt[0])), Vt[0])
plt.title("Feature Loadings on First Principal Component")
plt.xlabel("Feature Index")
plt.ylabel("Loading")
plt.show()





exit()

































import matplotlib
matplotlib.use('qt5Agg')
import matplotlib.pyplot as plt
import pathlib
from fancyimpute import IterativeSVD
import numpy as np
from movement.utils.vector import compute_norm
import pandas as pd
import xarray as xr
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from params import hmm_parameters_object
from hmm_utils import load_pickle_file, extract_frame_range, get_param_string, interpolate_circular_nan

if hmm_parameters_object.seed is not None:
    # Set random seed for reproducibility
    seed_num = hmm_parameters_object.seed
    np.random.seed(seed_num)

root_path = Path(hmm_parameters_object.root_path)
hmm_base_outpath = Path(hmm_parameters_object.hmm_base_outpath)
mouse_id = hmm_parameters_object.mouse_id
session_id = hmm_parameters_object.session_id
param_string = get_param_string(hmm_parameters_object)
print(f"Param string: {param_string}")

mouse_dir = root_path / 'derivatives' / mouse_id / session_id / 'behav'
output_path = hmm_base_outpath / mouse_id / param_string
output_path.mkdir(parents=True, exist_ok=True)

block_transition_path = mouse_dir / "blocks_with_transitions.pkl"
distances_path = mouse_dir / "pairwise_distances.pkl"
position_path = mouse_dir / "tracking_data.nc"
behaviour_data_path = mouse_dir / "session_behav_data.csv"

block_id = hmm_parameters_object.block_id
fps = 40
# start_frame = 30*fps# for sub-004=1640  for sub-006=1000
# end_frame = 1057*fps# for sub-004=28800   # for sub-006=28800


# Load the dataset
ds = xr.open_dataset(position_path)
# ds = load_poses.from_sleap_file(position_path, fps=fps)
pairwise_data = load_pickle_file(distances_path)
behaviour_data = pd.read_csv(behaviour_data_path)
block_transitions = load_pickle_file(block_transition_path)
for i, block in enumerate(block_transitions):
    print(block)
    if block.get('id') == block_id:
        present_object = block.get('original_label')

print('ds', ds)
print('pairwise_data', pairwise_data)
print('behaviour_data', behaviour_data)

# Extract the correct block from frame range or block transitions
position_subset, pairwise_data_subset, behaviour_data_subset, velocity, accel, speed, head_direction = extract_frame_range(ds, pairwise_data, behaviour_data, 
                                                                                                           block_transition_path=block_transition_path, 
                                                                                                           blocks=block_id)


# Extract different data features
object_distance_body = behaviour_data_subset[f'{present_object}_distance']
object_distance_head = behaviour_data_subset[f'{present_object}_head_distance']
distance_to_rim_body = behaviour_data_subset['body_center_distance_rim']
object_head_direction = behaviour_data_subset[f'{present_object}_head_direction']
inside_object = behaviour_data_subset[f'inside_{present_object}'] # boolean
inside_object = inside_object*2 # convert boolean to 0 and 2 

da = position_subset.sel(keypoints="Body Centre")

# Compute norms and handle NaNs
speed = compute_norm(velocity).squeeze()
speed_nonan = pd.Series(speed).interpolate(method='linear', limit_direction='both').to_numpy()
accel_norm = compute_norm(accel).squeeze()
accel_norm_nonan = pd.Series(accel_norm).interpolate(method='linear', limit_direction='both').to_numpy()

object_distance_body_norm_nonan = pd.Series(object_distance_body).interpolate(method='linear', limit_direction='both').to_numpy()
object_distance_head_norm_nonan = pd.Series(object_distance_head).interpolate(method='linear', limit_direction='both').to_numpy()
distance_to_rim_body_norm_nonan = pd.Series(distance_to_rim_body).interpolate(method='linear', limit_direction='both').to_numpy()

# Account for the wrapped angle 2pi data
head_direction_interp = interpolate_circular_nan(head_direction)
object_head_direction_interp = interpolate_circular_nan(object_head_direction)
hd_x = np.cos(head_direction_interp)
hd_y = np.sin(head_direction_interp)
obj_hd_x = np.cos(object_head_direction_interp)
obj_hd_y = np.sin(object_head_direction_interp)


# features = [1, 31, 42, 49, 25] # choose specific feature columns by index
N = hmm_parameters_object.N_features
variances = np.var(pairwise_data_subset.values, axis=0)
top_k = np.argsort(variances)[-N:][::-1]
print(f"Top {N} features by variance:", top_k)
features = list(top_k)
featarr = []
feature_names = []

#featarr = np.array((len(data.values), len(features)))]
plt.figure()
if hmm_parameters_object.pairwise:
    for i, x in enumerate(features): 
        print(x)
        #featarr[:,i] = data.values[:, x]
        plt.plot(pairwise_data_subset.values[:,x])
        featarr.append(pairwise_data_subset.values[:, x])
        feature_names.append(f"pairwise_{i}")
if hmm_parameters_object.speed:
    featarr.append(speed_nonan)
    feature_names.append("speed")
if hmm_parameters_object.acceleration:
    featarr.append(accel_norm_nonan)
    feature_names.append("acceleration")
if hmm_parameters_object.head_direction:
    featarr.append(hd_x)
    feature_names.append("head_direction_x")
    featarr.append(hd_y)
    feature_names.append("head_direction_y")
if hmm_parameters_object.object_distance_body:
    featarr.append(object_distance_body_norm_nonan)
    feature_names.append("object_distance_body")
if hmm_parameters_object.object_distance_head:
    featarr.append(object_distance_head_norm_nonan)
    feature_names.append("object_distance_head")
if hmm_parameters_object.distance_to_rim_body:
    featarr.append(distance_to_rim_body_norm_nonan)
    feature_names.append("distance_to_rim_body")
if hmm_parameters_object.object_head_direction:
    featarr.append(obj_hd_x)
    feature_names.append("object_head_direction_x")
    featarr.append(obj_hd_y)
    feature_names.append("object_head_direction_y")
if hmm_parameters_object.inside_object:
    featarr.append(inside_object) # boolean
    feature_names.append("inside_object")

featarr_combined = np.stack(featarr, axis=1)  # Shape: (T, n_features)



plt.show()
print("NaNs before imputation:", np.isnan(featarr_combined).sum())
min_dim = min(featarr_combined.shape)
rank_safe = min(5, min_dim - 1)  # rank must be < min_dim
A_completed = IterativeSVD(rank=rank_safe, max_iters=200).fit_transform(featarr_combined)

#make sure it is normalized
scaler = StandardScaler()
A_completed_normalized = scaler.fit_transform(A_completed)

# A_centered = A_completed - A_completed.mean(axis=0)
# U, S, Vt = np.linalg.svd(A_centered, full_matrices=False)
U, S, Vt = np.linalg.svd(A_completed_normalized, full_matrices=False)

explained_var = (S ** 2) / np.sum(S ** 2)
plt.plot(np.cumsum(explained_var))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.grid(True)
#save the plot
plt.savefig(output_path / f"explained_variance_{param_string}.png")
plt.show()


abs_loadings = np.abs(Vt[0])
top_feature_indices = np.argsort(abs_loadings)[::-1]

top_features = [(feature_names[i], Vt[0][i]) for i in top_feature_indices]


print("Top 5 features contributing to the 1st principal component:")
idx_best_features = []
for name, loading in top_features[:N]:
    print(f"{name}: loading = {loading:.4f}")
    idx_best_features.append(feature_names.index(name))

# Save the top features and their loadings
top_features_dict = {name: loading for name, loading in top_features[:N]}
print("Top_features and their loadings:", top_features_dict)
with open(output_path / f"top_features_{param_string}.txt", "w") as f:
    for name, loading in top_features_dict.items():
        f.write(f"{name}: {loading:.4f}\n")

np.save(output_path / f"completed_distances_{param_string}.npy", A_completed_normalized)

plt.figure(figsize=(12, 4))
plt.imshow(A_completed_normalized.T, aspect='auto', cmap='viridis')
plt.title("Imputed Features Over Time")
plt.xlabel("Frame")
plt.ylabel("Feature Index")
plt.colorbar(label="Value")
plt.tight_layout()
plt.show()


np.save(output_path / f"feature_array_{param_string}.npy", featarr)

plt.figure(figsize=(10, 4))
plt.bar(range(len(Vt[0])), Vt[0])
plt.title("Feature Loadings on First Principal Component")
plt.xlabel("Feature Index")
plt.ylabel("Loading")
plt.show()


