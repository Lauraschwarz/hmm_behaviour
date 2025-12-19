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

root_path=r"F:\social_sniffing\derivatives"
hmm_base_outpath=r"F:\social_sniffing\behaviour_model\hmm\output"
k=5
N_features=5 # Number of features to extract
block_id=0
seed=42

fps = 50


    # -------------------------
    # Feature Extraction
    # -------------------------

all_files = list(Path(root_path).rglob('*hmm_features.csv'))

for file in all_files:
    print(f"Processing file: {file}")
    hmm_features = pd.read_csv(str(file))
    output_path = Path(file).parent
    selected_columns = ['smoothed_speed', 
                        'smoothed_acceleration', 
                        'abdomen_abdomen', 
                        'snout_groin', 
                        'abdomen_port0', 
                        'abdomen_port1', 
                        'sniff_freq'
                    ]  # example columns
    featarr_combined = hmm_features[selected_columns].copy()

    feature_names = list(featarr_combined.columns)

    # -------------------------
    # Feature Selection
    # -------------------------
    N = N_features


    # -------------------------
    # Imputation + Normalization
    # -------------------------
    print("NaNs before imputation:", np.isnan(featarr_combined).sum())
    rank_safe = min(5, min(featarr_combined.shape) - 1)
    featarr_combined = np.where(np.isfinite(featarr_combined), featarr_combined, np.nan)

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
    plt.savefig(output_path / f"explained_variance.png")
    # plt.show()

    # Identify top features contributing to PC1
    abs_loadings = np.abs(Vt[0])
    top_feature_indices = np.argsort(abs_loadings)[::-1]
    top_features = [(feature_names[i], Vt[0][i]) for i in top_feature_indices]

    print("Top features contributing to the 1st principal component:")
    idx_best_features = []
    for name, loading in top_features[:N]:
        print(f"{name}: loading = {loading:.4f}")
        idx_best_features.append(feature_names.index(name))

    # Save results
    top_features_dict = {name: loading for name, loading in top_features[:N]}
    with open(output_path / f"top_features.txt", "w") as f:
        for name, loading in top_features_dict.items():
            f.write(f"{name}: {loading:.4f}\n")

    df_imputed_scaled = pd.DataFrame(A_completed_normalized, columns=feature_names, index=hmm_features.index)
    df_imputed_scaled.to_csv(output_path / f"completed_distances_imputed.csv", index=False)
    np.save(output_path / f"completed_distances_imputed.npy", A_completed_normalized)
    np.save(output_path / f"feature_array.npy", featarr_combined)


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
    #plt.show()

    plt.figure(figsize=(10, 4))
    plt.bar(range(len(Vt[0])), Vt[0])
    plt.title("Feature Loadings on First Principal Component")
    plt.xlabel("Feature Index")
    plt.ylabel("Loading")
    #plt.show()




exit()


