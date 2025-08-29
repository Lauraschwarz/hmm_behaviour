from hmm_run_class import HMMProcessor
import os
import xarray as xr
from hmm_utils import get_param_string
from params import hmm_parameters_object

def run_hmm_analysis(hmm_parameters_object, start_frame=30000, end_frame=36000, K_range=None):
    """
    Run HMM analysis for one or multiple (mouse_id, session_id) pairs.

    Parameters 
    ----------
    hmm_parameters_object : object
        Your HMM parameters object.
    start_frame : int
        Start frame for analysis.
    end_frame : int
        End frame for analysis.
    K_range : tuple or list, optional
        If provided, should be (start_K, end_K). If None, runs once with current hmm_parameters_object.K.
    """  
    
    Ks_to_run = [hmm_parameters_object.k]  # Default: single run
    if K_range is not None:
        Ks_to_run = list(range(K_range[0], K_range[1] + 1))

    # Ensure mouse_id and session_id are iterable pairs
    if isinstance(hmm_parameters_object.mouse_id, list) and isinstance(hmm_parameters_object.session_id, list):
        mouse_session_pairs = list(zip(hmm_parameters_object.mouse_id, hmm_parameters_object.session_id))
    else:
        mouse_session_pairs = [(hmm_parameters_object.mouse_id, hmm_parameters_object.session_id)]
    
    for mouse_id, session_id in mouse_session_pairs:
        print(f"\n=== Running analysis for Mouse: {mouse_id}, Session: {session_id} ===\n")
        
        # Update the parameter object for this run
        hmm_parameters_object.mouse_id = mouse_id
        hmm_parameters_object.session_id = session_id

        for K in Ks_to_run:
            print(f"\nRunning HMM analysis for K={K}...\n")
            hmm_parameters_object.k = K  # Update K for this run

            # Initialize processor
            processor = HMMProcessor(
                parameters_object=hmm_parameters_object,
                show_plots=False,
                save_plots=True,  
                create_animations=True, 
                save_animations=True, 
                fit_new_model=True  
            )

            # Run full analysis
            param_string = processor.run_full_analysis(start_frame=start_frame, end_frame=end_frame)

            # Ensure param_string is set correctly
            param_string = get_param_string(hmm_parameters_object)

            # Cross-validation
            if mouse_id == "sub-006_id-1123131_type-wtshelterswitch": 
                alt_id = "sub-005_id-1122877_type-wtshelterswitch"
            else:
                alt_id = mouse_id

            other_dataset_path = fr"O:\slenzi\raphe\behaviour_model\other\hmm_data\{alt_id}\{param_string}"
            other_states_cross_val, log_likelihood = processor.cross_validate(other_dataset_path)
            print(f"Cross-validation log likelihood for K={K}: {log_likelihood}") 

        

    print("All runs completed.")

# --- USAGE ---

# Run for a single K
# run_hmm_analysis(hmm_parameters_object)

run_hmm_analysis(hmm_parameters_object , K_range=(2, 6))
exit()


























from hmm_run_class import HMMProcessor
import os
import xarray as xr
from hmm_utils import get_param_string

from params import hmm_parameters_object

# Initialize processor with options
processor = HMMProcessor(
    parameters_object=hmm_parameters_object,
    show_plots=False,
    save_plots=True,  
    create_animations=False,
    save_animations=False, 
    fit_new_model=True  # Load existing model
)

# Run full analysis
param_string = processor.run_full_analysis(start_frame=30000, end_frame=36000)

# Ensure param_string is not None or empty
param_string = get_param_string(hmm_parameters_object)


other_dataset_path = fr"O:\slenzi\raphe\behaviour_model\other\hmm_data\sub-006_id-1123131_type-wtshelterswitch\{param_string}"
other_states_cross_val, log_likelihood = processor.cross_validate(other_dataset_path)
# print(f"Cross-validation results: {log_likelihood}")

tracking_data_xr_path = os.path.join(hmm_parameters_object.root_path, 'derivatives', hmm_parameters_object.mouse_id, hmm_parameters_object.session_id, "behav", "tracking_data.nc")
with xr.open_dataset(tracking_data_xr_path) as tracking_data_xr:
    tracking_data_xr = tracking_data_xr.load()  # Load into memory and close file

states_csv_path = os.path.join(hmm_parameters_object.hmm_base_outpath, hmm_parameters_object.mouse_id, param_string, f"K={hmm_parameters_object.k}", f"states_{param_string}.csv")
processor.add_states_to_xarray(states_csv_path, tracking_data_xr)

# processor.create_video_annotations(start_frame=32000, end_frame=37000)
# Print summary
# print(processor.get_results_summary())


# Minimal usage (no plots, no animations)
# processor = HMMProcessor(
#     show_plots=False,
#     save_plots=False,
#     create_animations=False
# )
# processor.load_data()
# processor.fit_hmm()
# processor.predict_states()



# assess the fitting onto another data set using viterbi algorithm


exit()






























# import numpy as np
# import ssm
# import pandas as pd
# from itertools import groupby
# import pickle
# from pathlib import Path
# from params import hmm_parameters_object
# from hmm_utils import get_param_string

# # Set random seed for reproducibility
# if hmm_parameters_object.seed is not None:
#     # Set random seed for reproducibility
#     seed_num = hmm_parameters_object.seed
#     np.random.seed(seed_num)

# hmm_base_outpath = Path(hmm_parameters_object.hmm_base_outpath)
# mouse_id = hmm_parameters_object.mouse_id
# root_path = Path(hmm_parameters_object.root_path)
# session_id = hmm_parameters_object.session_id
# output_path = hmm_base_outpath
# session_path = root_path / "derivatives" / mouse_id / session_id / "behav"

# # Path to feature array
# # mouse_dir = [home_dir / "sub-004_id-1122876_type-wtshelterswitch",
# #                 home_dir / "sub-005_id-1122877_type-wtshelterswitch",
# #                 home_dir / "sub-006_id-1123131_type-wtshelterswitch"             
# # ]

# #single session
# param_string = get_param_string(hmm_parameters_object)
# mouse_dir = [output_path / mouse_id / param_string]
# mouse_dir[0].mkdir(parents=True, exist_ok=True)
# mouse_dir_feature = mouse_dir[0] / f"K={hmm_parameters_object.k}"
# mouse_dir_feature.mkdir(parents=True, exist_ok=True)

# print('param_string:', param_string)
    
# K = hmm_parameters_object.k  # number of hidden states
# block_id = hmm_parameters_object.block_id


# # Load and preprocess each session separately
# if len(mouse_dir) > 1:

#     features = []
#     for dir_path in mouse_dir:
#         arr = np.load(dir_path / "feature_array.npy").T  # transpose if needed
#         arr[~np.isfinite(arr)] = np.nan                                           
#         arr = pd.DataFrame(arr).interpolate(axis=0, limit_direction='both').to_numpy()
#         arr = arr[1550:]  # trim start if needed
#         features.append(arr) 
#     D = features[0].shape[1]
#     hmm = ssm.HMM(K=K, D=D, observations="gaussian")
#     hmm.fit(features, method="em", num_iters=500)
    
# else:    

#     features_full = np.load(mouse_dir[0] / f"feature_array_{param_string}.npy")
#     features_full = features_full.T
#     print("Original shape:", features_full.shape)

#     features_full[~np.isfinite(features_full)] = np.nan
#     # Use pandas to interpolate across time (axis=0)
#     features_interp = pd.DataFrame(features_full).interpolate(axis=0, limit_direction='both').to_numpy()

#     # Truncate start of data
#     features = features_interp[1550:]
#     D = features.shape[1]

#     hmm = ssm.HMM(K=K, D=D, observations="gaussian", random_state=seed_num)
#     hmm.fit([features], method="em", num_iters=500)



# # Create HMM model
# if len(mouse_dir) > 1:
#     # Decode each session into its own array
#     states_list = [hmm.most_likely_states(sess) for sess in features]
    
#     # Save each session's states
#     for dir_path, sess_states in zip(mouse_dir, states_list):
#         np.save(dir_path / "states.npy", sess_states)
    
#     # Concatenate *all* sessions into one long array
#     all_states = np.concatenate(states_list, axis=0)
#     np.save(output_path / "all_states.npy", all_states)

# else:
#     # Single-session case
#     # Step 1: Compute HMM states
#     states = hmm.most_likely_states(features)
#     np.save(mouse_dir_feature / f"states_K={K}_{param_string}.npy", states)

#     if session_path is None:
#         raise FileNotFoundError(f"No folder ending in 'behav' found for session: {mouse_id}")
    
#     blocks_with_transitions_path = session_path / "blocks_with_transitions.pkl"
#     blocks_with_transitions = pickle.load(open(blocks_with_transitions_path, 'rb'))

#     # Step 4: Load session CSV
#     shelter_cat_data = pd.read_csv(session_path / "session_behav_data.csv")

#     # Step 5: Find end frame from blocks
#     if block_id == 0:
#         for block in blocks_with_transitions:
#             if block.get('id') == block_id:
#                 end_frame = block.get('end')
#                 break
#         else:
#             raise ValueError(f"No block 0 coded for yet and id == {block_id} found")

#     # Step 6: Inject HMM state values into DataFrame
#     shelter_cat_data['states'] = np.nan
#     start_frame = (end_frame - len(states))
#     print('len(range(start_frame:end_frame))', len(range(start_frame, end_frame)))

#     if end_frame > len(shelter_cat_data):
#         print(f"Warning: end_frame ({end_frame}) exceeds DataFrame length ({len(shelter_cat_data)})")
#         # Adjust end_frame to DataFrame length
#         end_frame = len(shelter_cat_data)
#         start_frame = end_frame - len(states)
        
#         # Ensure start_frame is not negative
#         if start_frame < 0:
#             print("Warning: states array is longer than DataFrame, truncating states")
#             states = states[-end_frame:]  # Take the last end_frame elements
#             start_frame = 0

#     shelter_cat_data.iloc[start_frame:end_frame, shelter_cat_data.columns.get_loc('states')] = states

#     # Save the updated DataFrame
#     shelter_cat_data.to_csv(session_path / "session_behav_data.csv", index=False)
        



# if len(mouse_dir) > 1:
#     states = all_states

# unique_states, state_counts = np.unique(states, return_counts=True)

# # State durations
# state_durations = []
# state_labels = []

# for state, group in groupby(states):
#     length = len(list(group))
#     state_labels.append(state)
#     state_durations.append(length)

# state_labels = np.array(state_labels)
# state_durations = np.array(state_durations)

# # Empirical transition matrix
# num_states = len(unique_states)
# transitions = np.zeros((num_states, num_states), dtype=int)

# for (s1, s2) in zip(states[:-1], states[1:]):
#     transitions[s1, s2] += 1

# # Normalize to get probabilities
# trans_probs = transitions / (transitions.sum(axis=1, keepdims=True) + 1e-10)

# analysis_results = {}
# # Store results
# analysis_results['state_statistics'] = {
#     'unique_states': unique_states,
#     'state_counts': state_counts,
#     'state_durations': state_durations,
#     'state_duration_labels': state_labels,
#     'empirical_transitions': transitions,
#     'empirical_transition_probs': trans_probs
# }

# means, scales = hmm.observations.params       # both are (K, D)
# covs = scales**2                              # diagonal covariances

# # Get transition matrix
# transition_matrix = hmm.transitions.transition_matrix

# # Store in analysis results
# analysis_results['model_parameters'] = {
#     'means': means,
#     'covariances': covs,
#     'transition_matrix': transition_matrix,
#     'K': K,
#     'D': D
# }

# output_file = output_path / "hmm_analysis_results.pkl"
# with open(output_file, 'wb') as f:
#     pickle.dump(analysis_results, f)

