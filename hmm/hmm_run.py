from hmm_run_class import HMMProcessor
import os
import xarray as xr
from hmm_utils import get_param_string
from params import hmm_parameters_object
from pathlib import Path

def run_hmm_analysis(hmm_parameters_object, start_frame=0, end_frame=36000, K_range=None):
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
    params = get_param_string(hmm_parameters_object)
    Ks_to_run = [hmm_parameters_object.k]  # Default: single run
    if K_range is not None:
        Ks_to_run = list(range(K_range[0], K_range[1] + 1))

    #check if there is a concat_features file folder
    path = Path(hmm_parameters_object.hmm_base_outpath)
    concat_features_path = path / "concat_training_features" / params
    if concat_features_path.exists():
        print(f"Found concatenated features at {concat_features_path}, loading...")
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
                create_animations=False, 
                save_animations=False, 
                fit_new_model=False  
            )

            # Run full analysis
            param_string = processor.run_full_analysis(start_frame=start_frame, end_frame=end_frame)

            # Ensure param_string is set correctly
            param_string = get_param_string(hmm_parameters_object)

            # Cross-validation
            if mouse_id == "1125132": 
                alt_id = "1125132"
            else:
                alt_id = mouse_id

            # other_dataset_path = fr"F:\social_sniffing\behaviour_model\hmm\output\{alt_id}\{param_string}"
            # other_states_cross_val, log_likelihood = processor.cross_validate(other_dataset_path)
            #print(f"Cross-validation log likelihood for K={K}: {log_likelihood}") 

        

    print("All runs completed.")

# --- USAGE ---

# Run for a single K
# run_hmm_analysis(hmm_parameters_object)

run_hmm_analysis(hmm_parameters_object , K_range=(6,7))
exit()



















