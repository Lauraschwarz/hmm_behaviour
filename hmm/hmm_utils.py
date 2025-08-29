import pickle
import numpy as np


def load_pickle_file(filepath):
    """
    Load a pickle file from the given filepath.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        object: The Python object loaded from the pickle file.
    """
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def extract_frame_range(ds=None, pairwise_distances_df=None, behaviour_data=None, start_frame=None, end_frame=None, block_transition_path=None, blocks=None):
    """
    Extracts a specific frame range from both:
      - an xarray.Dataset containing position data
      - a pandas.DataFrame containing distance data

    If `block_transition_path` is not None, it overrides `start_frame`
    with the first frame where the keypoint "Body Centre" is not NaN.

    Parameters:
        ds (xarray.Dataset): The pose dataset with `position` as a variable.
        distances_df (pd.DataFrame): A DataFrame with rows indexed by frame or time.
        block_transition_path (str or None): Path to a block transition file, or None.
        start_frame (int): The starting frame index (inclusive).
        end_frame (int): The ending frame index (exclusive).

    Returns:
        tuple: (position_subset, distances_subset)
    """
    if block_transition_path is not None:
        # get the start frame
        # Get the "Body Centre" keypoint index
        if blocks == 0:
            if "Body Centre" in ds.keypoints.values:
                keypoint_idx = int(list(ds.keypoints.values).index("Body Centre"))
                # Get x and y values for that keypoint
                body_centre_x = ds.position[:, 0, keypoint_idx, 0].values
                body_centre_y = ds.position[:, 1, keypoint_idx, 0].values

                # Find the first frame where both x and y are not NaN
                for i in range(len(body_centre_x)):
                    if not (np.isnan(body_centre_x[i]) or np.isnan(body_centre_y[i])):
                        start_frame = i + 40
                        break
                
                print('First frame with tracking is here:', start_frame)

            else:
                raise ValueError('"Body Centre" keypoint not found in dataset. Need this to estimate the first frame when the mouse enters the arena.')
        else:
            print('need to write code for when the start frame is from the block start')
        
        # Now get the end frame
        # load block transitions
        with open(block_transition_path, 'rb') as file:
            block_transitions = pickle.load(file) 

        print(block_transitions[0])
        for i, data in enumerate(block_transitions):
            block_id = data['id']
            if block_id == blocks:
                end_frame = data['end']
                print('End frame for block {} is: {}'.format(block_id, end_frame))
                break

    elif start_frame is None and end_frame is None:
        raise ValueError("Either `block_transition_path` must be provided or both `start_frame` and `end_frame` must be specified.")

    print('start_frame', start_frame)
    print('end_frame', end_frame)

    position_subset = ds.position.isel(time=slice(start_frame, end_frame))
    velocity = ds.velocity.isel(time=slice(start_frame, end_frame))
    accel = ds.acceleration.isel(time=slice(start_frame, end_frame))
    speed = ds.speed.isel(time=slice(start_frame, end_frame))
    head_direction = ds.head_direction.isel(time=slice(start_frame, end_frame))

    # Extract distances data by row index
    distances_subset = pairwise_distances_df.iloc[start_frame:end_frame]

    behaviour_data_subset = behaviour_data[start_frame:end_frame]

    return position_subset, distances_subset, behaviour_data_subset, velocity, accel, speed, head_direction


def get_param_string(hmm_parameters_object):
    """
    Generate a parameter string based on the attributes of the hmm_parameters_object.

    Args:
        hmm_parameters_object (hmm_parameters): An instance of the hmm_parameters class.

    Returns:
        str: A formatted string containing the parameters.
    """
    block_id = hmm_parameters_object.block_id
    param_flags = [      
        (f"N={hmm_parameters_object.N_features}", hmm_parameters_object.N_features),  
        ("pairwise", hmm_parameters_object.pairwise),
        ("speed", hmm_parameters_object.smoothed_speed),
        ("accel", hmm_parameters_object.smoothed_acceleration),
        ("discon", hmm_parameters_object.abdomen_abdomen),
        ("disbum", hmm_parameters_object.snout_groin),
        ("disport0", hmm_parameters_object.abdomen_port0),
        ("disport1", hmm_parameters_object.abdomen_port1),
    
    ]
    param_parts = [name for name, flag in param_flags if flag]

    param_string = f"block{block_id}_" + "_".join(param_parts) if param_parts else f"block{block_id}"
    return param_string


import numpy as np
from scipy.interpolate import interp1d

def interpolate_circular_nan(arr, kind="linear"):
    """
    Interpolate missing values (NaNs) in circular data like angles (wrapped at 2π).

    Parameters
    ----------
    arr : array-like
        Input angles in radians, possibly with NaNs. Can be 1D or 2D (single column).
    kind : str
        Interpolation type for scipy.interpolate.interp1d.

    Returns
    -------
    arr_interp : np.ndarray
        Interpolated array with values wrapped back to [0, 2π).
    """
    arr = np.asarray(arr).squeeze()  # ensure 1D
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D input after squeeze, got shape {arr.shape}")

    unwrapped = np.unwrap(arr)
    idx = np.arange(len(arr))
    mask = ~np.isnan(unwrapped)

    if mask.sum() < 2:
        # Not enough points to interpolate
        return np.mod(unwrapped, 2*np.pi)

    interp_fn = interp1d(idx[mask], unwrapped[mask], kind=kind, fill_value="extrapolate")
    arr_interp = np.mod(interp_fn(idx), 2*np.pi)
    return arr_interp
