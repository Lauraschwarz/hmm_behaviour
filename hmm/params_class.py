from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=False)
class hmm_parameters:
    """A class to store mouse by mouse information"""

    mouse_id: str
    session_id: str
    root_path: str
    session_mouse_dict: dict  # Mapping of mouse IDs to their session IDs
    hmm_base_outpath: str


    k: int  # Number of hidden states
    N_features: int
    block_id: int  # Block ID for the mouse
    seed: int

    # Features to include in the HMM
    # These are boolean flags indicating whether to include each feature
    pairwise: bool
    speed: bool
    acceleration: bool
    smoothed_speed: bool
    smoothed_acceleration: bool
    abdomen_abdomen: bool
    snout_groin: bool
    abdomen_port0: bool
    abdomen_port1: bool
