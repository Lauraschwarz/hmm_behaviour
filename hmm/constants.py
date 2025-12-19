import pathlib
import numpy as np

ARENA = { 'port1': ((400, 500), (400, 600), (500, 600), (500, 500)), 
            'port0': ((400, 0), (400, 100), (500, 100), (500, 0)), 
            'Arena': ((100, 310), (275, 10), (450, 10), (625, 10), (800, 310), (625, 620), (450, 620), (275, 620)),
            'Arena_inner': ((138.24, 310.0), (289.71, 40.29), (450.0, 60.0), (610.29, 40.29), (761.76, 310.0), (610.29, 589.71), (450.0, 570.0), (289.71, 589.71))
            }

ARENA_VERTICES = (
            (100, 310),
            (275, 10),
            (450, 10),
            (625, 10),
            (800, 310),
            (625, 620),
            (450, 620),
            (275, 620)
)
inner_vertices = ((138.24, 310.0),
                  (289.71, 40.29),
                  (450.0, 60.0),
                  (610.29, 40.29),
                  (761.76, 310.0),
                  (610.29, 589.71),
                  (450.0, 570.0),
                  (289.71, 589.71),
                  (289.71, 589.71))

def save_rois_to_npy(rois: dict, output_dir: str):
    """
    Save each ROI in the dictionary to a separate .npy file.

    Args:
        rois (dict): Dictionary where keys are ROI names and values are the ROI data (e.g., numpy arrays or lists).
        output_dir (str): Directory to save the .npy files.
    """
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for name, data in rois.items():
        np.save(output_path / f"{name}.npy", data)
        print(f"Saved {name} to {output_path / f'{name}.npy'}")

#save_rois_to_npy(ARENA, "F:/social_sniffing/derivatives/1125132/2025-07-01T13-25-31/Video/ObjectPaths")

TRIAL_LENGTH = 60  # seconds##

DEVICE = r'F:/social_sniffing'
OUTPUT_EXPANDER = r'F:/social_sniffing/output_expander'
STEPMOTOR = r'F:/social_sniffing/stepmotor'
