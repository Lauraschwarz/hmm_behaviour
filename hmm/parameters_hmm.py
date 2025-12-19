from pathlib import Path
import numpy as np
import pandas as pd


TRAININGDICT = {
    "1125563": "2025-09-08T13-06-32", 
    "1125131": "2025-07-01T12-47-28",
    "1125132": "2025-07-01T13-25-31",
    "1106077": "2025-09-23T12-31-23",
    "1106010": "2025-09-17T12-23-11",
    "1106079": "2025-09-23T13-06-24" 
   
    }

def get_session_list(TRAININGDICT):
    session_paths = []

    for mouse_id, session_id in TRAININGDICT.items():
        session_path = Path(r"F:\social_sniffing\derivatives") / mouse_id / session_id / "completed_distances_imputed.csv"
        session_paths.append(session_path)

    return session_paths

def concat_sessions(session_paths):
    all_sessions_data = []

    for session_path in session_paths:
        session_data = pd.read_csv(session_path)
        all_sessions_data.append(session_data)

    concatenated_data = pd.concat(all_sessions_data, ignore_index=True)
    output_path = Path(r"F:\social_sniffing") / "training_data_hmm" / "concatenated_sessions.csv"
    concatenated_data.to_csv(output_path, index=False)
    print(f"Saved concatenated sessions to {output_path}")

concat_sessions(get_session_list(TRAININGDICT))
