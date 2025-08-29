from dataclasses import dataclass
from params_class import hmm_parameters

 
hmm_parameters_object = hmm_parameters(  # Mouse specific
            mouse_id= "1125132", #"sub-005_id-1122877_type-wtshelterswitch", # "sub-006_id-1123131_type-wtshelterswitch",
            session_id= "2025-07-02T14-37-53", #"ses-001_type-shelterswitch_condition-wtcontrol_datetime-20250412T122632", #"ses-001_type-shelterswitch_condition-wtcontrol_datetime-20250412T133948", 
            root_path=r"F:\social_sniffing",
            hmm_base_outpath=r"F:\social_sniffing\behaviour_model\hmm\output",
            k=5,
            N_features=6,  # Number of features to extract
            block_id=0,
            seed=42,
            pairwise=True,
            speed=True,
            acceleration=True,
            smoothed_speed=True,
            smoothed_acceleration=True,
            abdomen_abdomen=True,
            snout_groin=True,
            abdomen_port0=True,
            abdomen_port1=True
)   