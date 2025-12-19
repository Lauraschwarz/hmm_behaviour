from dataclasses import dataclass
from params_class import hmm_parameters

#mouse_id: can pass a list of all the subjects.
#session_id : can pass a list of all sessions
 
hmm_parameters_object = hmm_parameters(  # Mouse specific
            mouse_id= "1106077",#["1125132", "1125131","1125561", "1125563"], #"sub-005_id-1122877_type-wtshelterswitch", # "sub-006_id-1123131_type-wtshelterswitch",
            session_id= "2025-09-23T12-31-23",#"concatenated_sessions", # can also pass a list of sessions for each mouse, e.g. ["2025-07-01T13-25-31", "2025-07-02T13-43-32"], #"2025-07-02T14-37-53", "2025-07-03T14-32-28"]
            session_mouse_dict =[],# {"1125132": ["2025-07-01T13-25-31"],
                                  #"1125131": ["2025-07-02T13-43-32"],
                                  #"1125561": ["2025-09-09T12-24-47"],
                                  #"1125563": ["2025-09-08T13-06-32"]
                                  #}, #"2025-07-02T14-37-53", "2025-07-03T14-32-28"]
            root_path=r"F:\social_sniffing",
            hmm_base_outpath=r"F:\social_sniffing\behaviour_model\hmm\output",
            k=5,
            N_features=5,  # Number of features to extract
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