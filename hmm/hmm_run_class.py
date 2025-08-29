import numpy as np
import ssm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import matplotlib.colors as mcolors
from pathlib import Path
from itertools import groupby
from tqdm import tqdm
from params import hmm_parameters_object
from hmm_utils import get_param_string
from animations_utils import VideoAnnotator
import os
import xarray as xr
import cv2


class HMMProcessor:
    """
    A comprehensive Hidden Markov Model processor for behavioral data analysis.
    
    This class handles HMM model fitting, state prediction, analysis, visualization,
    and video annotation with optional components and progress tracking.
    """
    
    def __init__(self, parameters_object=None, show_plots=True, save_plots=True, 
                 create_animations=False, save_animations=True, fit_new_model=True):
        """
        Initialize the HMM Processor.
        
        Args:
            parameters_object: HMM parameters object (defaults to hmm_parameters_object)
            show_plots (bool): Whether to display plots
            save_plots (bool): Whether to save plots to disk
            create_animations (bool): Whether to create video animations
            save_animations (bool): Whether to save animations to disk
        """
        self.params = parameters_object or hmm_parameters_object
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.create_animations = create_animations
        self.save_animations = save_animations
        self.fit_new_model = fit_new_model
        
        # Set random seed for reproducibility
        if self.params.seed is not None:
            np.random.seed(self.params.seed)

        # Get total frame count of video
        self.total_video_frames = self.get_total_video_frames()
        # Initialize paths
        self._setup_paths()
        
        # Initialize model components
        self.hmm = None
        self.features = None
        self.states = None
        self.analysis_results = {}
    
    def get_total_video_frames(self):
        vid_path = os.path.join(hmm_parameters_object.root_path, 'rawdata', hmm_parameters_object.mouse_id, hmm_parameters_object.session_id, "Video", f"{hmm_parameters_object.mouse_id}.avi")

        print(f"Video path: {vid_path}")
        if os.path.exists(vid_path):
            cap = cv2.VideoCapture(vid_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"Total frames in video: {total_frames}")
            self.total_video_frames = total_frames

        else:
            print("Video file not found.")
            exit()
            self.total_video_frames = None
        return self.total_video_frames
        
    def load_existing_model(self):
        """Load existing HMM model and states from disk."""
        print("Loading existing HMM model and states...")
        
        # Check for existing model file
        model_file = self.mouse_dir_feature / f"hmm_model_K={self.params.k}_{self.param_string}.pkl"
        if model_file.exists():
            print(f"Loading HMM model from: {model_file}")
            with open(model_file, 'rb') as f:
                self.hmm = pickle.load(f)
        else:
            print(f"Warning: HMM model file not found at {model_file}")
            print("Model will be fitted on-demand if needed for cross-validation.")
            self.hmm = None
        
        # Load existing states
        states_file = self.mouse_dir_feature / f"states_K={self.params.k}_{self.param_string}.npy"
        if states_file.exists():
            print(f"Loading states from: {states_file}")
            self.states = np.load(states_file)
            print(f"Loaded {len(self.states)} states")
        else:
            print(f"Warning: States file not found at {states_file}")
            print("You may need to predict states first.")
            self.states = None
        
        # Load features if available (needed for some analyses and model fitting)
        features_path = self.home_dir / f"feature_array_{self.param_string}.npy"
        if features_path.exists():
            print(f"Loading features from: {features_path}")
            features_full = np.load(features_path).T
            # Apply same preprocessing as in load_data()
            features_full[~np.isfinite(features_full)] = np.nan
            features_interp = pd.DataFrame(features_full).interpolate(
                axis=0, limit_direction='both'
            ).to_numpy()
            self.features = features_interp[1500:]
            print(f"Loaded features with shape: {self.features.shape}")
        else:
            print(f"Warning: Features file not found at {features_path}")
            print("Some analyses may not be available without features.")
            self.features = None
        
        print("Finished loading existing model components.")
    
    def save_model(self):
        """Save the fitted HMM model to disk."""
        if self.hmm is None:
            print("Warning: No HMM model to save.")
            return
        
        model_file = self.mouse_dir_feature / f"hmm_model_K={self.params.k}_{self.param_string}.pkl"
        print(f"Saving HMM model to: {model_file}")
        
        with open(model_file, 'wb') as f:
            pickle.dump(self.hmm, f)
        
        print("HMM model saved successfully!")
        
    def _setup_paths(self):
        """Setup all necessary file paths."""
        self.hmm_base_outpath = Path(self.params.hmm_base_outpath)
        self.mouse_id = self.params.mouse_id
        self.root_path = Path(self.params.root_path)
        self.session_id = self.params.session_id
        self.output_path = self.hmm_base_outpath
        self.session_path = self.root_path / "derivatives" / self.mouse_id / self.session_id / "behav"
        
        # Create parameter string and directories
        self.param_string = get_param_string(self.params)
        self.home_dir = self.output_path / self.mouse_id / self.param_string
        self.home_dir.mkdir(parents=True, exist_ok=True)
        self.mouse_dir_feature = self.home_dir / f"K={self.params.k}"
        self.mouse_dir_feature.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load and preprocess feature data."""
        print("Loading and preprocessing data...")
        
        # Load features
        features_path = self.home_dir / f"feature_array_{self.param_string}.npy"
        features_full = np.load(features_path)
        print(f"Original feature shape: {features_full.shape}")
        
        # Handle NaN values
        features_full[~np.isfinite(features_full)] = np.nan
        features_interp = pd.DataFrame(features_full).interpolate(
            axis=0, limit_direction='both'
        ).to_numpy()
        
        # Truncate start of data
        self.features = features_interp[1550:]
        print(f"Processed feature shape: {self.features.shape}")
        
    def fit_hmm(self):
        """Fit the HMM model to the feature data."""
        if self.features is None:
            raise ValueError("Features not loaded. Call load_data() first.")
        
        print(f"Fitting HMM with K={self.params.k} states...")
        
        D = self.features.shape[1]
        self.hmm = ssm.HMM(
            K=self.params.k, 
            D=D, 
            observations="gaussian", 
            random_state=self.params.seed
        )
        
        # Fit with progress bar
        with tqdm(total=500, desc="HMM Training") as pbar:
            def callback(model, data, iteration):
                pbar.update(1)
                
            self.hmm.fit([self.features], method="em", num_iters=500)
            pbar.close()
        
        print("HMM fitting completed!")
        
        # Save the fitted model
        self.save_model()

    # def add_states_to_xarray(self, states_csv_path: Path, tracking_data_xr: xr.Dataset):
    #     """
    #     Adds a 'states' variable from a states CSV into an existing xarray Dataset.
        
    #     Parameters
    #     ----------
    #     states_csv_path : Path
    #         Path to the CSV file like 'states_{param_string}.csv'.
    #     tracking_data_xr : xr.Dataset
    #         The xarray Dataset to update.
    #     """
    #     # Load states CSV
    #     states_df = pd.read_csv(states_csv_path)
    #     states_df = states_df.copy()

    #     # Ensure 'state' column exists
    #     if 'state' not in states_df.columns:
    #         raise ValueError(f"'state' column not found in {states_csv_path}")

    #     # Get number of frames from an existing variable in the Dataset
    #     if 'position' in tracking_data_xr:
    #         num_frames = tracking_data_xr['position'].shape[0]
    #     else:
    #         # fallback: use the first variable
    #         first_var = list(tracking_data_xr.data_vars)[0]
    #         num_frames = tracking_data_xr[first_var].shape[0]

    #     # Check frame count matches
    #     if len(states_df) != num_frames:
    #         raise ValueError(
    #             f"Frame count mismatch: CSV has {len(states_df)}, "
    #             f"xarray has {num_frames}"
    #         )

    #     # Add states as a new variable
    #     tracking_data_xr['states'] = (('frame',), states_df['state'].values)

    #     # Save updated xarray
    #     states_csv_path = Path(states_csv_path)  # ensure it's a Path
    #     save_path = states_csv_path.with_name(
    #         states_csv_path.name.replace('states_', 'tracking_data_').replace('.csv', '.nc')
    #     )
    #     tracking_data_xr.to_netcdf(save_path)

    #     print(f"Updated xarray saved to: {save_path}")

    def add_states_to_xarray(self, states_csv_path: Path, tracking_data_xr: xr.Dataset):
        import pandas as pd
        from pathlib import Path

        states_df = pd.read_csv(states_csv_path).copy()

        if 'state' not in states_df.columns:
            raise ValueError(f"'state' column not found in {states_csv_path}")

        if 'position' not in tracking_data_xr:
            raise ValueError("No 'position' variable found in xarray Dataset")

        da = tracking_data_xr['position']

        # Use the length of the 'time' dimension
        time_len = da.sizes['time']
        if len(states_df) != time_len:
            raise ValueError(
                f"Frame count mismatch: CSV has {len(states_df)}, "
                f"'time' dimension has {time_len}"
            )

        # Attach 'states' along the 'time' dimension
        tracking_data_xr = tracking_data_xr.assign_coords(states=('time', states_df['state'].values))

        tracking_data_xr['position'] = da

        states_csv_path = Path(states_csv_path)
        save_path = states_csv_path.with_name(
            states_csv_path.name.replace('states_', 'tracking_data_with_states_').replace('.csv', '.nc')
        )
        if hasattr(tracking_data_xr, 'close'):
            tracking_data_xr.close()

        tracking_data_xr.to_netcdf(save_path)

        print(f"Updated xarray saved to: {save_path}")




        
    def predict_states(self):
        """Predict states using the fitted HMM."""
        if self.hmm is None:
            raise ValueError("HMM not fitted. Call fit_hmm() first.")
        
        print("Predicting states...")
        self.states = self.hmm.most_likely_states(self.features)
        
        # Save states
        states_file = self.mouse_dir_feature / f"states_K={self.params.k}_{self.param_string}.npy"
        np.save(states_file, self.states)
        print(f"States saved to: {states_file}")
        
    def update_session_data(self):
        """Update session behavioral data with HMM states and save separate states CSV."""
        if self.states is None:
            raise ValueError("States not predicted. Call predict_states() first. Could be that there is no model yet")
        
        print("Updating session data with HMM states...")

        # Load necessary data
        blocks_path = self.session_path / "blocks_with_transitions.pkl"
        blocks_with_transitions = pickle.load(open(blocks_path, 'rb'))
        
        shelter_cat_data = pd.read_csv(self.session_path / "session_behav_data.csv")
        
        # Find end frame for block 0
        if self.params.block_id == 0:
            for block in blocks_with_transitions:
                if block.get('id') == self.params.block_id:
                    end_frame = block.get('end')
                    break
            else:
                raise ValueError(f"Block {self.params.block_id} not found")
        
        # Inject HMM states into session DataFrame
        shelter_cat_data['states'] = np.nan
        start_frame = end_frame - len(self.states)
        
        # Handle edge cases
        if end_frame > len(shelter_cat_data):
            print(f"Warning: end_frame ({end_frame}) exceeds DataFrame length")
            end_frame = len(shelter_cat_data)
            start_frame = end_frame - len(self.states)
            
            if start_frame < 0:
                print("Warning: states array longer than DataFrame, truncating")
                self.states = self.states[-end_frame:]
                start_frame = 0
        
        # Update DataFrame
        shelter_cat_data.iloc[
            start_frame:end_frame, 
            shelter_cat_data.columns.get_loc('states')
        ] = self.states
        
        # Save updated full session CSV
        shelter_cat_data.to_csv(self.session_path / "session_behav_data.csv", index=False)
        print("Session data updated successfully!")

        # ---- NEW: Save states-only CSV for gradual build-up ----
        states_file = self.mouse_dir_feature / f"states_{self.param_string}.csv"

        self.total_video_frames = self.get_total_video_frames()
        print(f"Total video frames: {self.total_video_frames}")

        # Create full-length DataFrame with NaNs
        full_states_df = pd.DataFrame({
            "frame": range(self.total_video_frames),
            "state": np.nan
        })

        # Fill in the current portion with the states
        full_states_df.loc[start_frame:end_frame - 1, "state"] = self.states

        # Merge with existing CSV if it exists
        if states_file.exists():
            existing_df = pd.read_csv(states_file).set_index('frame')
            full_states_df.update(existing_df)
        
        # Save the combined CSV
        full_states_df.to_csv(states_file, index=False)
        print(f"States-only CSV updated: {states_file}")



    def load_other_features(self, other_features_path):
        """Load features from another dataset for cross-validation."""

        other_features_path = Path(other_features_path)
        other_features_path = other_features_path / f"feature_array_{self.param_string}.npy"
        print(f"Loading other features from: {other_features_path}")
        other_features = np.load(other_features_path).T
        
        # Handle NaN values
        other_features[~np.isfinite(other_features)] = np.nan
        other_features_interp = pd.DataFrame(other_features).interpolate(
            axis=0, limit_direction='both'
        ).to_numpy()

        self.other_features = other_features_interp[1500:] # just get rid of first 30 seconds of data just in case
        
        return self.other_features
    
    def cross_validate(self, other_features_path):
        """Cross-validate the HMM model on another dataset."""

        if self.hmm is None:
            print("Warning: No HMM model loaded. Attempting to fit model with current features...")
        if self.features is None:
            print("Loading features for model fitting...")
            self.load_data()
            self.fit_hmm()

        print("Cross-validating HMM on new features...")

        self.other_features = self.load_other_features(other_features_path)
        print(f"Other features shape: {self.other_features.shape}")

        # Predict states for the new features
        other_states = self.hmm.most_likely_states(self.other_features)

        # Save cross-validated states
        cross_val_file = self.mouse_dir_feature / f"cross_val_states_K={self.params.k}_{self.param_string}.npy"
        # np.save(cross_val_file, other_states)
        print(f"Cross-validated states saved to: {cross_val_file}")

        # ---- Log-likelihood Calculation ----
        log_likelihood = self.hmm.log_likelihood(self.other_features)
        log_likelihood_per_point = log_likelihood / self.other_features.shape[0]
        print(f"Log-likelihood of new features: {log_likelihood:.2f}")
        print(f"Log-likelihood per time point: {log_likelihood_per_point:.5f}")

        # Path to log file
        log_path = self.mouse_dir_feature / f"log_likelihood_summary_K={self.params.k}_{self.param_string}.csv"

        # Extra info columns
        trained_session = self.params.mouse_id
        comparison_session = Path(other_features_path).parent.name

        # Prepare row of info to write
        log_data = {
            "K": self.params.k,
            "param_string": self.param_string,
            "log_likelihood": log_likelihood,
            "log_likelihood_per_point": log_likelihood_per_point,
            "n_points": self.other_features.shape[0],
            "cross_val_file": str(cross_val_file),
            "trained_session": trained_session,
            "comparison_session": comparison_session
        }

        # If file exists, check for duplicates before writing
        if os.path.exists(log_path):
            df_existing = pd.read_csv(log_path)

            duplicate_exists = (
                (df_existing["K"] == log_data["K"]) &
                (df_existing["param_string"] == log_data["param_string"]) &
                (df_existing["trained_session"] == log_data["trained_session"]) &
                (df_existing["comparison_session"] == log_data["comparison_session"])
            ).any()

            if duplicate_exists:
                print("Matching row already exists. Skipping write.")
            else:
                pd.DataFrame([log_data]).to_csv(log_path, mode='a', header=False, index=False)
                print(f"New log-likelihood info appended to: {log_path}")
        else:
            pd.DataFrame([log_data]).to_csv(log_path, index=False)
            print(f"Log-likelihood info saved to: {log_path}")

        return other_states, log_likelihood

        
    def analyze_states(self):
        """Perform comprehensive state analysis."""
        if self.states is None:
            raise ValueError("States not predicted. Call predict_states() first.")
        
        print("Analyzing HMM states...")
        
        # Basic statistics
        unique_states, state_counts = np.unique(self.states, return_counts=True)
        
        # State durations
        state_durations = []
        state_labels = []
        for state, group in groupby(self.states):
            length = len(list(group))
            state_labels.append(state)
            state_durations.append(length)
        
        # Transition matrix
        num_states = len(unique_states)
        transitions = np.zeros((num_states, num_states), dtype=int)
        for s1, s2 in zip(self.states[:-1], self.states[1:]):
            transitions[s1, s2] += 1
        trans_probs = transitions / transitions.sum(axis=1, keepdims=True)
        
        # Store results
        self.analysis_results = {
            'unique_states': unique_states,
            'state_counts': state_counts,
            'state_durations': np.array(state_durations),
            'state_labels': np.array(state_labels),
            'transition_matrix': trans_probs,
            'num_states': num_states
        }
        
        print(f"Analysis completed. Found {num_states} unique states.")
        
    def plot_state_distribution(self):
        """Plot state distribution."""
        if not self.analysis_results:
            self.analyze_states()
        
        plt.figure(figsize=(10, 6))
        plt.bar(self.analysis_results['unique_states'], self.analysis_results['state_counts'])
        plt.xlabel('State')
        plt.ylabel('Count')
        plt.title('HMM State Distribution')
        
        if self.save_plots:
            save_path = self.mouse_dir_feature / f"state_distribution_K={self.params.k}_{self.param_string}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.show_plots:
            plt.show()
        else:
            plt.close()
            
    def plot_state_sequence(self):
        """Plot state sequence over time with progress bar."""
        if self.states is None:
            raise ValueError("States not predicted. Call predict_states() first.")
        
        print("Creating state sequence plot...")
        
        plt.figure(figsize=(15, 3))
        
        # Plot with progress bar for large sequences
        if len(self.states) > 10000:
            with tqdm(total=len(self.states), desc="Plotting sequence") as pbar:
                chunk_size = 1000
                for i in range(0, len(self.states), chunk_size):
                    end_idx = min(i + chunk_size, len(self.states))
                    plt.plot(
                        range(i, end_idx), 
                        self.states[i:end_idx], 
                        drawstyle='steps-post'
                    )
                    pbar.update(end_idx - i)
        else:
            plt.plot(self.states, drawstyle='steps-post')
        
        plt.xlabel('Time index')
        plt.ylabel('State')
        plt.title('HMM State Sequence Over Time')
        
        if hasattr(self, 'analysis_results') and self.analysis_results:
            plt.yticks(self.analysis_results['unique_states'])
        
        if self.save_plots:
            save_path = self.mouse_dir_feature / f"state_sequence_K={self.params.k}_{self.param_string}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.show_plots:
            plt.show()
        else:
            plt.close()
            
    def plot_dwell_times(self):
        """Plot state dwell time distributions."""
        if not self.analysis_results:
            self.analyze_states()
        
        plt.figure(figsize=(12, 6))
        
        for state in self.analysis_results['unique_states']:
            durations = self.analysis_results['state_durations'][
                self.analysis_results['state_labels'] == state
            ]
            plt.hist(durations, bins=30, alpha=0.7, label=f'State {state}', density=True)
        
        plt.xlabel('Dwell time (consecutive frames)')
        plt.ylabel('Density')
        plt.title('State Dwell Time Distributions')
        plt.legend()
        
        if self.save_plots:
            save_path = self.mouse_dir_feature / f"dwell_times_K={self.params.k}_{self.param_string}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.show_plots:
            plt.show()
        else:
            plt.close()
            
    def plot_transition_matrix(self):
        """Plot state transition matrix."""
        if not self.analysis_results:
            self.analyze_states()
        
        plt.figure(figsize=(8, 6))
        plt.imshow(
            self.analysis_results['transition_matrix'], 
            cmap='viridis', 
            interpolation='none'
        )
        plt.colorbar(label='Transition Probability')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.title('HMM State Transition Matrix')
        
        if self.save_plots:
            save_path = self.mouse_dir_feature / f"transition_matrix_K={self.params.k}_{self.param_string}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.show_plots:
            plt.show()
        else:
            plt.close()
            
    def plot_state_behavior_correlation(self, behavior_columns=None):
        """Plot correlation between states and behaviors."""
        if behavior_columns is None:
            behavior_columns = [
                'close_to_OpenShelter1', 'mouse_exploring_OpenShelter1', 
                'near_and_facing_OpenShelter1', 'inside_OpenShelter1', 
                'mouse_stationary', 'mouse_exploring_arena', 'at_arena_edge',
                'facing_wall'
            ]
        
        # Load behavioral data
        shelter_cat_data = pd.read_csv(self.session_path / "session_behav_data.csv")
        
        # Clean data
        df = shelter_cat_data.dropna(subset=['states']).copy()
        df['states'] = df['states'].astype(int)
        df[behavior_columns] = df[behavior_columns].apply(pd.to_numeric, errors='coerce')
        
        # One-hot encode states
        state_dummies = pd.get_dummies(df['states'], prefix='state')
        
        # Calculate correlations
        combined = pd.concat([df[behavior_columns], state_dummies], axis=1)
        correlation_matrix = combined.corr()
        state_behavior_corr = correlation_matrix.loc[state_dummies.columns, behavior_columns]
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            state_behavior_corr, 
            annot=True, 
            cmap='coolwarm', 
            center=0, 
            fmt=".2f", 
            linewidths=0.5
        )
        plt.title("Correlation: HMM States vs Behaviors")
        plt.xlabel("Behavior Columns")
        plt.ylabel("HMM States")
        
        if self.save_plots:
            save_path = self.mouse_dir_feature / f"state_behavior_correlation_K={self.params.k}_{self.param_string}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_statewise_feature_summary(self, feature_column='speed'):
        """
        Create a bar plot showing the mean and 95% confidence interval (CI) of a feature
        (e.g., speed) for each HMM state.
        """
        import scipy.stats as stats

        # Load behavior/session data (should include 'states' and the feature column)
        df = pd.read_csv(self.session_path / "session_behav_data.csv")

        # Drop rows where 'states' or feature are missing
        df = df.dropna(subset=['states', feature_column])
        df['states'] = df['states'].astype(int)

        # Get unique states
        unique_states = sorted(df['states'].unique())
        means = []
        cis = []

        for state in unique_states:
            state_values = df.loc[df['states'] == state, feature_column]
            mean_val = state_values.mean()
            std_val = state_values.std()
            n = len(state_values)
            ci_95 = 1.65 * (std_val / np.sqrt(n)) if n > 1 else 0  # 95% CI approximation
            means.append(mean_val)
            cis.append(ci_95)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            unique_states,
            means,
            yerr=cis,
            capsize=8,
            color='skyblue',
            edgecolor='black'
        )
        ax.set_xticks(unique_states)
        ax.set_xticklabels([f"State {s}" for s in unique_states])
        ax.set_ylabel(feature_column.capitalize())
        ax.set_xlabel("HMM State")
        ax.set_title(f"{feature_column.capitalize()} by HMM State\n(mean ± 95% CI)")
        plt.tight_layout()

        # Save
        output_path = self.mouse_dir_feature / f"statewise_{feature_column}_summary_K={self.params.k}_{self.param_string}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

        if self.show_plots:
            plt.show()
        else:
            plt.close()
            
    def plot_states_with_behaviors(self, behavior_columns=None, start_idx=None, end_idx=None):
        """Plot states with superimposed behavior annotations."""
        if behavior_columns is None:
            behavior_columns = [
                'close_to_OpenShelter1', 'mouse_exploring_OpenShelter1', 
                'near_and_facing_OpenShelter1', 'inside_OpenShelter1', 
                'mouse_stationary', 'at_arena_edge', 'mouse_exploring_arena', 
                'facing_wall'
            ]
        
        # Load data
        shelter_cat_data = pd.read_csv(self.session_path / "session_behav_data.csv")
        
        # Auto-determine range if not provided
        if start_idx is None or end_idx is None:
            valid_mask = ~shelter_cat_data['states'].isna()
            start_idx = valid_mask[valid_mask].index[0]
            end_idx = valid_mask[valid_mask].index[-1] + 1
        
        # Slice data
        data = shelter_cat_data.iloc[start_idx:end_idx].copy()
        time = np.arange(start_idx, end_idx)
        
        # Prepare states
        states = data['states'].fillna(-1).astype(int)
        unique_states = np.unique(states[states >= 0])
        cmap = plt.get_cmap('tab10' if len(unique_states) <= 10 else 'tab20')
        norm = mcolors.Normalize(vmin=unique_states.min(), vmax=unique_states.max())
        
        # Create plot
        n_behaviors = len(behavior_columns)
        fig, ax = plt.subplots(figsize=(15, 0.7 * n_behaviors + 1.5))
        
        # Plot state background with progress bar for large datasets
        print("Plotting states with behaviors...")
        if len(time) > 5000:
            with tqdm(total=len(unique_states), desc="Plotting states") as pbar:
                for val in unique_states:
                    mask = (states == val)
                    ax.bar(
                        time[mask], height=1, width=1, bottom=0,
                        color=cmap(norm(val)), align='edge', label=f'State {val}'
                    )
                    pbar.update(1)
        else:
            for val in unique_states:
                mask = (states == val)
                ax.bar(
                    time[mask], height=1, width=1, bottom=0,
                    color=cmap(norm(val)), align='edge', label=f'State {val}'
                )
        
        # Plot behavior overlays
        y_offsets = np.linspace(0.05, 0.9, n_behaviors)
        for i, col in enumerate(behavior_columns):
            behavior = data[col].fillna(0).astype(int)
            mask = (behavior == 1)
            ax.bar(
                time[mask], height=0.08, width=1, bottom=y_offsets[i],
                color='black', alpha=0.8, align='edge'
            )
        
        # Format plot
        ax.set_yticks(y_offsets)
        ax.set_yticklabels(behavior_columns)
        ax.set_xlim(start_idx, end_idx)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Frame')
        ax.set_title('HMM States with Superimposed Behaviors')
        
        # Legend
        handles, labels = ax.get_legend_handles_labels()
        state_handles = handles[-len(unique_states):]
        state_labels = labels[-len(unique_states):]
        ax.legend(
            state_handles, state_labels, 
            bbox_to_anchor=(1.01, 1), loc='upper left', title='States'
        )
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = self.mouse_dir_feature / f"states_with_behaviors_K={self.params.k}_{self.param_string}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def create_video_annotations(self, start_frame=None, end_frame=None):
        """Create video annotations with behavioral overlays."""
        if not self.create_animations:
            print("Animation creation disabled. Skipping video annotations.")
            return
        
        print("Creating video annotations...")
        
        # Setup paths
        shelter_cat_data_path = self.session_path / "session_behav_data.csv"
        video_path = Path(str(shelter_cat_data_path.parent).replace("derivatives", "rawdata"))
        video_path = video_path / "cam.avi"
        
        video_dir = video_path.parent
        derivatives_path = Path(*[
            ("derivatives" if part == "rawdata" else part) 
            for part in video_dir.parts
        ])
        
        list_of_blocks_data_path = derivatives_path / 'list_of_blocks_data.pkl'
        
        # Initialize annotator
        annotator = VideoAnnotator()
        
        # Create behavioral video clip
        output_path = self.mouse_dir_feature / f"behavior_annotated_K={self.params.k}_{self.param_string}.mp4"
        
        annotator.create_behavioral_video_clip(
            csv_path=shelter_cat_data_path,
            video_path=video_path,
            output_path=output_path if self.save_animations else None,
            start_frame=start_frame,
            end_frame=end_frame,
            list_of_blocks_data_path=list_of_blocks_data_path
        )
        
        print(f"Video annotation completed: {output_path}")
        
    def run_full_analysis(self, behavior_columns=None, start_frame=None, end_frame=None):
        """Run the complete HMM analysis pipeline."""
        print("Starting full HMM analysis pipeline...")
        
        try:
            # Core processing
            self.load_data()
            
            if self.fit_new_model:
                self.fit_hmm()
                self.predict_states()
            else:
                self.load_existing_model()
            
            self.update_session_data()
            
            # Analysis and visualization
            if self.show_plots or self.save_plots:
                print("Creating visualizations...")
                self.plot_statewise_feature_summary(feature_column='speed')
                self.plot_statewise_feature_summary(feature_column='acceleration')
                self.plot_statewise_feature_summary(feature_column='body_length')
                self.plot_state_distribution()
                self.plot_state_sequence()
                self.plot_dwell_times()
                self.plot_transition_matrix()
                self.plot_state_behavior_correlation(behavior_columns)
                # self.plot_states_with_behaviors(behavior_columns)
            
            # Video annotations
            if self.create_animations:
                self.create_video_annotations(start_frame=start_frame, end_frame=end_frame)


            
        except Exception as e:
            print(f"Error in analysis pipeline: {e}")
            raise

        return self.param_string
            
    def get_results_summary(self):
        """Get a summary of analysis results."""
        if not self.analysis_results:
            return "No analysis results available. Run analyze_states() first."
        
        summary = f"""
HMM Analysis Results Summary:
============================
Number of states: {self.analysis_results['num_states']}
Total time points: {len(self.states) if self.states is not None else 'N/A'}
State distribution: {dict(zip(self.analysis_results['unique_states'], self.analysis_results['state_counts']))}
Average dwell time: {np.mean(self.analysis_results['state_durations']):.2f} frames
Files saved to: {self.mouse_dir_feature}
        """
        return summary