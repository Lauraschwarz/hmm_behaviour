import numpy as np
import ssm
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
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
import tempfile



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
        matplotlib.use("Agg")
        plt.ioff()  # Turn off interactive mode
        self.params = parameters_object or hmm_parameters_object
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.create_animations = create_animations
        self.save_animations = save_animations
        self.fit_new_model = fit_new_model
        
        # Set random seed for reproducibility
        if self.params.seed is not None:
            np.random.seed(self.params.seed)

        # Initialize paths
        self._setup_paths()

        # Get total frame count of video
        self.total_video_frames = self.get_total_video_frames()
        
        # Initialize model components
        self.hmm = None
        self.features = None
        self.states = None
        self.analysis_results = {}
    
    def get_total_video_frames(self):
   
        if hmm_parameters_object.session_id == "concatenated_sessions":
            # If concatenated then go to the very last entry and find the end
            block_transitions_path = self.session_path / "blocks_with_transitions.pkl"
            with open(block_transitions_path, 'rb') as f:
                block_transitions = pickle.load(f)
            self.total_video_frames = block_transitions[-1][-1]['end_frame']
        
        else:
            vid_path = os.path.join(hmm_parameters_object.root_path, 'derivatives', hmm_parameters_object.mouse_id, hmm_parameters_object.session_id, 'hmm_features.pkl')

            print(f"Video path: {vid_path}")
            if os.path.exists(vid_path):
                with open(vid_path, 'rb') as f:
                    features = pickle.load(f)
                self.total_video_frames = features.shape[0]

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
            features_full = np.load(features_path) ##if this crashes add .T back in 
            # Apply same preprocessing as in load_data()
            features_full[~np.isfinite(features_full)] = np.nan
            features_interp = pd.DataFrame(features_full).interpolate(
                axis=0, limit_direction='both'
            ).to_numpy()
            self.features = features_interp
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
        if self.session_id == "concatenated_sessions":
            self.session_path = self.output_path / 'concat_training_features' / get_param_string(self.params)
            self.param_string = get_param_string(self.params)
            self.home_dir = self.session_path
            self.home_dir.mkdir(parents=True, exist_ok=True)
            self.mouse_dir_feature = self.home_dir / f"K={self.params.k}"   
            self.mouse_dir_feature.mkdir(parents=True, exist_ok=True)
            self.tracking_data_xr_path = self.session_path / "tracking_data.nc"
            self.states_csv_path = self.home_dir / f"K={self.params.k}" / f"states_{self.param_string}.csv"
        else:
            self.session_path = self.root_path / "derivatives" / self.mouse_id / self.session_id 
            

            # Create parameter string and directories
            self.param_string = get_param_string(self.params)
            self.home_dir = self.output_path / self.mouse_id / self.param_string
            self.home_dir.mkdir(parents=True, exist_ok=True)
            self.mouse_dir_feature = self.home_dir / f"K={self.params.k}"
            self.mouse_dir_feature.mkdir(parents=True, exist_ok=True)

            self.tracking_data_xr_path = self.session_path / "tracking_data.nc" 
            self.states_csv_path = self.home_dir / f"K={self.params.k}" / f"states_{self.param_string}.csv" 

        
    def load_data(self):
        """Load and preprocess feature data."""
        print("Loading and preprocessing data...")

        # Add states to tracking data
        with xr.open_dataset(self.tracking_data_xr_path) as self.tracking_data_xr:
            self.tracking_data_xr = self.tracking_data_xr.load()
        
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
        self.features = features_interp
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


    def add_states_to_xarray(self, states_csv_path: Path, tracking_data_xr: xr.Dataset):
      
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

        print('self.session_path:', self.session_path)

        # Load necessary data
        blocks_path = self.session_path / "blocks_with_transitions.pkl"
        blocks_with_transitions = pickle.load(open(blocks_path, 'rb'))
        
        shelter_cat_data = pd.read_csv(self.session_path / "hmm_features.csv")
        frame_offset = 0
        if "concat_training_features" in str(blocks_path):
            print("Found 'concatenated_sessions' in the path")
            # If concatenated then go to the very last entry and find the end
            for session in blocks_with_transitions:
                max_end = max(block.get('end_frame') for block in session)
                print('end_frame:', max_end)
                frame_offset += max_end
            end_frame = frame_offset
            print('Total end_frame for concatenated sessions:', end_frame)

        # Find end frame for block 0
        elif self.params.block_id == 0:
            for block in blocks_with_transitions:
                if block.get('id') == self.params.block_id:
                    end_frame = blocks_with_transitions[1]['end_frame']
                    break
            else:
                raise ValueError(f"Block {self.params.block_id} not found")
        

        
        # Inject HMM states into session DataFrame
        shelter_cat_data['states'] = np.nan
        start_frame = 0
        
        # Handle edge cases
        if end_frame > len(shelter_cat_data):
            print(f"Warning: end_frame ({end_frame}) exceeds DataFrame length")
            print('len(shelter_cat_data):', len(shelter_cat_data))
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
            "frame": range(end_frame),
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

    def compute_log_likelihood(self, test_data):
        """Compute the log likelihood of the current model."""
        obs = np.array(test_data)
        self.test_lls = self.model.log_likelihood(obs)
        

    def load_other_features(self, other_features_path):
        """Load features from another dataset for cross-validation."""

        other_features_path = Path(other_features_path)
        other_features_path = other_features_path / f"feature_array_{self.param_string}.npy"
        print(f"Loading other features from: {other_features_path}")
        other_features = np.load(other_features_path)
        
        # Handle NaN values
        other_features[~np.isfinite(other_features)] = np.nan
        other_features_interp = pd.DataFrame(other_features).interpolate(
            axis=0, limit_direction='both'
        ).to_numpy()

        self.other_features = other_features_interp # just get rid of first 30 seconds of data just in case
        
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
    
    def plot_statewise_feature_summary(self, feature_column='smoothed_speed'):
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
        output_path = self.mouse_dir_feature / f"statewise_{feature_column}_K={self.params.k}_{self.param_string}.png"
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
    


    def create_state_summary_videos(self, min_duration=30, max_clips=12, grid_shape=(3,4), 
                                           videos_per_state=3, sampling_method='behavioral_clustering'):
        """
        Create summary videos for each HMM state with flexible sampling methods.

        For behavioral_clustering mode, creates separate videos for behavioral sub-types:
        E.g., State 3 might produce:
        - state_3_cluster_0_high_speed_center.mp4
        - state_3_cluster_1_slow_wall_following.mp4
        - state_3_cluster_2_medium_speed_exploration.mp4
        
        For other sampling methods, creates single summary video per state:
        - state_3_summary_temporal.mp4
        - state_3_summary_behavioral.mp4  
        - state_3_summary_location.mp4

        Parameters
        ----------
        min_duration : int
            Minimum number of consecutive frames for a block to count.
        max_clips : int
            Maximum clips to extract per cluster video (or per state for summary methods).
        grid_shape : tuple
            Shape of the combined grid video (rows, cols).
        videos_per_state : int
            Number of behavioral cluster videos to create per state (only used for 'behavioral_clustering').
        sampling_method : str
            Method for sampling blocks: 
            - 'behavioral_clustering': Create multiple cluster videos per state with descriptive names
            - 'behavioral': Create single video per state using behavioral diversity sampling
            - 'temporal': Create single video per state using temporal sampling
            - 'location': Create single video per state using spatial sampling
            - 'first': Create single video per state using first N blocks
        """
        hmm_base_outpath = Path(self.params.hmm_base_outpath)
        mouse_id = self.params.mouse_id
        root_path = Path(self.params.root_path)
        session_id = self.params.session_id
        output_path = hmm_base_outpath
        session_path = root_path / "derivatives" / mouse_id / session_id / "Video"
        rawdata_path = root_path / "rawdata" / mouse_id / session_id / "Video"

        # Create parameter string and directories
        param_string = get_param_string(self.params)
        home_dir = output_path / mouse_id / param_string
        mouse_dir_feature = home_dir / f"K={self.params.k}"
        xarray_path = mouse_dir_feature / f"tracking_data_with_states_{param_string}.nc"
        df_csv_path = session_path / "session_behav_data.csv"

        df_csv = pd.read_csv(df_csv_path)

        ds = xr.load_dataset(xarray_path)
        states = ds["states"].values  # shape (T,)
        unique_states = [s for s in np.unique(states) if not np.isnan(s)]

        temp_dir = Path(tempfile.mkdtemp())
        
        # Create appropriate output directory based on sampling method
        if sampling_method == 'behavioral_clustering':
            output_dir = mouse_dir_feature / "behavioral_clusters"
            output_dir.mkdir(exist_ok=True)
            print(f"Creating behavioral cluster videos with {videos_per_state} clusters per state")
            
            # Initialize metadata for behavioral clustering
            clustering_metadata = {
                'session_info': {
                    'mouse_id': mouse_id,
                    'session_id': session_id,
                    'param_string': param_string,
                    'k_states': self.params.k,
                    'total_unique_states': len(unique_states)
                },
                'clustering_parameters': {
                    'min_duration': min_duration,
                    'max_clips': max_clips,
                    'videos_per_state': videos_per_state,
                    'grid_shape': grid_shape
                },
                'states': {}
            }
        else:
            output_dir = mouse_dir_feature
            print(f"Creating summary videos using {sampling_method} sampling")

        for state in unique_states:
            print(f"\n=== Processing State {state} ===")
            
            # ---- 1. Find contiguous blocks of this state ----
            mask = (states == state).astype(int)
            starts, ends = [], []
            in_block = False
            for i in range(len(mask)):
                if mask[i] == 1 and not in_block:
                    in_block = True
                    start = i
                elif (mask[i] == 0 or i == len(mask)-1) and in_block:
                    end = i if mask[i] == 0 else i+1
                    if end - start >= min_duration:
                        starts.append(start)
                        ends.append(end)
                    in_block = False
            
            blocks = list(zip(starts, ends))
            if not blocks:
                print(f"No blocks found for state {state}")
                continue

            # ---- 2. Process based on sampling method ----
            if sampling_method == 'behavioral_clustering':
                self._create_behavioral_cluster_videos_for_state(
                    ds, df_csv, state, blocks, videos_per_state, max_clips, grid_shape, 
                    output_dir, temp_dir, rawdata_path, min_duration
                )
            else:
                self._create_summary_video_for_state(
                    ds, state, blocks, max_clips, grid_shape, sampling_method,
                    output_dir, temp_dir, rawdata_path
                )

        print(f"\nAll videos saved in {output_dir}")


    def _create_behavioral_cluster_videos_for_state(self, ds, df_csv, state, blocks, videos_per_state, 
                                                max_clips, grid_shape, output_dir, temp_dir, rawdata_path, min_duration):
        """Create multiple cluster videos for a single state using behavioral clustering."""
        
        state_metadata = {
            'total_blocks_found': len(blocks),
            'min_duration_threshold': min_duration,
            'requested_videos_per_state': videos_per_state,
            'warnings': [],
            'clustering_info': {},
            'videos_created': []
        }
        
        if len(blocks) < videos_per_state:
            warning = f"Only {len(blocks)} blocks found for state {state}, reducing videos_per_state to {len(blocks)}"
            print(warning)
            state_metadata['warnings'].append(warning)
            videos_per_state_adjusted = len(blocks)
        else:
            videos_per_state_adjusted = videos_per_state
        
        state_metadata['actual_videos_per_state'] = videos_per_state_adjusted

        # ---- Calculate behavioral features for all blocks ----
        block_features, valid_blocks, feature_names = self.calculate_block_features(ds, df_csv, blocks)

        state_metadata['valid_blocks_after_feature_calc'] = len(valid_blocks)
        
        if len(valid_blocks) < videos_per_state_adjusted:
            warning = f"Only {len(valid_blocks)} valid blocks for state {state}, creating {len(valid_blocks)} videos"
            print(warning)
            state_metadata['warnings'].append(warning)
            videos_per_state_adjusted = len(valid_blocks)
            state_metadata['final_videos_per_state'] = videos_per_state_adjusted

        # ---- Cluster blocks into behavioral sub-types ----
        cluster_assignments, cluster_names, clustering_info = self.cluster_behavioral_blocks(
            block_features, feature_names, videos_per_state_adjusted, state
        )
        
        # Store clustering information in metadata
        state_metadata['clustering_info'] = clustering_info

        # ---- Create video for each behavioral cluster ----
        annotator = VideoAnnotator()
        video_path = rawdata_path / 'cam.avi'

        for cluster_id in range(videos_per_state_adjusted):
            cluster_mask = cluster_assignments == cluster_id
            cluster_blocks = [valid_blocks[i] for i in range(len(valid_blocks)) if cluster_mask[i]]
            
            if not cluster_blocks:
                continue
                
            print(f"  Cluster {cluster_id} ({cluster_names[cluster_id]}): {len(cluster_blocks)} blocks")
            
            # Sample up to max_clips from this cluster
            original_cluster_size = len(cluster_blocks)
            if len(cluster_blocks) > max_clips:
                # Sample evenly across the cluster
                indices = np.linspace(0, len(cluster_blocks)-1, max_clips, dtype=int)
                cluster_blocks = [cluster_blocks[i] for i in indices]
            
            # ---- Generate individual clips for this cluster ----
            cluster_temp_dir = temp_dir / f"state_{state}_cluster_{cluster_id}"
            cluster_temp_dir.mkdir(exist_ok=True)
            
            clip_paths = []
            for j, (s, e) in enumerate(cluster_blocks):
                out_path = cluster_temp_dir / f"clip_{j}.mp4"

                annotator.create_keypoint_video_clip(
                    dataset=ds, video_path=video_path, output_path=out_path,
                    start_frame=s, end_frame=e
                )

                if out_path.exists():
                    clip_paths.append(out_path)
                else:
                    print(f"    Failed to create clip {j} for cluster {cluster_id}")
            
            if not clip_paths:
                print(f"    No valid clips created for cluster {cluster_id}, skipping")
                continue

            # ---- Combine clips into cluster grid video ----
            cluster_name_clean = cluster_names[cluster_id].replace(" ", "_").lower()
            final_out_path = output_dir / f"state_{state}_cluster_{cluster_id}_{cluster_name_clean}.mp4"
            
            self.combine_clips_grid(clip_paths, final_out_path, grid_shape, max_clips)
            print(f"    Created: {final_out_path.name}")
            
            # Store video info in metadata
            video_info = {
                'cluster_id': cluster_id,
                'cluster_name': cluster_names[cluster_id],
                'filename': final_out_path.name,
                'total_blocks_in_cluster': original_cluster_size,
                'clips_used_in_video': len(clip_paths),
                'max_clips_limit': max_clips
            }
            state_metadata['videos_created'].append(video_info)
        
        return state_metadata


    def _create_summary_video_for_state(self, ds, state, blocks, max_clips, grid_shape, 
                                    sampling_method, output_dir, temp_dir, rawdata_path):
        """Create single summary video for a state using specified sampling method."""
        
        # ---- Sample blocks using specified method ----
        if len(blocks) > max_clips:
            if sampling_method == 'behavioral':
                blocks = self.sample_by_behavioral_features(ds, blocks, state, max_clips)
            elif sampling_method == 'temporal':
                blocks = self.sample_by_time(blocks, max_clips)
            elif sampling_method == 'location':
                blocks = self.sample_by_location(ds, blocks, state, max_clips)
            else:  # 'first' or default
                blocks = blocks[:max_clips]
        
        print(f"Selected {len(blocks)} blocks for state {state} using {sampling_method} sampling")

        # ---- Generate individual clips ----
        annotator = VideoAnnotator()
        video_path = rawdata_path / 'cam.avi'
        clip_paths = []
        
        for j, (s, e) in enumerate(blocks):
            out_path = temp_dir / f"state{state}_clip{j}.mp4"

            annotator.create_keypoint_video_clip(
                dataset=ds, video_path=video_path, output_path=out_path,
                start_frame=s, end_frame=e
            )

            if out_path.exists():
                clip_paths.append(out_path)
            else:
                print(f"Failed to create clip {j} for state {state}")
        
        if not clip_paths:
            print(f"No valid clips created for state {state}, skipping grid video")
            return

        # ---- Combine clips into grid video ----
        final_out_path = output_dir / f"state_{state}_summary_{sampling_method}.mp4"
        self.combine_clips_grid(clip_paths, final_out_path, grid_shape, max_clips)
        print(f"Created: {final_out_path.name}")


    def calculate_block_features(self, ds, df_csv, blocks):
        """Calculate comprehensive behavioral features for each block."""
        import numpy as np
        
        block_features = []
        valid_blocks = []
        
        for start, end in blocks:
            try:
                # Extract segment
                segment = ds.isel(time=slice(start, end))
                segment_csv = df_csv.iloc[start:end]
                keypoint_names = segment.keypoints.values
                features = {}

                body_center_idx = list(keypoint_names).index("Body Centre")
                body_center_only = segment['position'][:, :, body_center_idx:body_center_idx+1, :]

                # Flatten x, y to 1D
                x = body_center_only[:, 0, 0, 0].values.reshape(-1)  # (time,)
                y = body_center_only[:, 1, 0, 0].values.reshape(-1)  # (time,)

                # Speed (scalar already)
                speed = segment['speed'].isel(individuals=0).values.reshape(-1)  # (time,)

                # Acceleration: vector → magnitude per frame
                if "acceleration" in segment:
                    accel_vec = segment['acceleration'].isel(individuals=0).values  # (time, space)
                    acceleration = np.linalg.norm(accel_vec, axis=1).reshape(-1)   # (time,)
                else:
                    acceleration = np.zeros_like(speed)  # fallback to keep features non-empty

                print(f"\n--- Block {start}:{end} ---")
                print("x shape:", x.shape, "y shape:", y.shape)
                print("speed shape:", speed.shape, "accel shape:", acceleration.shape)


                features['speed_mean'] = np.nanmean(speed)
                features['speed_std'] = np.nanstd(speed)
                features['speed_max'] = np.nanmax(speed)
                print("x shape:", x.shape, "y shape:", y.shape)
                print("speed shape:", speed.shape, "accel shape:", acceleration.shape)
                
                features['accel_mean'] = np.nanmean(np.abs(acceleration))
                features['accel_std'] = np.nanstd(acceleration)

                
                # Movement smoothness (jerk)
                jerk = np.diff(acceleration) if len(acceleration) > 1 else [0]
                features['jerk_mean'] = np.nanmean(np.abs(jerk))

                
                # ---- Position Features ----
                # Central tendency
                features['pos_x_mean'] = x.mean().item()
                features['pos_y_mean'] = y.mean().item()

                # Variability
                features['pos_x_std'] = x.std().item()
                features['pos_y_std'] = y.std().item()
                
                # Range of movement
                features['pos_x_range'] = (x.max() - x.min()).item()
                features['pos_y_range'] = (y.max() - y.min()).item()

                # Distance from arena center (assuming center is at 0.5, 0.5 normalized coords)
                center_dist = np.sqrt((x - 0.5)**2 + (y - 0.5)**2)
                features['center_distance_mean'] = center_dist.mean().item()
                features['center_distance_std'] = center_dist.std().item()

                
                # ---- Body Shape Features ----
                # Body length
                if "body_length" in segment_csv.columns:
                    features["body_length_mean"] = segment_csv["body_length"].mean()
                    features["body_length_std"] = segment_csv["body_length"].std()
                    features["body_length_range"] = segment_csv["body_length"].max() - segment_csv["body_length"].min()
                else:
                    features["body_length_mean"] = 0
                    features["body_length_std"] = 0
                    features["body_length_range"] = 0

                # Orientation (from head_direction column)
                if "head_direction" in segment_csv.columns:
                    features["orientation_mean"] = segment_csv["head_direction"].mean()
                    features["orientation_std"] = segment_csv["head_direction"].std()
                else:
                    features["orientation_mean"] = 0
                    features["orientation_std"] = 0
                
                # ---- Path Features ----
                # Total path length
                path_length = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
                # Displacement (straight-line distance)
                displacement = np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2)
                # Tortuosity (path length / displacement)
                features['path_length'] = path_length
                features['displacement'] = displacement
                features['tortuosity'] = path_length / max(displacement, 0.001)  # Avoid division by zero

                
                block_features.append(features)
                valid_blocks.append((start, end))
                
            except Exception as e:
                print(f"    Error calculating features for block {start}-{end}: {e}")
                continue
        
        # Get feature names
        feature_names = list(block_features[0].keys()) if block_features else []
        
        return block_features, valid_blocks, feature_names


    def cluster_behavioral_blocks(self, block_features, feature_names, n_clusters, state):
        """Cluster blocks based on behavioral features and generate descriptive names."""
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Initialize clustering info dictionary
        clustering_info = {
            'n_clusters_requested': n_clusters,
            'total_blocks': len(block_features),
            'behavioral_features_used': [],
            'feature_variances': {},
            'warnings': [],
            'cluster_details': {}
        }
        
        # Filter out temporal features that aren't behaviorally meaningful for clustering
        behavioral_feature_names = [name for name in feature_names 
                                if not any(temporal in name for temporal in ['duration', 'start_time'])]
        
        clustering_info['behavioral_features_used'] = behavioral_feature_names
        print(f"  Using {len(behavioral_feature_names)} behavioral features for clustering:")
        print(f"    Features: {behavioral_feature_names}")
        
        # Convert to numpy array with only behavioral features
        feature_matrix = np.array([[f[name] for name in behavioral_feature_names] for f in block_features])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix)
        
        # Calculate and store feature variances
        print(f"  Feature variance analysis:")
        non_zero_variances = {}
        zero_variance_count = 0
        
        for i, name in enumerate(behavioral_feature_names):
            variance = np.var(feature_matrix[:, i])
            if variance > 1e-10:  # Non-zero variance threshold
                non_zero_variances[name] = float(variance)
                print(f"    {name}: {variance:.6f}")
            else:
                zero_variance_count += 1
        
        clustering_info['feature_variances'] = non_zero_variances
        if zero_variance_count > 0:
            warning = f"{zero_variance_count} features had zero or near-zero variance"
            clustering_info['warnings'].append(warning)
            print(f"    {warning}")
        
        # Check total variance
        total_variance = np.sum(np.var(feature_matrix, axis=0))
        clustering_info['total_variance'] = float(total_variance)
        print(f"  Total variance: {total_variance:.6f}")
        
        if total_variance < 1e-10:
            warning = "Very low variance detected - blocks may be too similar to cluster meaningfully"
            clustering_info['warnings'].append(warning)
            clustering_info['clustering_method'] = 'sequential_fallback'
            print(f"  WARNING: {warning}")
            
            # Create simple sequential naming
            cluster_labels = np.arange(len(block_features)) % n_clusters
            cluster_names = [f"group_{i}" for i in range(n_clusters)]
            
            for i in range(n_clusters):
                clustering_info['cluster_details'][f'cluster_{i}'] = {
                    'name': cluster_names[i],
                    'distinguishing_features': ['insufficient_variance_for_clustering'],
                    'n_blocks': int(np.sum(cluster_labels == i))
                }
            
            return cluster_labels, cluster_names, clustering_info
        
        clustering_info['clustering_method'] = 'kmeans'
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # Check cluster distribution
        unique, counts = np.unique(cluster_labels, return_counts=True)
        cluster_distribution = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
        clustering_info['cluster_distribution'] = cluster_distribution
        print(f"  Cluster distribution: {cluster_distribution}")
        
        # Generate descriptive names for each cluster
        cluster_names = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = feature_matrix[cluster_mask]
            
            if len(cluster_features) == 0:
                cluster_name = f"empty_cluster_{cluster_id}"
                clustering_info['cluster_details'][f'cluster_{cluster_id}'] = {
                    'name': cluster_name,
                    'distinguishing_features': ['no_blocks_assigned'],
                    'n_blocks': 0
                }
                cluster_names.append(cluster_name)
                continue
            
            # Calculate mean features for this cluster
            cluster_mean = np.mean(cluster_features, axis=0)
            
            # Find the most distinctive features for this cluster
            # Compare to overall mean
            overall_mean = np.mean(feature_matrix, axis=0)
            feature_importance = np.abs(cluster_mean - overall_mean)
            
            # Get top 3 most distinctive features
            top_features_idx = np.argsort(feature_importance)[-3:]
            distinguishing_features = []
            
            name_parts = []
            for idx in reversed(top_features_idx):  # Start with most important
                feature_name = behavioral_feature_names[idx]
                feature_value = cluster_mean[idx]
                overall_value = overall_mean[idx]
                
                # Store distinguishing feature info
                importance_score = feature_importance[idx]
                if importance_score > 1e-6:  # Only store if meaningfully different
                    distinguishing_features.append({
                        'feature_name': feature_name,
                        'cluster_mean': float(feature_value),
                        'overall_mean': float(overall_value),
                        'importance_score': float(importance_score),
                        'relative_difference': float((feature_value - overall_value) / max(abs(overall_value), 1e-6))
                    })
                
                # Generate descriptive terms based on feature name and relative value
                if feature_value > overall_value:
                    intensity = "high" if feature_value > overall_value * 1.2 else "elevated"
                else:
                    intensity = "low" if feature_value < overall_value * 0.8 else "reduced"
                
                # Simplify feature names for readability with more specific behavioral descriptors
                if 'speed_mean' in feature_name:
                    name_parts.append(f"{intensity}_speed")
                elif 'speed_std' in feature_name:
                    if 'high' in intensity:
                        name_parts.append("variable_speed")
                    else:
                        name_parts.append("steady_speed")
                elif 'center_distance_mean' in feature_name:
                    if 'high' in intensity or 'elevated' in intensity:
                        name_parts.append("periphery")
                    else:
                        name_parts.append("center")
                elif 'pos_x_mean' in feature_name:
                    if feature_value > 0.6:
                        name_parts.append("right_side")
                    elif feature_value < 0.4:
                        name_parts.append("left_side")
                    else:
                        name_parts.append("x_center")
                elif 'pos_y_mean' in feature_name:
                    if feature_value > 0.6:
                        name_parts.append("top_area")
                    elif feature_value < 0.4:
                        name_parts.append("bottom_area")
                    else:
                        name_parts.append("y_center")
                elif 'pos_x_std' in feature_name or 'pos_y_std' in feature_name:
                    if 'high' in intensity:
                        name_parts.append("wide_ranging")
                    else:
                        name_parts.append("localized")
                elif 'tortuosity' in feature_name:
                    if 'high' in intensity:
                        name_parts.append("winding")
                    else:
                        name_parts.append("direct")
                elif 'accel_mean' in feature_name:
                    name_parts.append(f"{intensity}_accel")
                elif 'jerk_mean' in feature_name:
                    if 'high' in intensity:
                        name_parts.append("erratic")
                    else:
                        name_parts.append("smooth")
                elif any(shape in feature_name for shape in ['body_length', 'width', 'area']):
                    body_part = feature_name.split('_')[0]
                    name_parts.append(f"{intensity}_{body_part}")
                elif 'eccentricity' in feature_name:
                    if 'high' in intensity:
                        name_parts.append("elongated")
                    else:
                        name_parts.append("compact")
                
                if len(name_parts) >= 2:  # Limit to 2 main descriptors
                    break
            
            # Create final name
            if not name_parts:
                cluster_name = f"cluster_{cluster_id}"
            else:
                cluster_name = "_".join(name_parts[:2])
            
            cluster_names.append(cluster_name)
            
            # Store cluster details in metadata
            clustering_info['cluster_details'][f'cluster_{cluster_id}'] = {
                'name': cluster_name,
                'distinguishing_features': distinguishing_features,
                'n_blocks': int(np.sum(cluster_labels == cluster_id))
            }
        
        print(f"  Identified behavioral clusters for state {state}:")
        for i, name in enumerate(cluster_names):
            n_blocks = np.sum(cluster_labels == i)
            print(f"    Cluster {i}: {name} ({n_blocks} blocks)")
        
        return cluster_labels, cluster_names, clustering_info


    def sample_by_behavioral_features(self, ds, blocks, state, max_clips):
        """Sample blocks based on behavioral diversity using clustering."""
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Calculate features for each block
        block_features, valid_blocks, feature_names = self.calculate_block_features(ds, blocks)
        
        if len(valid_blocks) <= max_clips:
            return valid_blocks
        
        # Convert to numpy array for clustering
        feature_matrix = np.array([[f[name] for name in feature_names] for f in block_features])
        
        # Handle NaN values
        feature_matrix = np.nan_to_num(feature_matrix)
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Perform clustering
        n_clusters = min(max_clips, len(valid_blocks))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix_scaled)
        
        # Select representative block from each cluster
        selected_blocks = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # Find block closest to cluster center
                cluster_center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(feature_matrix_scaled[cluster_mask] - cluster_center, axis=1)
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_blocks.append(valid_blocks[closest_idx])
        
        print(f"  Clustered {len(valid_blocks)} blocks into {n_clusters} groups for state {state}")
        return selected_blocks


    def sample_by_time(self, blocks, max_clips):
        """Sample blocks evenly distributed across time."""
        time_positions = [start for start, end in blocks]
        sorted_blocks = sorted(zip(time_positions, blocks), key=lambda x: x[0])
        
        indices = np.linspace(0, len(sorted_blocks)-1, max_clips, dtype=int)
        selected_blocks = [sorted_blocks[i][1] for i in indices]
        
        print(f"  Selected {len(selected_blocks)} blocks using temporal sampling")
        return selected_blocks


    def sample_by_location(self, ds, blocks, state, max_clips):
        """Sample blocks from different spatial regions."""
        import random
        
        # Get center positions for each block
        block_positions = []
        valid_blocks = []
        
        for start, end in blocks:
            try:
                segment = ds.isel(time=slice(start, end))
                if 'centroid_x' in segment and 'centroid_y' in segment:
                    x_mean = segment['centroid_x'].mean().item()
                    y_mean = segment['centroid_y'].mean().item()
                    block_positions.append((x_mean, y_mean))
                    valid_blocks.append((start, end))
            except:
                continue
        
        if len(valid_blocks) <= max_clips:
            return valid_blocks
        
        # Create spatial bins
        x_coords, y_coords = zip(*block_positions)
        x_bins = np.percentile(x_coords, np.linspace(0, 100, 5))  # 4x4 grid
        y_bins = np.percentile(y_coords, np.linspace(0, 100, 5))
        
        # Sample from each spatial bin
        selected_blocks = []
        random.seed(42)  # For reproducibility
        
        for x_bin in range(len(x_bins)-1):
            for y_bin in range(len(y_bins)-1):
                candidates = []
                for i, (x, y) in enumerate(block_positions):
                    if (x_bins[x_bin] <= x < x_bins[x_bin+1] and 
                        y_bins[y_bin] <= y < y_bins[y_bin+1]):
                        candidates.append(valid_blocks[i])
                
                if candidates and len(selected_blocks) < max_clips:
                    selected_blocks.append(random.choice(candidates))
        
        # If we didn't get enough from spatial sampling, fill with random selection
        if len(selected_blocks) < max_clips:
            remaining = [b for b in valid_blocks if b not in selected_blocks]
            additional_needed = max_clips - len(selected_blocks)
            if remaining:
                selected_blocks.extend(random.sample(remaining, 
                                                min(additional_needed, len(remaining))))
        
        print(f"  Selected {len(selected_blocks)} blocks using spatial sampling")
        return selected_blocks


    def combine_clips_grid(self, clip_paths, out_path, grid_shape, max_clips):
        """Combine clips into a grid video."""
        rows, cols = grid_shape
        cap_list = [cv2.VideoCapture(str(p)) if p.exists() else None for p in clip_paths]

        # Assume all clips same fps/resolution (otherwise resize them)
        fps = 30
        width, height = 320, 240

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (cols*width, rows*height))

        # Get max frames of all clips
        max_frames = max([int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in cap_list if c is not None])

        for f in range(max_frames):
            frame_grid = []
            for r in range(rows):
                row_frames = []
                for c in range(cols):
                    idx = r*cols + c
                    if idx < len(cap_list) and cap_list[idx] is not None:
                        ret, frame = cap_list[idx].read()
                        if not ret:
                            frame = np.zeros((height,width,3), np.uint8)
                    else:
                        frame = np.zeros((height,width,3), np.uint8)
                    row_frames.append(cv2.resize(frame, (width, height)))
                frame_grid.append(np.hstack(row_frames))
            out.write(np.vstack(frame_grid))

        out.release()
        for c in cap_list:
            if c is not None:
                c.release()

        
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
                self.predict_states()
            
            self.update_session_data()

            self.add_states_to_xarray(self.states_csv_path, self.tracking_data_xr)

            # Analysis and visualization
            if self.show_plots or self.save_plots:
                print("Creating visualizations...")
                self.plot_statewise_feature_summary(feature_column='smoothed_speed')
                self.plot_statewise_feature_summary(feature_column='smoothed_acceleration')
                #self.plot_statewise_feature_summary(feature_column='smoothed_body_length')
                self.plot_state_distribution()
                self.plot_state_sequence()
                self.plot_dwell_times()
                self.plot_transition_matrix()
                #self.plot_state_behavior_correlation(behavior_columns)
                #self.plot_states_with_behaviors(behavior_columns)
            
            # Video annotations
            if self.create_animations:
                # self.create_video_annotations(start_frame=start_frame, end_frame=end_frame)
                self.create_state_summary_videos(min_duration=40, max_clips=12, grid_shape=(3,4),
                                                 videos_per_state=3, sampling_method='temporal')
                self.create_state_summary_videos(min_duration=40, max_clips=4, grid_shape=(2,2),
                                                 videos_per_state=3, sampling_method='behavioral_clustering')


            
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