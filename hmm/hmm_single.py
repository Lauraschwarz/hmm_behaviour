from os import mkdir
from xml.parsers.expat import model
import autograd.numpy as np
import autograd.numpy.random as npr
from scipy.ndimage import gaussian_filter1d

npr.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import seaborn as sns

import ssm


class AeonHMM:
    """A class for training and analysing Hidden Markov Models (HMM) using the `ssm` library."""

    def __init__(self, n_state):
        """Initialise AeonHMM with the number of hidden states."""
        self.n_state = n_state  # Number of hidden states
        self.features = [
            "smoothed_speed",
            "smoothed_acceleration",
            "snout_groin",
            "neckL_groin",
            "neckR_groin",
            "sniff_freq"
                    ]  # Expected features in the input data
        self.model = None  # HMM model instance
        self.parameters = None  # Sorted model parameters (mean, variance, covariance)
        self.transition_mat = None  # Sorted transition matrix
        self.states = None  # Inferred states
        self.connectivity_mat = None  # Connectivity matrix
        self.test_lls = None  # Log-likelihoods of the test data
        self.train_lls = None  # Log-likelihoods of the training data

    def df_to_sequences(self, df, feature_cols,mouse, trial_col="trial"):
        sequences = []
        indices = []
        for _, g in df.groupby([mouse, trial_col], sort=False):
            if len(g) < 2:
                continue
            sequences.append(g[feature_cols].to_numpy())
            indices.append(g.index.to_numpy())
        return sequences, indices


    def get_connectivity_matrix(self):
        """Compute the normalised connectivity matrix from the inferred states."""
        connectivity_mat = np.zeros((self.n_state, self.n_state))
        states = self.states
        # Count transitions between states
        for i in range(len(states) - 1):
            if states[i + 1] != states[i]:
                connectivity_mat[states[i]][states[i + 1]] += 1
        # Normalise to sum to 1
        for i in range(self.n_state):
            total = np.sum(connectivity_mat[i])
            if total > 0:
                connectivity_mat[i] /= total

        return connectivity_mat

    def fit_model(self, train_data, num_iters=250):
        """Fit the HMM model to the training data using the EM algorithm."""
        fitting_input, _ = self.df_to_sequences(df=train_data, feature_cols=self.features, mouse = 'mouse_id',trial_col="trial")

        self.model = ssm.HMM(
            self.n_state,
            fitting_input[0].shape[1],
            observations="gaussian",
            transitions="sticky",
            transition_kwargs=dict(kappa=5.0, alpha=1.0)
            

            
        )

        lls = self.model.fit(
            fitting_input,
            method="em",
            num_iters=num_iters,
            init_method="kmeans", 
           
        )
  
        self.train_lls = lls

    def infer_states(self, test_data):
        """Infer states for the test data, respecting trial boundaries."""
        sequences, indices = self.df_to_sequences(
            test_data, self.features, mouse = 'mouse_id', trial_col="trial"
        )

        # allocate output array aligned to test_data
        states = np.full(len(test_data), -1, dtype=int)

        lls = []
        for X, idx in zip(sequences, indices):
            z = self.model.most_likely_states(X)
            states[idx] = z
            lls.append(self.model.log_likelihood(X))

        self.states = states
        self.test_lls = np.array(lls)

    def infer_states_with_posteriors(self, test_data):

        sequences, indices = self.df_to_sequences(
            test_data, self.features, mouse='mouse_id', trial_col="trial"
        )

        states = np.full(len(test_data), -1, dtype=int)
        posteriors = np.zeros((len(test_data), self.model.K))

        lls = []

        for X, idx in zip(sequences, indices):

            # Viterbi path
            z = self.model.most_likely_states(X)
            states[idx] = z

            # Posterior probabilities
            Ez, _, _ = self.model.expected_states(X)
            posteriors[idx] = Ez

            lls.append(self.model.log_likelihood(X))

        self.states = states
        self.posteriors = posteriors
        self.test_lls = np.array(lls)

    def sort(self, sort_idx):
        """Sort the model parameters, transition matrix, and inferred states based on the provided indices."""
        # Sort Gaussian means: shape (n_features, n_state)
        parameters_mean_sorted = self.model.observations.params[0][sort_idx].T
        # Extract and sort variances: shape (n_features, n_state)
        parameters_var = np.zeros((self.n_state, len(self.features)))
        for i in range(self.n_state):
            for j in range(len(self.features)):
                # state i, feature j
                parameters_var[i, j] = self.model.observations.params[1][i][j][j]
        parameters_var_sorted = parameters_var[sort_idx].T
        # Sort covariance matrices: shape (n_state, n_features, n_features)
        parameters_covar_sorted = self.model.observations.params[1][sort_idx]
        self.parameters = [
            parameters_mean_sorted,
            parameters_var_sorted,
            parameters_covar_sorted,
        ]
        # Sort transition matrix: shape (n_state, n_state)
        self.transition_mat = (
            self.model.transitions.transition_matrix[sort_idx].T[sort_idx].T
        )
        # Compute connectivity matrix
        self.connectivity_mat = self.get_connectivity_matrix()
        # Reassign state labels to reflect new order
        new_values = np.empty_like(self.states)
        for i, val in enumerate(sort_idx):
            new_values[self.states == val] = i
        self.states = new_values

    def save(self, path):
        """Save the fitted HMM and relevant AeonHMM state to a single file (pickle)."""
      
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump({
                "model": self.model,
                "n_state": self.n_state,
                "features": self.features,
                "parameters": self.parameters,
                "transition_mat": self.transition_mat,
                "states": self.states,
                "connectivity_mat": self.connectivity_mat,
                "train_lls": self.train_lls,
                "test_lls": self.test_lls
            }, fh)

    def check_convergence(self):
        if self.train_lls is None:
            raise RuntimeError("Model not fit yet")

        return {
            "initial_ll": self.train_lls[0],
            "final_ll": self.train_lls[-1],
            "delta_ll": self.train_lls[-1] - self.train_lls[0],
            "n_iters": len(self.train_lls),
            "monotonic": np.all(np.diff(self.train_lls) >= -1e-6)
    }

    def check_transitions(self):
        A = self.model.transitions.transition_matrix
        diag = np.diag(A)

        return {
            "transition_matrix": A,
            "self_transition": diag,
            "mean_self_transition": diag.mean(),
            "dead_states": np.where(A.sum(axis=1) == 0)[0]
        }
    def get_emission_summary(self):
        means = self.model.observations.params[0]          # (K, D)
        covs = self.model.observations.params[1]           # (K, D, D)

        variances = np.stack([np.diag(c) for c in covs])

        return {
            "means": means,
            "variances": variances,
            "min_variance": variances.min(),
            "max_variance": variances.max()
        }
    
    def state_occupancy(self):
        valid = self.states[self.states >= 0]
        counts = np.bincount(valid, minlength=self.n_state)
        return counts / counts.sum()
    
    def state_dwell_times(self):
        dwell = {k: [] for k in range(self.n_state)}
        s = self.states
        prev = s[0]
        run = 1

        for curr in s[1:]:
            if curr == prev:
                run += 1
            else:
                dwell[prev].append(run)
                prev = curr
                run = 1
        dwell[prev].append(run)

        return dwell
    
    def state_feature_stats(self, df):
        if "states" not in df:
            raise ValueError("df must contain inferred states")

        return df.groupby("states")[self.features].agg(["mean", "std"])

    # def framewise_loglikelihood(self, df):
    #     sequences, indices = self.df_to_sequences(df, self.features, trial_col="trial")

    #     ll = np.full(len(df), np.nan)
    #     for X, idx in zip(sequences, indices):
    #         ll[idx] = self.model.log_likelihood(X, per_frame=True)

    #     return ll



    @classmethod
    def load(cls, path):
        """Load an AeonHMM instance (model + state) from a pickle file created by save()."""
        data = pickle.load(open(path, "rb"))
        obj = cls(data["n_state"])
        obj.model = data.get("model")
        obj.features = data.get("features", obj.features)
        obj.parameters = data.get("parameters")
        obj.transition_mat = data.get("transition_mat")
        obj.states = data.get("states")
        obj.connectivity_mat = data.get("connectivity_mat")
        obj.train_lls = data.get("train_lls")
        obj.test_lls = data.get("test_lls")
        return obj

def plot_transition_matrix(trans, output_path=None):
    matrix = trans['transition_matrix']
    annot_array = np.array([[round(item, 3) for item in row] for row in matrix])
    labels = ["$S_{" + str(i + 1) + "}$" for i in range(len(matrix))]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        cmap="RdBu",
        ax=ax,
        square="True",
        cbar=True,
        annot=annot_array,
    )
    ax.set_title("Transition Matrix")
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels, rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def run_inference_hmm(model_dir,inference_path, k=5, model_type='gaussian' ):

    model_path = model_dir / model_type / f"best_hmm_K={k}.pkl"
    output_dir = inference_path.parent / "model_predictions" / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AeonHMM.load(model_path)
    qc = model.check_convergence()
    print(qc) #qc delta should be much larger than 0 monotonic should be true

    trans = model.check_transitions()
    print(trans["mean_self_transition"]) #should be around 1

    occ = model.state_occupancy()
    print(occ)

    dwell = model.state_dwell_times()

    inference_list = inference_path.rglob("*imputed_split_sampling_bouts_v2.csv")
    for path in inference_list:
        print(f"Running inference on {path}...")
        infer_data = pd.read_csv(path)[
            [
                
            "smoothed_speed",
            "smoothed_acceleration",
            "snout_groin",
            "neckL_groin",
            "neckR_groin",
            "sniff_freq",
            "trial", 
            'mouse_id'
            ]
        ]
        infer_data["speed_lag1"] = infer_data["smoothed_speed"].shift(1)
        infer_data['speed_lag2'] = infer_data["smoothed_speed"].shift(2)
        infer_data["snout_groin_lag1"] = infer_data["snout_groin"].shift(1)
        infer_data["sniff_lag1"] = infer_data["sniff_freq"].shift(1)

        # infer_data = infer_data.groupby(["mouse_id","trial"])[infer_data.columns].transform(lambda x: x.rolling(3, center=True, min_periods=1).mean())
        # infer_data = bin_dataframe(infer_data, bin_size=0, feature_cols=infer_data.columns)

        model.infer_states(infer_data)
        model.infer_states_with_posteriors(infer_data)
        infer_data['states'] = model.states
        Ez_smooth = posterior_smooth(model.posteriors, window=50)

        infer_data["states_smoothed"] = Ez_smooth.argmax(axis=1)
        infer_data["state_confidence"] = Ez_smooth.max(axis=1)


        #infer_data["ll"] = model.framewise_loglikelihood(infer_data)
        stats = model.state_feature_stats(infer_data)

        output_path = output_dir / f"hmm_inferred_states_K={k}.csv" 
        plot_transition_matrix(trans, output_path=(output_dir/ f"transition_matrix_K={k}.png"))

        infer_data.to_csv(output_path, index=False)
        add_states_to_df(output_dir, inference_path, k=k)

        print(f"Saved inferred states to {output_path}")

def posterior_smooth(post, window=5):
    sm = gaussian_filter1d(post, sigma=2, axis=0, mode="nearest")
    sm /= sm.sum(axis=1, keepdims=True)
    return sm



def bin_dataframe(df, bin_size, feature_cols,
                  group_cols=("mouse_id", "trial")):

    binned_parts = []

    for _, g in df.groupby(list(group_cols)):
        g = g.reset_index(drop=True)

        bins = g.index // bin_size

        g_bin = (
            g.groupby(bins)[feature_cols]
            .mean()
            .reset_index(drop=True)
        )

        # Reattach metadata
        for col in group_cols:
            g_bin[col] = g[col].iloc[0]

        binned_parts.append(g_bin)

    return pd.concat(binned_parts, ignore_index=True)

def fit_model_hmm(concat_data_path,output_path, restarts=5, Ks=range(2,11)):
    train_data = pd.read_csv(concat_data_path)[ 
        [
            "smoothed_speed",
            "smoothed_acceleration",
            "snout_groin",
            "neckL_groin",
            "neckR_groin",
            "sniff_freq",
            "trial", 
            'mouse_id'
        ]
    ]
    train_data["speed_lag1"] = train_data["smoothed_speed"].shift(1)
    train_data['speed_lag2'] = train_data["smoothed_speed"].shift(2)
    train_data["snout_groin_lag1"] = train_data["snout_groin"].shift(1)
    train_data["sniff_lag1"] = train_data["sniff_freq"].shift(1)
#     train_data[train_data.columns] = (
#     train_data
#     .groupby(["mouse_id","trial"])[train_data.columns]
#     .transform(lambda x: x.rolling(3, center=True, min_periods=1).mean())
# )
#     train_data = bin_dataframe(train_data, bin_size=0, feature_cols=train_data.columns)

    best_ll = -np.inf
    best = None
    for K in Ks:
        print(f"Fitting HMM with K={K} states...")
        for seed in range(restarts):
            npr.seed(seed)
            h = AeonHMM(n_state=K)

            h.fit_model(train_data, num_iters=200)
            h.infer_states(train_data)
            h.get_connectivity_matrix()
            if h.train_lls[-1] > best_ll:
                best_ll = h.train_lls[-1]
                best = h
        best.save(Path(output_path) / (f"best_hmm_K={K}.pkl"))



def add_states_to_df(path, inference_path, k=5, bin_factor=0):
    
    hmm_states_df = pd.read_csv(path / f"hmm_inferred_states_K={k}.csv")
    sampling_bouts_df = pd.read_csv(inference_path / "imputed_split_sampling_bouts_v2.csv")
    all_session_df = pd.read_csv(inference_path / "all_sessions_df_v4.csv")

    print(f"Adding {k} states to inference dataframe...")
    # z_bin = hmm_states_df["states"].to_numpy()
    # z_full = np.repeat(z_bin, bin_factor)
    # N = len(sampling_bouts_df)
    # if len(z_full) < N:
    #     pad = np.full(N - len(z_full), z_full[-1])
    #     z_full = np.concatenate([z_full, pad])
    # else:
    #     z_full = z_full[:N]
    sampling_bouts_df["states"] = hmm_states_df['states']
    sampling_bouts_df["states_smoothed"] = hmm_states_df['states_smoothed'] 
    sampling_bouts_df['states_confidence'] = hmm_states_df['state_confidence'] 
    sampling_bouts_df.to_csv(path / f"inference_{k}states.csv", index=False)
    merged_df_all = all_session_df.merge(
        sampling_bouts_df[['Unnamed: 0', 'states', 'states_smoothed', 'states_confidence']],
        on='Unnamed: 0',
        how='left'   # safer than outer here
    )
    merged_df_all.to_csv(path / f"all_session_data_{k}states.csv", index=False)
    print(f"Saved merged dataframe with states to {path / f'all_session_data_{k}states.csv'}")

def run_hmm_analysis(K, session_path, fit_model=False):
    mouse_hmm = AeonHMM(n_state=K)
    if fit_model:
        train_data = pd.read_csv(session_path / "imputed_split_sampling_bouts_v2.csv")[  # Replace with actual path
            [
            "smoothed_speed",
            "smoothed_acceleration",
            "snout_groin",
            "neckL_groin",
            "neckR_groin",
            "sniff_freq",
            "trial",
            "mouse_id"
            ]
        ]
        train_data["speed_lag1"] = train_data["smoothed_speed"].shift(1)
        train_data['speed_lag2'] = train_data["smoothed_speed"].shift(2)
        train_data["snout_groin_lag1"] = train_data["snout_groin"].shift(1)
        train_data["sniff_lag1"] = train_data["sniff_freq"].shift(1)
        mouse_hmm.fit_model(train_data)
    else:
        model = AeonHMM.load(session_path / f"hmm_model_K={K}.pkl")  # Load pre-trained model
        infer_data = pd.read_csv(session_path / "imputed_split_sampling_bouts_v2.csv")[  # Replace with actual path
            [
            "smoothed_speed",
            "smoothed_acceleration",
            "snout_groin",
            "neckL_groin",
            "neckR_groin",
            "sniff_freq",
            "trial",
            "mouse_id"
            ]
        ]
        infer_data["speed_lag1"] = infer_data["smoothed_speed"].shift(1)
        infer_data['speed_lag2'] = infer_data["smoothed_speed"].shift(2)
        infer_data["snout_groin_lag1"] = infer_data["snout_groin"].shift(1)
        infer_data["sniff_lag1"] = infer_data["sniff_freq"].shift(1)

        model.infer_states(infer_data)

#fit_model_hmm(r"F:\social_sniffing\derivatives\test_concat_hmm_features_social_all_mice.csv", output_path=r"F:\hmm\models\social\gaussian_raw", restarts=5, Ks=range(3,8))
for k in range(4,7):
    run_inference_hmm(model_dir=Path(r"F:\hmm\models\all_mice"), k=k, model_type='gaussian_raw', inference_path=Path(rf"F:\social_sniffing\derivatives\1106010\olfactory_ctrls\CTRL\hmm_combined"))
    #
    #run_inference_hmm(model_dir=Path(r"F:\hmm\models\all_mice"), k=k, model_type='gaussian_raw', inference_path=Path(rf"F:\social_sniffing\derivatives\1106009\olfactory_ctrls\CTRL\hmm_combined"))
