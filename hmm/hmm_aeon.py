import autograd.numpy as np
import autograd.numpy.random as npr

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
            "abdomen_abdomen",
            "snout_groin",
            "abdomen_port0",
            "abdomen_port1",
            "sniff_freq"
        ]  # Expected features in the input data
        self.model = None  # HMM model instance
        self.parameters = None  # Sorted model parameters (mean, variance, covariance)
        self.transition_mat = None  # Sorted transition matrix
        self.states = None  # Inferred states
        self.connectivity_mat = None  # Connectivity matrix
        self.test_lls = None  # Log-likelihoods of the test data
        self.train_lls = None  # Log-likelihoods of the training data

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
        fitting_input = np.array(train_data)
        self.model = ssm.HMM(
            self.n_state, len(fitting_input[0]), observations="gaussian"
        )
        lls = self.model.fit(
            fitting_input, method="em", num_iters=num_iters, init_method="kmeans"
        )
        self.train_lls = lls

    def infer_states(self, test_data):
        """Infer states for the test data."""
        obs = np.array(test_data)
        self.test_lls = self.model.log_likelihood(obs)
        self.states = self.model.most_likely_states(obs)

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


def run_inference_hmm(model_path, inference_path):
    k=model_path.split('=')[1][0]
    model = AeonHMM.load(model_path)
    inference_list = inference_path.rglob("*completed_distances_imputed.csv")
    for path in inference_list:
        infer_data = pd.read_csv(path)[
            [
                "smoothed_speed",
                "smoothed_acceleration",
                "abdomen_abdomen",
                "snout_groin",
                "abdomen_port0",
                "abdomen_port1",
                "sniff_freq"
            ]
        ]
        model.infer_states(infer_data)
        infer_data['states'] = model.states
        output_path = path.parent / f"hmm_inferred_states_K={k}.csv"
        infer_data.to_csv(output_path, index=False)
        print(f"Saved inferred states to {output_path}")

def fit_model_hmm(concat_data_path, restarts=5, Ks=range(2,11)):
    train_data = pd.read_csv(concat_data_path)[ 
        [
            "smoothed_speed",
            "smoothed_acceleration",
            "abdomen_abdomen",
            "snout_groin",
            "abdomen_port0",
            "abdomen_port1",
            "sniff_freq"
        ]
    ]
    best_ll = -np.inf
    best = None
    for K in Ks:
        print(f"Fitting HMM with K={K} states...")
        for seed in range(restarts):
            npr.seed(seed)
            h = AeonHMM(n_state=K)
            h.fit_model(train_data, num_iters=200)
            if h.train_lls[-1] > best_ll:
                best_ll = h.train_lls[-1]
                best = h
        best.save(f'F:\\hmm\\models\\best_hmm_K={K}.pkl')


def run_hmm_analysis(K, session_path, fit_model=False):
    mouse_hmm = AeonHMM(n_state=K)
    if fit_model:
        train_data = pd.read_csv(session_path / "completed_distances_imputed.csv")[  # Replace with actual path
            [
                "smoothed_speed",
                "smoothed_acceleration",
                "abdomen_abdomen",
                "snout_groin",
                "abdomen_port0",
                "abdomen_port1",
                "sniff_freq"
            ]
        ]
        mouse_hmm.fit_model(train_data)
    else:
        model = AeonHMM.load(session_path / f"hmm_model_K={K}.pkl")  # Load pre-trained model
        infer_data = pd.read_csv(session_path / "completed_distances_imputed.csv")[  # Replace with actual path
            [
                "smoothed_speed",
                "smoothed_acceleration",
                "abdomen_abdomen",
                "snout_groin",
                "abdomen_port0",
                "abdomen_port1",
                "sniff_freq"
            ]
        ]
        model.infer_states(infer_data)

#fit_model_hmm(r"F:\social_sniffing\training_data_hmm\concatenated_sessions.csv", restarts=3, Ks=range(12, 15))
run_inference_hmm(model_path=r"F:\hmm\models\best_hmm_K=6.pkl", inference_path= Path(r"F:\social_sniffing\derivatives"))

