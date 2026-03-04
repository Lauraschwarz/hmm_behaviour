import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import ssm
import numpy.random as npr
from plotting_hmm import  plot_transition_circos, plot_emission_means_line
import matplotlib.pyplot as plt
import seaborn as sns



class AeonHMM:
    """
    HMM wrapper around `ssm` with:
      - bout-respecting sequences (grouped by mouse_id + trial)
      - derivative (diff) features instead of explicit lag columns
      - **raw-row alignment preserved** in inference outputs
        (rows that can't be used for emissions get states=-1, posteriors=NaN)
    """

    def __init__(
        self,
        n_state: int,
        mouse_col: str = "mouse_id",
        trial_col: str = "trial",
        transitions_kappa: float = 5.0,
        transitions_alpha: float = 1.0,
        add_derivatives: bool = True,
        derivative_cols=None,
        zscore: bool = False,
        zscore_group_cols=None,
    ):
        self.n_state = n_state
        self.mouse_col = mouse_col
        self.trial_col = trial_col

        self.base_features = [
            "smoothed_speed",
            "smoothed_acceleration",
            "snout_groin",
            "neckL_groin",
            "neckR_groin",
            "sniff_freq",
            "speed_con_smoothed",
            "acceleration_con_smoothed",
            "snout_groin_con",
            "neckL_groin_con",
            "neckR_groin_con",
        ]

        self.add_derivatives = add_derivatives
        self.derivative_cols = derivative_cols or [
            "smoothed_speed",
            "sniff_freq",
            "snout_groin",
            "snout_groin_con",
        ]

        self.zscore = zscore
        self.zscore_group_cols = zscore_group_cols  # e.g. ["mouse_id"] or ["mouse_id","session_id"]

        self.transitions_kappa = transitions_kappa
        self.transitions_alpha = transitions_alpha

        self.model = None
        self.parameters = None
        self.transition_mat = None
        self.connectivity_mat = None

        # Inference outputs (ALIGNED TO INPUT DF LENGTH)
        self.states = None                 # shape (N,), -1 where not inferred
        self.posteriors = None             # shape (N,K), NaN where not inferred
        self.test_lls = None               # per-sequence LLs
        self.train_lls = None

        # Per-sequence outputs (use these for dwell/connectivity)
        self.state_sequences = None        # list of z arrays, one per bout
        self.sequence_indices = None       # list of index arrays (raw indices) used for each sequence

    # ---------- Utilities ----------

    def _require_columns(self, df: pd.DataFrame, cols):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _feature_list(self):
        feats = list(self.base_features)
        if self.add_derivatives:
            feats += [f"d_{c}" for c in self.derivative_cols]
        return feats

    # ---------- Feature engineering (alignment-preserving) ----------

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a COPY of df with derivative columns added (if enabled).
        Does NOT drop rows (keeps alignment). Rows with NaNs in feature columns
        will simply be excluded from sequences during fit/infer.
        """
        df = df.copy()
        self._require_columns(df, [self.mouse_col, self.trial_col] + self.base_features)

        if self.add_derivatives:
            self._require_columns(df, self.derivative_cols)
            g = df.groupby([self.mouse_col, self.trial_col], sort=False)
            for c in self.derivative_cols:
                df[f"d_{c}"] = g[c].diff()

        # Optional z-scoring (done without dropping rows; NaNs remain NaNs)
        if self.zscore:
            feats = self._feature_list()
            if self.zscore_group_cols:
                self._require_columns(df, self.zscore_group_cols)
                gg = df.groupby(self.zscore_group_cols, sort=False)
                df[feats] = gg[feats].transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-8))
            else:
                df[feats] = (df[feats] - df[feats].mean()) / (df[feats].std(ddof=0) + 1e-8)

        return df

    def df_to_sequences(self, df: pd.DataFrame, feature_cols):
        """
        Convert df into a list of sequences and matching index arrays.
        Excludes rows with NaNs in ANY feature col (since ssm Gaussian can't handle NaNs).
        Keeps raw alignment by returning raw indices for included rows.
        """
        sequences = []
        indices = []

        for _, g in df.groupby([self.mouse_col, self.trial_col], sort=False):
            # Keep only rows valid for emissions
            valid = np.isfinite(g[feature_cols].to_numpy()).all(axis=1)
            g_valid = g.loc[valid]

            if len(g_valid) < 2:
                continue

            sequences.append(g_valid[feature_cols].to_numpy())
            indices.append(g_valid.index.to_numpy())  # raw index values

        return sequences, indices

    # ---------- Fit / infer ----------

    def fit_model(self, train_data: pd.DataFrame, num_iters: int = 250, init_method: str = "kmeans"):
        df = self.prepare_dataframe(train_data)
        feats = self._feature_list()
        fitting_input, _ = self.df_to_sequences(df=df, feature_cols=feats)

        if len(fitting_input) == 0:
            raise ValueError("No valid sequences after preprocessing (check NaNs / grouping).")

        D = fitting_input[0].shape[1]
        self.model = ssm.HMM(
            self.n_state,
            D,
            observations="gaussian",
            transitions="sticky",
            transition_kwargs=dict(kappa=self.transitions_kappa, alpha=self.transitions_alpha),
        )

        lls = self.model.fit(
            fitting_input,
            method="em",
            num_iters=num_iters,
            init_method=init_method,
        )
        self.train_lls = lls

    def infer_states_with_posteriors(self, test_data: pd.DataFrame):
        """
        Infer Viterbi states + posteriors per bout.

        IMPORTANT: outputs are aligned to the *input test_data rows*:
          - self.states has length N (N=len(test_data))
          - rows excluded from sequences (NaNs etc.) remain states=-1
          - self.posteriors is (N,K) with NaNs for excluded rows

        Returns: df_prepped (same length/order as input, with derivative cols added if enabled)
        """
        if self.model is None:
            raise RuntimeError("Model not fit yet.")

        df = self.prepare_dataframe(test_data)
        feats = self._feature_list()
        sequences, indices = self.df_to_sequences(df, feats)

        N = len(df)
        K = self.model.K

        states = np.full(N, -1, dtype=int)
        posteriors = np.full((N, K), np.nan, dtype=float)

        state_seqs = []
        lls = []

        # Map raw index -> positional row in df (0..N-1)
        # Works even if df index is not 0..N-1
        index_to_pos = pd.Series(np.arange(N), index=df.index)

        for X, idx in zip(sequences, indices):
            print("X shape:", getattr(X, "shape", None), "dtype:", getattr(X, "dtype", None))
            print("model K:", self.model.K)
            try:
                mus = self.model.observations.mus
                Sigmas = self.model.observations.Sigmas
                print("mus shape:", mus.shape)         # expected (K, D_model)
                print("Sigmas shape:", Sigmas.shape)   # expected (K, D_model, D_model)
            except Exception as e:
                print("Could not read mus/Sigmas:", e)
            z = self.model.most_likely_states(X)
            Ez, _, _ = self.model.expected_states(X)

            pos = index_to_pos.loc[idx].to_numpy()
            states[pos] = z
            posteriors[pos] = Ez

            state_seqs.append(z)
            lls.append(self.model.log_likelihood(X))

        self.states = states
        self.posteriors = posteriors
        self.test_lls = np.array(lls)
        self.state_sequences = state_seqs
        self.sequence_indices = indices

        return df

    # ---------- Sequence-respecting diagnostics ----------

    def get_connectivity_matrix(self):
        """
        Empirical transition probabilities computed from per-bout inferred sequences.
        """
        if self.state_sequences is None:
            raise RuntimeError("Run infer_states_with_posteriors first.")

        C = np.zeros((self.n_state, self.n_state), dtype=float)
        for z in self.state_sequences:
            for a, b in zip(z[:-1], z[1:]):
                if a != b:
                    C[a, b] += 1.0

        row_sums = C.sum(axis=1, keepdims=True)
        C = np.divide(C, row_sums, where=row_sums > 0)
        self.connectivity_mat = C
        return C

    def state_occupancy(self):
        """
        Occupancy pooled over per-bout Viterbi paths (boundary-safe).
        """
        if self.state_sequences is None:
            raise RuntimeError("Run infer_states_with_posteriors first.")

        counts = np.zeros(self.n_state, dtype=int)
        total = 0
        for z in self.state_sequences:
            counts += np.bincount(z, minlength=self.n_state)
            total += len(z)
        return counts / max(total, 1)

    def state_dwell_times(self):
        """
        Dwell times per state pooled across sequences (boundary-safe).
        """
        if self.state_sequences is None:
            raise RuntimeError("Run infer_states_with_posteriors first.")

        dwell = {k: [] for k in range(self.n_state)}
        for z in self.state_sequences:
            prev = z[0]
            run = 1
            for curr in z[1:]:
                if curr == prev:
                    run += 1
                else:
                    dwell[prev].append(run)
                    prev = curr
                    run = 1
            dwell[prev].append(run)
        return dwell

    # ---------- Existing-style QC ----------

    def check_convergence(self):
        if self.train_lls is None:
            raise RuntimeError("Model not fit yet")
        return {
            "initial_ll": float(self.train_lls[0]),
            "final_ll": float(self.train_lls[-1]),
            "delta_ll": float(self.train_lls[-1] - self.train_lls[0]),
            "n_iters": int(len(self.train_lls)),
            "monotonic": bool(np.all(np.diff(self.train_lls) >= -1e-6)),
        }

    def check_transitions(self):
        A = self.model.transitions.transition_matrix
        diag = np.diag(A)
        return {
            "transition_matrix": A,
            "self_transition": diag,
            "mean_self_transition": float(diag.mean()),
            "dead_states": np.where(A.sum(axis=1) == 0)[0],
        }

    def get_emission_summary(self):
        means = self.model.observations.params[0]  # (K, D)
        covs = self.model.observations.params[1]   # (K, D, D)
        variances = np.stack([np.diag(c) for c in covs])
        return {
            "means": means,
            "variances": variances,
            "min_variance": float(variances.min()),
            "max_variance": float(variances.max()),
            "feature_names": self._feature_list(),
        }

    # ---------- Save / load ----------

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(
                {
                    "model": self.model,
                    "n_state": self.n_state,
                    "mouse_col": self.mouse_col,
                    "trial_col": self.trial_col,
                    "base_features": self.base_features,
                    "add_derivatives": self.add_derivatives,
                    "derivative_cols": self.derivative_cols,
                    "zscore": self.zscore,
                    "zscore_group_cols": self.zscore_group_cols,
                    "transitions_kappa": self.transitions_kappa,
                    "transitions_alpha": self.transitions_alpha,
                    "parameters": self.parameters,
                    "transition_mat": self.transition_mat,
                    "train_lls": self.train_lls,
                    "test_lls": self.test_lls,
                },
                fh,
            )

    @classmethod
    def load(cls, path):
        data = pickle.load(open(path, "rb"))
        obj = cls(
            n_state=data["n_state"],
            mouse_col=data.get("mouse_col", "mouse_id"),
            trial_col=data.get("trial_col", "trial"),
            transitions_kappa=data.get("transitions_kappa", 5.0),
            transitions_alpha=data.get("transitions_alpha", 1.0),
            add_derivatives=data.get("add_derivatives", True),
            derivative_cols=data.get("derivative_cols", None),
            zscore=data.get("zscore", False),
            zscore_group_cols=data.get("zscore_group_cols", None),
        )
        obj.model = data.get("model")
        obj.base_features = data.get("base_features", obj.base_features)
        obj.parameters = data.get("parameters")
        obj.transition_mat = data.get("transition_mat")
        obj.train_lls = data.get("train_lls")
        obj.test_lls = data.get("test_lls")
        return obj



def fit_model_hmm(
    concat_data_path,
    output_path,
    restarts=5,
    Ks=range(2, 11),
    num_iters=200,
    transitions_kappa=5.0,
    transitions_alpha=1.0,
    add_derivatives=True,
    derivative_cols=None,
    zscore=False,                      # default False to avoid changing your pipeline
    zscore_group_cols=("mouse_id",),   # only used if zscore=True
    init_method="kmeans",
):
    """
    Minimal pipeline changes:
      - no more unused lag columns
      - best model tracked per K (fixes global-best saving bug)
      - still trains on (mouse_id, trial) sequences internally
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(concat_data_path)

    # Keep IDs + base features; AeonHMM adds derivatives internally
    base = AeonHMM(n_state=2).base_features
    required = ["mouse_id", "trial"] + base
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df = df[required].copy()

    results = []

    for K in Ks:
        print(f"Fitting HMM with K={K} states...")

        best_ll_K = -np.inf
        best_K = None

        for seed in range(restarts):
            npr.seed(seed)

            h = AeonHMM(
                n_state=K,
                transitions_kappa=transitions_kappa,
                transitions_alpha=transitions_alpha,
                add_derivatives=add_derivatives,
                derivative_cols=derivative_cols,
                zscore=zscore,
                zscore_group_cols=list(zscore_group_cols) if zscore_group_cols else None,
            )

            h.fit_model(df, num_iters=num_iters, init_method=init_method)
            final_ll = float(h.train_lls[-1])

            results.append(
                {
                    "K": K,
                    "seed": seed,
                    "final_train_ll": final_ll,
                    "n_iters": len(h.train_lls),
                    "monotonic": h.check_convergence()["monotonic"],
                }
            )

            if final_ll > best_ll_K:
                best_ll_K = final_ll
                best_K = h

        model_file = output_path / f"best_hmm_K={K}.pkl"
        best_K.save(model_file)
        print(f"  Saved best K={K} model to {model_file} (train LL={best_ll_K:.3f})")

    # Optional: summary CSV (harmless; remove if you don't want extra outputs)
    summary_df = pd.DataFrame(results).sort_values(["K", "final_train_ll"], ascending=[True, False])
    summary_df.to_csv(output_path / "hmm_fit_summary_by_K_and_seed.csv", index=False)

    return summary_df

def plot_transition_matrix(trans, output_path=None):
    matrix = trans['transition_matrix']
    annot_array = np.array([[round(item, 3) for item in row] for row in matrix])
    labels = ["$S_{" + str(i) + "}$" for i in range(len(matrix))]
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

    sns.heatmap(
        matrix,
        cmap=palette,
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

def plot_model_transition_circos(model, output_path=None,
                                  min_prob=0.001,
                                  max_edges_per_node=4,
                                  arrow=True):
    """
    Plots circos-style transition graph using the fitted model's
    transition matrix (not empirical data).
    """
    A = model.model.transitions.transition_matrix
    K = A.shape[0]

    fig, ax = plot_transition_circos(
        A,
        min_prob=min_prob,
        max_edges_per_node=max_edges_per_node,
        arrow=arrow,
        title="Model transition probability"
    )

    if output_path is not None:
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

    return fig, ax

def run_inference_hmm(
    model_dir,
    inference_path,
    k=5,
    model_type="gaussian",
    pattern="*imputed_split_sampling_bouts_v2.csv",
    smooth_posteriors=False,
    posterior_sigma=2.0,
):
    """
    Minimal pipeline changes:
      - reads your usual columns
      - runs inference once
      - outputs SAME NUMBER OF ROWS as input (alignment preserved)
      - keeps your existing downstream hooks
    """
    from scipy.ndimage import gaussian_filter1d

    model_dir = Path(model_dir)
    inference_path = Path(inference_path)

    model_path = model_dir / model_type / f"best_hmm_K={k}.pkl"
    output_dir = inference_path.parent / "model_predictions" / model_type
    output_dir.mkdir(parents=True, exist_ok=True)

    model = AeonHMM.load(model_path)

    qc = model.check_convergence()
    print(qc)

    trans = model.check_transitions()
    print(trans["mean_self_transition"])

    # Model-based plots (unchanged)
    plot_transition_matrix(trans, output_path=(output_dir / f"transition_matrix_K={k}.svg"))
    plot_model_transition_circos(model, output_path=(output_dir / f"transition_circos_K={k}.svg"))

    inference_list = inference_path.rglob(pattern)

    for path in inference_list:
        print(f"Running inference on {path}...")

        infer_data = pd.read_csv(path)

        required = ["mouse_id", "trial"] + model.base_features
        missing = [c for c in required if c not in infer_data.columns]
        if missing:
            print(f"  Skipping (missing columns): {missing}")
            continue

        infer_small = infer_data[required].copy()

        # Inference (alignment-preserving)
        df_prepped = model.infer_states_with_posteriors(infer_small)

        # Attach to the ORIGINAL infer_small rows (same length/order)
        infer_small["states"] = model.states

        # confidence: NaN where posteriors are NaN (excluded rows)
        conf = np.nanmax(model.posteriors, axis=1)
        infer_small["state_confidence"] = conf

        if smooth_posteriors:
            # Smooth only where posteriors exist; keep excluded rows as -1/NaN
            post = model.posteriors.copy()
            valid_rows = np.isfinite(post).all(axis=1)

            sm = post.copy()
            sm[~valid_rows] = 0.0
            sm = gaussian_filter1d(sm, sigma=posterior_sigma, axis=0, mode="nearest")

            # Renormalize only on valid rows
            row_sums = sm.sum(axis=1, keepdims=True)
            sm[valid_rows] = sm[valid_rows] / (row_sums[valid_rows] + 1e-12)

            states_sm = np.full(len(infer_small), -1, dtype=int)
            conf_sm = np.full(len(infer_small), np.nan, dtype=float)

            states_sm[valid_rows] = sm[valid_rows].argmax(axis=1)
            conf_sm[valid_rows] = sm[valid_rows].max(axis=1)

            infer_small["states_smoothed"] = states_sm
            infer_small["state_confidence_smoothed"] = conf_sm
        else:
            infer_small["states_smoothed"] = infer_small["states"]
            infer_small["state_confidence_smoothed"] = infer_small["state_confidence"]

        # If you want derivative columns saved too (optional, low impact):
        # merge df_prepped derivative cols back in by index (they align 1:1)
        # This keeps your outputs richer without changing row count.
        if model.add_derivatives:
            for c in model.derivative_cols:
                dc = f"d_{c}"
                infer_small[dc] = df_prepped[dc].values

        output_path = output_dir / f"hmm_inferred_states_K={k}.csv"
        infer_small.to_csv(output_path, index=False)

        # Your existing hook (unchanged)
        add_states_to_df(output_dir, inference_path, k=k)

        # Your existing sorting/plotting block (kept, but note: you were sorting after inference)
        try:
            state_mean_speed = model.model.observations.params[0].T[0]
            sort_idx = np.argsort(state_mean_speed, -1)
            # model.sort(sort_idx=sort_idx)  # only if you still have sort() implemented in your class
        except Exception as e:
            print("Warning: sort/plot block skipped:", e)

        print(f"Saved inferred states to {output_path}")
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

#fit_model_hmm(rf"F:\social_sniffing\derivatives\test_concat_hmm_features_social_CTRLall_mice_doublefeature.csv", output_path=rf"F:\hmm\models\social\gaussian_raw_fixed", restarts=2, Ks=range(4,6))
run_inference_hmm(model_dir=Path(r"F:\hmm\models\social_all_mice"), k=5, model_type='gaussian_raw', inference_path=Path(rf"F:\social_sniffing\derivatives\1106077\social\CTRL\hmm_combined"))
