import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
palette = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

def plot_transition_circos(P, state_colors=None, min_prob=0.05, max_edges_per_node=4,
                           edge_cmap="Greys", edge_width=(0.5, 6.0),
                           node_size=900, figsize=(8,8), title="Transition probability",
                           arrow=True):
    P = np.asarray(P, float)
    K = P.shape[0]

    angles = np.linspace(0, 2*np.pi, K, endpoint=False)

    # start at top (pi/2) and go clockwise (negative direction)
    angles = (np.pi/2) - angles

    pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

    # node colors
    if state_colors is None:
        node_cmap = cm.get_cmap("Spectral", K)
        state_colors = [node_cmap(i) for i in range(K)]

    # collect edges
    edges = []
    for i in range(K):
        probs = P[i].copy()
        probs[i] = 0.0  # hide self transitions in this plot (optional)

        js = np.where(probs >= min_prob)[0]
        if max_edges_per_node is not None and len(js) > 0:
            js = js[np.argsort(probs[js])[::-1][:max_edges_per_node]]

        for j in js:
            edges.append((i, j, probs[j]))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    if edges:
        w = np.array([e[2] for e in edges])
        norm = colors.Normalize(vmin=min_prob, vmax=w.max())
    else:
        norm = colors.Normalize(vmin=min_prob, vmax=min_prob + 1e-6)

    cmap = cm.get_cmap(edge_cmap)

    wmin, wmax = edge_width
    def lw(p):
        return wmin + (wmax - wmin) * norm(p)

    # draw edges (curved)
    for i, j, p in edges:
        (x1, y1) = pos[i]
        (x2, y2) = pos[j]

        # curvature heuristic
        dist = (j - i) % K
        rad = 0.15 if dist in (1, K-1) else 0.30

        patch = FancyArrowPatch(
                (x1, y1), (x2, y2),
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=18,      # bigger arrowhead
                linewidth=lw(p),
                color=cmap(norm(p)),
                alpha=0.9,
                shrinkA=12,             # pull start away from node
                shrinkB=12,             # pull end away from node so arrowhead is visible
                zorder=4                # above nodes/labels
            )
        ax.add_patch(patch)

    # draw nodes + labels
    for i in range(K):
        x, y = pos[i]
        ax.scatter([x], [y], s=node_size, color=state_colors[i],
                   edgecolor="k", linewidth=1.5, zorder=2)
        ax.text(x, y, str(i), ha="center", va="center", color="k",
                fontsize=15, zorder=3)

    # # colorbar
    # sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])
    # cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label(title)

    return fig, ax

def empirical_transition_matrix_by_trial(df, state_col, trial_col, K, drop_self=False):
    counts = np.zeros((K, K), dtype=float)

    for _, g in df.groupby(trial_col, sort=False):
        s = g[state_col].to_numpy()
        s = s[s >= 0]
        for a, b in zip(s[:-1], s[1:]):
            if drop_self and a == b:
                continue
            counts[int(a), int(b)] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return P, counts

def empirical_transition_matrix_boutwise_by_trial(df, state_col, trial_col, K):
    """
    Boutwise transitions: compress consecutive identical states into bouts,
    then count transitions between consecutive bouts (per trial).
    """
    counts = np.zeros((K, K), dtype=float)

    for _, g in df.groupby(trial_col, sort=False):
        s = g[state_col].to_numpy()
        s = s[s >= 0].astype(int)
        if len(s) < 2:
            continue

        # run-length compress => bout states
        change_idx = np.flatnonzero(s[1:] != s[:-1]) + 1
        bouts = np.split(s, change_idx)
        bout_states = np.array([b[0] for b in bouts], dtype=int)

        for a, b in zip(bout_states[:-1], bout_states[1:]):
            counts[a, b] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return P, counts

def plot_transition_matrix_no_self(
    P,
    *,
    state_labels=None,
    cmap=palette,
    vmax=None,
    annotate=True,
    figsize=(6, 5),
    title="Transition probabilities (boutwise)",
):
    """
    P: (K x K) transition probability matrix
    Diagonal assumed to represent self-transitions (set to black visually)
    """

    K = P.shape[0]
    P_plot = P.copy()

    # mask diagonal for plotting
    mask = np.eye(K, dtype=bool)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        P_plot,
        mask=mask,
        cmap=cmap,
        vmin=0,
        vmax=vmax if vmax is not None else np.nanmax(P_plot),
        annot=annotate,
        fmt=".2f",
        square=True,
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )

    # draw black squares on diagonal
    for i in range(K):
        ax.add_patch(
            plt.Rectangle((i, i), 1, 1, fill=True, color="black", zorder=3)
        )

    if state_labels is None:
        state_labels = [f"S{i}" for i in range(K)]

    ax.set_xticks(np.arange(K) + 0.5)
    ax.set_yticks(np.arange(K) + 0.5)
    ax.set_xticklabels(state_labels)
    ax.set_yticklabels(state_labels, rotation=0)

    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax

def plot_emission_means_line(
    hmm: "AeonHMM",
    feature_labels=None,
    scales=None,
    ci=0.95,
    figsize=(5, 4),
    title="Emission means by state",
    palette=None,
    savepath=None,
):
    """
    Line plot of emission means across features for each state.
    Assumes hmm.parameters = [means(D,K), vars(D,K), covs(K,D,D)]
    """

    if hmm.parameters is None:
        raise ValueError("hmm.parameters is None. Run hmm.sort(sort_idx) first.")

    means = np.asarray(hmm.parameters[0])  # (D, K)
    vars_ = np.asarray(hmm.parameters[1])  # (D, K)

    D, K = means.shape

    if feature_labels is None:
        feature_labels = getattr(hmm, "features", [f"Feature {i}" for i in range(D)])
    if len(feature_labels) != D:
        raise ValueError("feature_labels length mismatch.")

    if scales is None:
        scales = np.ones(D)
    scales = np.asarray(scales)

    # z for CI
    z = 1.96 if ci == 0.95 else {0.90: 1.645, 0.99: 2.576}.get(ci, 1.96)

    # default palette
    if palette is None:
        palette = sns.color_palette("mako", K)

    x = np.arange(D)

    fig, ax = plt.subplots(figsize=figsize)

    for k in range(K):
        m = means[:, k] * scales
        se = np.sqrt(vars_[:, k]) * scales
        yerr = z * se

        ax.plot(x, m, marker="o",markersize=7,
                 linewidth=1, color=palette[k],label=f"State {k}")
        # ax.errorbar(
        #     x, m,
        #     yerr=yerr,
        #     fmt='-o',
        #     linewidth=1,
        #     capsize=3,
        #     color=palette[k],
        #     label=f"State {k}"
        # )

    ax.axhline(0, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=45, ha="right")
    ax.set_ylabel("Emission mean (scaled)")
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, title="State")

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300)
        plt.close(fig)

    return fig, ax

