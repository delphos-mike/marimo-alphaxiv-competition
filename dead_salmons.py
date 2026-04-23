# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo>=0.16.5",
#     "numpy>=2.2",
#     "scipy>=1.14",
#     "scikit-learn>=1.7",
#     "matplotlib>=3.8",
#     "altair>=6.0",
#     "pandas>=2.3",
# ]
# ///

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import altair as alt
    import pandas as pd
    from scipy import stats
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from importlib.util import find_spec

    IS_WASM = find_spec("js") is not None
    return (
        LogisticRegression,
        MLPClassifier,
        PCA,
        alt,
        load_digits,
        mo,
        np,
        pd,
        plt,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """# The Dead Salmons of AI Interpretability

    Can interpretability methods find 'signal' in random networks?
    A cautionary tale about false discoveries in AI interpretability.

    **Paper**: Meloux et al. 2025
    ([arXiv](https://arxiv.org/abs/2512.18792)) |
    [AlphaXiv](https://alphaxiv.org/abs/2512.18792)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## 2. Trained vs Random Networks

    We compare two MLPs with identical architecture `(64, 32)`:

    - **Trained MLP**: Fitted on real digit labels (learns genuine features)
    - **Random MLP**: Fitted on shuffled labels for 1 iteration (no real learning)

    If interpretability methods are sound, they should find meaningful structure
    only in the trained network. The "dead salmon" hypothesis: naive methods
    will find spurious signal even in the random network.
    """
    )
    return


@app.cell
def _(MLPClassifier, load_digits, np, train_test_split):
    digits = load_digits()
    X_digits = digits.data
    y_digits = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X_digits, y_digits, test_size=0.2, random_state=42
    )

    trained_mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=300, random_state=42
    )
    trained_mlp.fit(X_train, y_train)
    trained_acc = trained_mlp.score(X_test, y_test)

    rng_shuffle = np.random.RandomState(99)
    y_train_shuffled = rng_shuffle.permutation(y_train)
    random_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1, random_state=99)
    random_mlp.fit(X_train, y_train_shuffled)
    random_acc = random_mlp.score(X_test, y_test)
    return X_test, random_acc, random_mlp, trained_acc, trained_mlp, y_test


@app.cell
def _(X_test, np, random_mlp, trained_mlp):
    def get_activations(mlp, X):
        layer_activations = []
        a = X
        for i, (w, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
            a = a @ w + b
            if i < len(mlp.coefs_) - 1:
                a = np.maximum(a, 0)
                layer_activations.append(a.copy())
        return layer_activations

    trained_acts = get_activations(trained_mlp, X_test)
    random_acts = get_activations(random_mlp, X_test)
    return random_acts, trained_acts


@app.cell(hide_code=True)
def _(mo, random_acc, trained_acc):
    mo.md(
        f"""### Network Performance

    | Network | Test Accuracy |
    |---------|--------------|
    | **Trained MLP** | {trained_acc:.1%} |
    | **Random MLP** | {random_acc:.1%} |

    The trained network learns real digit features.
    The random network performs near chance (~10% for 10 classes).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## 3. Probing Analysis

    Linear probes (logistic regression) attempt to decode class labels from
    hidden activations. A valid interpretability signal should appear only in
    networks that learned meaningful representations.

    **Dead salmon effect**: Even random network activations allow above-chance
    probing accuracy, because high-dimensional projections create spurious
    linear separability.
    """
    )
    return


@app.cell
def _(mo):
    layer_dropdown = mo.ui.dropdown(
        options=["Layer 1 (64 units)", "Layer 2 (32 units)"],
        value="Layer 1 (64 units)",
        label="Hidden layer to examine",
    )
    layer_dropdown
    return (layer_dropdown,)


@app.cell
def _(LogisticRegression, layer_dropdown, random_acts, trained_acts, y_test):
    probe_results = []
    for layer_idx in range(2):
        layer_name = f"Layer {layer_idx + 1}"

        probe_trained = LogisticRegression(max_iter=500, n_jobs=1)
        probe_trained.fit(trained_acts[layer_idx], y_test)
        acc_trained = probe_trained.score(trained_acts[layer_idx], y_test)

        probe_random = LogisticRegression(max_iter=500, n_jobs=1)
        probe_random.fit(random_acts[layer_idx], y_test)
        acc_random = probe_random.score(random_acts[layer_idx], y_test)

        probe_results.append(
            {
                "layer": layer_name,
                "trained_acc": acc_trained,
                "random_acc": acc_random,
            }
        )

    _layer_map = {"Layer 1 (64 units)": 0, "Layer 2 (32 units)": 1}
    selected_layer_idx = _layer_map.get(layer_dropdown.value, 0)
    return probe_results, selected_layer_idx


@app.cell
def _(alt, pd, probe_results):
    probe_df = pd.DataFrame(
        [
            {
                "Layer": r["layer"],
                "Network": "Trained",
                "Probe Accuracy": r["trained_acc"],
            }
            for r in probe_results
        ]
        + [
            {
                "Layer": r["layer"],
                "Network": "Random",
                "Probe Accuracy": r["random_acc"],
            }
            for r in probe_results
        ]
    )

    chance_line = (
        alt.Chart(pd.DataFrame({"y": [0.1]}))
        .mark_rule(strokeDash=[4, 4], color="red")
        .encode(y="y:Q")
    )

    bars = (
        alt.Chart(probe_df)
        .mark_bar()
        .encode(
            x=alt.X("Network:N", title=None),
            y=alt.Y(
                "Probe Accuracy:Q",
                scale=alt.Scale(domain=[0, 1]),
                title="Probe Accuracy",
            ),
            color=alt.Color(
                "Network:N",
                scale=alt.Scale(
                    domain=["Trained", "Random"], range=["#2563eb", "#dc2626"]
                ),
            ),
            column=alt.Column("Layer:N", title="Hidden Layer"),
            tooltip=["Network:N", alt.Tooltip("Probe Accuracy:Q", format=".1%")],
        )
        .properties(
            width=160, height=300, title="Linear Probe Accuracy: Trained vs Random"
        )
    )

    bars + chance_line
    return


@app.cell(hide_code=True)
def _(mo, probe_results):
    mo.md(
        f"""### Probing Results

    | Layer | Trained Probe Acc | Random Probe Acc |
    |-------|-------------------|------------------|
    | Layer 1 | {probe_results[0]["trained_acc"]:.1%} | {probe_results[0]["random_acc"]:.1%} |
    | Layer 2 | {probe_results[1]["trained_acc"]:.1%} | {probe_results[1]["random_acc"]:.1%} |

    The random network's activations yield above-chance probing accuracy
    (well above 10%), despite learning nothing meaningful. This is the
    **dead salmon effect** -- high-dimensional random projections create
    spurious linear separability.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## 4. PCA Correlation Analysis

    We compute PCA on each network's hidden activations and correlate the
    principal components with class labels. A valid method should show strong,
    structured correlations only for the trained network.

    **Watch for**: The random network produces visually convincing but
    entirely spurious correlation patterns.
    """
    )
    return


@app.cell
def _(mo):
    n_pcs_slider = mo.ui.slider(
        start=5,
        stop=20,
        value=10,
        step=1,
        label="Number of PCs to display",
    )
    n_pcs_slider
    return (n_pcs_slider,)


@app.cell
def _(
    PCA,
    n_pcs_slider,
    np,
    plt,
    random_acts,
    selected_layer_idx,
    trained_acts,
    y_test,
):
    n_pcs = n_pcs_slider.value
    n_classes = 10

    labels_onehot = np.zeros((len(y_test), n_classes))
    labels_onehot[np.arange(len(y_test)), y_test] = 1.0

    def compute_pc_correlations(activations, n_components):
        pca = PCA(n_components=n_components)
        pcs = pca.fit_transform(activations)
        corr = np.zeros((n_components, n_classes))
        for pc_i in range(n_components):
            for cls_j in range(n_classes):
                corr[pc_i, cls_j] = np.corrcoef(pcs[:, pc_i], labels_onehot[:, cls_j])[
                    0, 1
                ]
        return corr

    corr_trained = compute_pc_correlations(trained_acts[selected_layer_idx], n_pcs)
    corr_random = compute_pc_correlations(random_acts[selected_layer_idx], n_pcs)

    vmax = max(np.abs(corr_trained).max(), np.abs(corr_random).max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    im1 = ax1.imshow(
        corr_trained.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto"
    )
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Digit Class")
    ax1.set_title("Trained Network")
    ax1.set_xticks(range(n_pcs))
    ax1.set_xticklabels([f"PC{i + 1}" for i in range(n_pcs)], rotation=45, fontsize=7)
    ax1.set_yticks(range(n_classes))

    im2 = ax2.imshow(corr_random.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Digit Class")
    ax2.set_title("Random Network")
    ax2.set_xticks(range(n_pcs))
    ax2.set_xticklabels([f"PC{i + 1}" for i in range(n_pcs)], rotation=45, fontsize=7)
    ax2.set_yticks(range(n_classes))

    fig.colorbar(im2, ax=[ax1, ax2], label="Correlation", shrink=0.8)
    fig.suptitle(
        f"PC-Class Correlations (Layer {selected_layer_idx + 1})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """Both heatmaps show non-trivial correlation patterns, but the random
    network's correlations are **entirely spurious** -- an artifact of
    projecting high-dimensional random data onto principal components.
    Without statistical testing, these patterns are indistinguishable
    from real signal.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## 5. The Permutation Test Solution

    A Monte Carlo permutation test provides the statistical rigor to
    distinguish real signal from noise:

    1. Compute the **observed** probe accuracy
    2. Repeatedly shuffle labels and re-train probes to build a **null distribution**
    3. The **p-value** = fraction of permuted accuracies >= observed accuracy

    If p < 0.05, the observed accuracy is unlikely under the null hypothesis
    of no real structure.
    """
    )
    return


@app.cell
def _(mo):
    n_perms_slider = mo.ui.slider(
        start=20,
        stop=100,
        value=50,
        step=10,
        label="Number of permutations",
    )
    n_perms_slider
    return (n_perms_slider,)


@app.cell
def _(
    LogisticRegression,
    n_perms_slider,
    np,
    random_acts,
    selected_layer_idx,
    trained_acts,
    y_test,
):
    n_perms = n_perms_slider.value
    layer_for_perm = selected_layer_idx

    obs_probe_trained = LogisticRegression(max_iter=500, n_jobs=1)
    obs_probe_trained.fit(trained_acts[layer_for_perm], y_test)
    obs_acc_trained = obs_probe_trained.score(trained_acts[layer_for_perm], y_test)

    obs_probe_random = LogisticRegression(max_iter=500, n_jobs=1)
    obs_probe_random.fit(random_acts[layer_for_perm], y_test)
    obs_acc_random = obs_probe_random.score(random_acts[layer_for_perm], y_test)

    perm_rng = np.random.RandomState(42)
    null_trained = np.zeros(n_perms)
    null_random = np.zeros(n_perms)

    for perm_i in range(n_perms):
        y_perm = perm_rng.permutation(y_test)

        probe_t = LogisticRegression(max_iter=500, n_jobs=1)
        probe_t.fit(trained_acts[layer_for_perm], y_perm)
        null_trained[perm_i] = probe_t.score(trained_acts[layer_for_perm], y_perm)

        probe_r = LogisticRegression(max_iter=500, n_jobs=1)
        probe_r.fit(random_acts[layer_for_perm], y_perm)
        null_random[perm_i] = probe_r.score(random_acts[layer_for_perm], y_perm)

    p_value_trained = np.mean(null_trained >= obs_acc_trained)
    p_value_random = np.mean(null_random >= obs_acc_random)
    return (
        null_random,
        null_trained,
        obs_acc_random,
        obs_acc_trained,
        p_value_random,
        p_value_trained,
    )


@app.cell
def _(
    np,
    null_random,
    null_trained,
    obs_acc_random,
    obs_acc_trained,
    p_value_random,
    p_value_trained,
    plt,
    selected_layer_idx,
):
    fig_perm, (ax_t, ax_r) = plt.subplots(1, 2, figsize=(12, 4))

    ax_t.hist(null_trained, bins=15, color="#93c5fd", edgecolor="white", alpha=0.8)
    ax_t.axvline(
        obs_acc_trained,
        color="#2563eb",
        linewidth=2,
        linestyle="--",
        label=f"Observed: {obs_acc_trained:.1%}",
    )
    ax_t.set_xlabel("Probe Accuracy")
    ax_t.set_ylabel("Count")
    ax_t.set_title(f"Trained Network (p={p_value_trained:.3f})")
    ax_t.legend(fontsize=9)

    ax_r.hist(null_random, bins=15, color="#fca5a5", edgecolor="white", alpha=0.8)
    ax_r.axvline(
        obs_acc_random,
        color="#dc2626",
        linewidth=2,
        linestyle="--",
        label=f"Observed: {obs_acc_random:.1%}",
    )
    ax_r.set_xlabel("Probe Accuracy")
    ax_r.set_ylabel("Count")
    ax_r.set_title(f"Random Network (p={p_value_random:.3f})")
    ax_r.legend(fontsize=9)

    all_vals = np.concatenate(
        [null_trained, null_random, [obs_acc_trained, obs_acc_random]]
    )
    shared_min = all_vals.min() - 0.02
    shared_max = all_vals.max() + 0.02
    ax_t.set_xlim(shared_min, shared_max)
    ax_r.set_xlim(shared_min, shared_max)

    fig_perm.suptitle(
        f"Permutation Test: Null Distribution vs Observed (Layer {selected_layer_idx + 1})",
        fontsize=13,
        fontweight="bold",
    )
    fig_perm.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo, obs_acc_random, obs_acc_trained, p_value_random, p_value_trained):
    trained_sig = (
        "**significant** (p < 0.05)"
        if p_value_trained < 0.05
        else "not significant (p >= 0.05)"
    )
    random_sig = (
        "**significant** (p < 0.05)"
        if p_value_random < 0.05
        else "not significant (p >= 0.05)"
    )
    mo.md(
        f"""### Permutation Test Results

    | Network | Observed Acc | p-value | Significant? |
    |---------|-------------|---------|-------------|
    | **Trained** | {obs_acc_trained:.1%} | {p_value_trained:.4f} | {trained_sig} |
    | **Random** | {obs_acc_random:.1%} | {p_value_random:.4f} | {random_sig} |

    The trained network's probe accuracy is far outside the null distribution,
    confirming genuine learned structure. The random network's accuracy falls
    within the null distribution -- its apparent "signal" was a statistical
    artifact all along.

    **Takeaway**: Permutation tests are essential for validating
    interpretability findings. Without them, we risk interpreting noise as signal
    -- the dead salmon effect of AI interpretability.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## 6. Extension — The Dead Salmon Detector

    A reusable diagnostic tool that packages the permutation test
    into an interactive widget. Select any interpretability method,
    layer, and network type, then run the statistical test to get a
    verdict: is the observed signal real, or is it a dead salmon?

    The detector computes a **test statistic** (probe accuracy for
    probing, max |correlation| for PCA), builds a null distribution
    via label permutation, and reports a p-value with a color-coded
    verdict.
    """
    )
    return


@app.cell
def _(mo):
    det_method_dd = mo.ui.dropdown(
        options=["Probing", "PCA Correlation"],
        value="Probing",
        label="Method",
    )
    det_layer_dd = mo.ui.dropdown(
        options=["Layer 1 (64 units)", "Layer 2 (32 units)"],
        value="Layer 1 (64 units)",
        label="Layer",
    )
    det_network_dd = mo.ui.dropdown(
        options=["Trained", "Random"],
        value="Trained",
        label="Network",
    )
    det_alpha_slider = mo.ui.slider(
        start=0.01,
        stop=0.10,
        value=0.05,
        step=0.01,
        label="Significance threshold (α)",
    )
    det_perms_slider = mo.ui.slider(
        start=20,
        stop=100,
        value=50,
        step=10,
        label="Permutations",
    )
    mo.vstack(
        [
            mo.hstack(
                [det_method_dd, det_layer_dd, det_network_dd],
                justify="start",
            ),
            mo.hstack(
                [det_alpha_slider, det_perms_slider],
                justify="start",
            ),
        ]
    )
    return (
        det_alpha_slider,
        det_layer_dd,
        det_method_dd,
        det_network_dd,
        det_perms_slider,
    )


@app.cell
def _(
    LogisticRegression,
    PCA,
    det_alpha_slider,
    det_layer_dd,
    det_method_dd,
    det_network_dd,
    det_perms_slider,
    np,
    random_acts,
    trained_acts,
    y_test,
):
    def run_detector(activations, labels, method, n_perms, rng_seed=42):
        rng = np.random.RandomState(rng_seed)
        n_classes = len(np.unique(labels))

        if method == "Probing":
            probe = LogisticRegression(max_iter=500, n_jobs=1)
            probe.fit(activations, labels)
            observed = probe.score(activations, labels)
            null_dist = np.zeros(n_perms)
            for i in range(n_perms):
                y_perm = rng.permutation(labels)
                p = LogisticRegression(max_iter=500, n_jobs=1)
                p.fit(activations, y_perm)
                null_dist[i] = p.score(activations, y_perm)
        else:
            labels_oh = np.zeros((len(labels), n_classes))
            labels_oh[np.arange(len(labels)), labels] = 1.0
            pca = PCA(n_components=min(5, activations.shape[1]))
            pcs = pca.fit_transform(activations)
            corr_mat = np.array(
                [
                    [
                        np.corrcoef(pcs[:, j], labels_oh[:, k])[0, 1]
                        for k in range(n_classes)
                    ]
                    for j in range(pcs.shape[1])
                ]
            )
            observed = float(np.abs(corr_mat).max())
            null_dist = np.zeros(n_perms)
            for i in range(n_perms):
                y_perm = rng.permutation(labels)
                oh_perm = np.zeros((len(y_perm), n_classes))
                oh_perm[np.arange(len(y_perm)), y_perm] = 1.0
                c_perm = np.array(
                    [
                        [
                            np.corrcoef(pcs[:, j], oh_perm[:, k])[0, 1]
                            for k in range(n_classes)
                        ]
                        for j in range(pcs.shape[1])
                    ]
                )
                null_dist[i] = float(np.abs(c_perm).max())

        p_val = float(np.mean(null_dist >= observed))
        return observed, null_dist, p_val

    _det_layer_map = {"Layer 1 (64 units)": 0, "Layer 2 (32 units)": 1}
    _layer_idx = _det_layer_map.get(det_layer_dd.value, 0)
    _acts = (
        trained_acts[_layer_idx]
        if det_network_dd.value == "Trained"
        else random_acts[_layer_idx]
    )
    _alpha = det_alpha_slider.value

    det_observed, det_null_dist, det_p_value = run_detector(
        _acts, y_test, det_method_dd.value, det_perms_slider.value
    )

    if det_p_value < _alpha:
        det_verdict = "REAL SIGNAL"
    elif det_p_value > 2 * _alpha:
        det_verdict = "DEAD SALMON"
    else:
        det_verdict = "INCONCLUSIVE"
    return det_null_dist, det_observed, det_p_value, det_verdict, run_detector


@app.cell
def _(
    det_alpha_slider,
    det_layer_dd,
    det_method_dd,
    det_network_dd,
    det_null_dist,
    det_observed,
    det_p_value,
    det_verdict,
    mo,
    plt,
):
    _style_map = {
        "REAL SIGNAL": ("#065f46", "#d1fae5", "🟢"),
        "DEAD SALMON": ("#991b1b", "#fee2e2", "🐟"),
        "INCONCLUSIVE": ("#92400e", "#fef3c7", "🟡"),
    }
    _fg, _bg, _icon = _style_map[det_verdict]

    _badge = mo.md(
        f'<div style="text-align:center; padding:16px; margin:8px 0;'
        f" background:{_bg}; border-radius:12px;"
        f' border:2px solid {_fg};">'
        f'<span style="font-size:2em;">{_icon}</span><br>'
        f'<span style="font-size:1.5em; font-weight:bold;'
        f' color:{_fg};">{det_verdict}</span><br>'
        f'<span style="color:{_fg};">'
        f"{det_network_dd.value}"
        f" · Layer {_det_layer_map.get(det_layer_dd.value, 0) + 1}"
        f" · {det_method_dd.value}<br>"
        f"p = {det_p_value:.4f}"
        f" | α = {det_alpha_slider.value}</span></div>"
    )

    _fig, _ax = plt.subplots(figsize=(8, 3.5))
    _ax.hist(
        det_null_dist,
        bins=15,
        color="#cbd5e1",
        edgecolor="white",
        alpha=0.9,
        label="Null distribution",
    )
    _ax.axvline(
        det_observed,
        color=_fg,
        linewidth=2.5,
        linestyle="--",
        label=f"Observed: {det_observed:.4f}",
    )
    _ax.set_xlabel("Test Statistic")
    _ax.set_ylabel("Count")
    _ax.set_title("Null Distribution vs Observed Statistic")
    _ax.legend(fontsize=9)
    _fig.tight_layout()

    mo.vstack([_badge, _fig])
    return


@app.cell
def _(
    det_alpha_slider,
    det_perms_slider,
    mo,
    pd,
    random_acts,
    run_detector,
    trained_acts,
    y_test,
):
    _alpha = det_alpha_slider.value
    _n_perms = det_perms_slider.value
    _rows = []

    for _net_name, _acts_list in [
        ("Trained", trained_acts),
        ("Random", random_acts),
    ]:
        for _li, _layer_name in [
            (0, "Layer 1"),
            (1, "Layer 2"),
        ]:
            for _method in ["Probing", "PCA Correlation"]:
                _obs, _null, _pv = run_detector(
                    _acts_list[_li],
                    y_test,
                    _method,
                    _n_perms,
                )
                if _pv < _alpha:
                    _verd = "🟢 REAL SIGNAL"
                elif _pv > 2 * _alpha:
                    _verd = "🐟 DEAD SALMON"
                else:
                    _verd = "🟡 INCONCLUSIVE"
                _rows.append(
                    {
                        "Network": _net_name,
                        "Layer": _layer_name,
                        "Method": _method,
                        "p-value": f"{_pv:.4f}",
                        "Verdict": _verd,
                    }
                )

    _summary_df = pd.DataFrame(_rows)

    _html_rows = ""
    for _, _r in _summary_df.iterrows():
        if "REAL SIGNAL" in _r["Verdict"]:
            _row_bg = "#d1fae5"
        elif "DEAD SALMON" in _r["Verdict"]:
            _row_bg = "#fee2e2"
        else:
            _row_bg = "#fef3c7"
        _html_rows += (
            f'<tr style="background:{_row_bg}">'
            f'<td style="padding:6px 8px">'
            f"{_r['Network']}</td>"
            f'<td style="padding:6px 8px">'
            f"{_r['Layer']}</td>"
            f'<td style="padding:6px 8px">'
            f"{_r['Method']}</td>"
            f'<td style="padding:6px 8px">'
            f"{_r['p-value']}</td>"
            f'<td style="padding:6px 8px">'
            f"<strong>{_r['Verdict']}</strong></td>"
            f"</tr>"
        )

    mo.md(
        f"""### All Combinations Summary

    <table style="width:100%; border-collapse:collapse;">
    <thead><tr style="background:#f1f5f9;">
    <th style="padding:8px; text-align:left">Network</th>
    <th style="padding:8px; text-align:left">Layer</th>
    <th style="padding:8px; text-align:left">Method</th>
    <th style="padding:8px; text-align:left">p-value</th>
    <th style="padding:8px; text-align:left">Verdict</th>
    </tr></thead>
    <tbody>{_html_rows}</tbody>
    </table>
    """
    )
    return


if __name__ == "__main__":
    app.run()
