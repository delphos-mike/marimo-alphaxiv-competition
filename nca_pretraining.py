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
    from scipy import stats
    from importlib.util import find_spec

    IS_WASM = find_spec("js") is not None
    return mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """# Training Language Models via Neural Cellular Automata

    Exploring how Neural Cellular Automata generate synthetic data
    with language-like statistical properties for pre-training LLMs.

    **Paper**: Lee et al. 2026
    ([arXiv](https://arxiv.org/abs/2603.10055)) |
    [AlphaXiv](https://alphaxiv.org/abs/2603.10055)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## What is a Neural Cellular Automaton?

    A Neural Cellular Automaton (NCA) is a discrete dynamical system
    on a grid. Each cell holds one of **S** discrete states.
    At every time step, each cell observes its 3×3 neighborhood,
    counts how often each state appears, and picks a new state via
    a softmax-weighted random sample. Periodic boundaries make the
    grid wrap around like a torus.

    The key insight from Lee et al.: sequences read from NCA grids
    exhibit statistical regularities (Zipf distributions, compressible
    structure) similar to natural language — making them useful as
    synthetic pre-training data.
    """
    )
    return


@app.cell(hide_code=True)
def _(np):
    import gzip

    def _softmax(x):
        e = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

    def nca_step(grid, n_states, temperature, rule="complex"):
        n = grid.shape[0]
        counts = np.zeros((n, n, n_states), dtype=np.float32)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                neighbor = np.roll(np.roll(grid, -di, axis=0), -dj, axis=1)
                for s in range(n_states):
                    counts[:, :, s] += (neighbor == s).astype(np.float32)

        if rule == "simple":
            return np.argmax(counts, axis=-1)

        if rule == "periodic":
            majority = np.argmax(counts, axis=-1)
            return (majority + 1) % n_states

        if rule == "chaotic":
            logits = counts / max(temperature, 3.0)
            probs = _softmax(logits)
            flat = probs.reshape(-1, n_states)
            cum = np.cumsum(flat, axis=1)
            r = np.random.rand(flat.shape[0], 1)
            choices = (cum < r).sum(axis=1)
            return np.clip(choices, 0, n_states - 1).reshape(n, n)

        if rule == "stochastic":
            majority = np.argmax(counts, axis=-1)
            mask = np.random.rand(n, n) < 0.3
            random_states = np.random.randint(0, n_states, (n, n))
            result = np.where(mask, random_states, majority)
            return result

        logits = counts / temperature
        probs = _softmax(logits)
        flat = probs.reshape(-1, n_states)
        cum = np.cumsum(flat, axis=1)
        r = np.random.rand(flat.shape[0], 1)
        choices = (cum < r).sum(axis=1)
        return np.clip(choices, 0, n_states - 1).reshape(n, n)

    def run_nca(grid_size, n_states, temperature, steps, rule="complex", seed=42):
        rng = np.random.RandomState(seed)
        grid = rng.randint(0, n_states, (grid_size, grid_size))
        history = [grid.copy()]
        for _ in range(steps):
            grid = nca_step(grid, n_states, temperature, rule)
            history.append(grid.copy())
        return history

    def grid_to_tokens(history):
        return np.concatenate([g.ravel() for g in history])

    def compute_gzip_ratio(tokens):
        data = tokens.astype(np.uint8).tobytes()
        if len(data) == 0:
            return 1.0
        return len(gzip.compress(data)) / len(data)

    def compute_bigram_entropy(tokens, n_states):
        trans = np.zeros((n_states, n_states), dtype=np.float64)
        for i in range(len(tokens) - 1):
            trans[tokens[i], tokens[i + 1]] += 1
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        probs = trans / row_sums
        probs = np.where(probs == 0, 1e-12, probs)
        row_entropy = -(probs * np.log2(probs)).sum(axis=1)
        weights = trans.sum(axis=1)
        total = weights.sum()
        if total == 0:
            return 0.0
        return float((row_entropy * weights).sum() / total)
    return compute_bigram_entropy, compute_gzip_ratio, grid_to_tokens, run_nca


@app.cell(hide_code=True)
def _(plt, run_nca):
    _demo = run_nca(32, 5, 1.0, 20, rule="complex", seed=7)
    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.imshow(_demo[-1], cmap="tab10", vmin=0, vmax=9, interpolation="nearest")
    _ax.set_title("NCA grid after 20 steps (S=5, T=1.0)")
    _ax.set_xticks([])
    _ax.set_yticks([])
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## Interactive NCA Evolution

    Adjust the parameters below to explore different NCA dynamics.
    The **rule family** controls how cells update:

    | Family | Behaviour |
    |--------|-----------|
    | simple | Deterministic majority vote |
    | complex | Softmax sampling with temperature (paper's approach) |
    | chaotic | Very high temperature — nearly random |
    | periodic | Majority + cyclic offset |
    | stochastic | Majority with 30 % random flips |
    """
    )
    return


@app.cell
def _(mo):
    grid_size_slider = mo.ui.slider(16, 64, value=32, label="Grid Size", step=8)
    n_states_slider = mo.ui.slider(2, 10, value=5, label="Number of States")
    temp_slider = mo.ui.slider(0.1, 5.0, value=1.0, label="Temperature", step=0.1)
    steps_slider = mo.ui.slider(10, 200, value=50, label="Evolution Steps", step=10)
    rule_dropdown = mo.ui.dropdown(
        options=["simple", "complex", "chaotic", "periodic", "stochastic"],
        value="simple",
        label="Rule Family",
    )
    mo.hstack(
        [grid_size_slider, n_states_slider, temp_slider, steps_slider, rule_dropdown],
        justify="start",
        gap=1,
    )
    return (
        grid_size_slider,
        n_states_slider,
        rule_dropdown,
        steps_slider,
        temp_slider,
    )


@app.cell(hide_code=True)
def _(
    grid_size_slider,
    n_states_slider,
    np,
    plt,
    rule_dropdown,
    run_nca,
    steps_slider,
    temp_slider,
):
    _gs = grid_size_slider.value
    _ns = n_states_slider.value
    _t = temp_slider.value
    _st = steps_slider.value
    _rule = rule_dropdown.value

    _history = run_nca(_gs, _ns, _t, _st, rule=_rule, seed=42)
    nca_history = _history

    _n_frames = 6
    _indices = np.linspace(0, len(_history) - 1, _n_frames, dtype=int)

    _fig, _axes = plt.subplots(1, _n_frames, figsize=(3 * _n_frames, 3))
    for _i, _idx in enumerate(_indices):
        _axes[_i].imshow(
            _history[_idx], cmap="tab10", vmin=0, vmax=9, interpolation="nearest"
        )
        _axes[_i].set_title(f"t={_idx}")
        _axes[_i].set_xticks([])
        _axes[_i].set_yticks([])
    _fig.suptitle(
        f"NCA Evolution — {_rule} | grid={_gs} states={_ns} T={_t:.1f}",
        fontsize=13,
    )
    _fig.tight_layout()
    plt.gca()
    return (nca_history,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## NCA Produces Language-Like Statistics

    The paper's key claim: reading an NCA grid as a token sequence
    produces statistics surprisingly close to natural language.
    We measure three properties:

    1. **Zipf distribution** — rank-frequency on log-log axes
    2. **Gzip compression ratio** — lower means more structure
    3. **Bigram entropy** — bits per transition; lower means more
       predictable sequences
    """
    )
    return


@app.cell(hide_code=True)
def _(
    compute_bigram_entropy,
    compute_gzip_ratio,
    grid_to_tokens,
    n_states_slider,
    nca_history,
    np,
    plt,
):
    _tokens = grid_to_tokens(nca_history)
    _ns = n_states_slider.value

    _freq = np.bincount(_tokens, minlength=_ns)
    _freq_sorted = np.sort(_freq)[::-1]
    _freq_sorted = _freq_sorted[_freq_sorted > 0]
    _ranks = np.arange(1, len(_freq_sorted) + 1)

    _gzip_r = compute_gzip_ratio(_tokens)
    _bigram_e = compute_bigram_entropy(_tokens, _ns)

    _fig, (_ax1, _ax2, _ax3) = plt.subplots(1, 3, figsize=(15, 4))

    _ax1.loglog(_ranks, _freq_sorted, "o-", color="#1f77b4", label="NCA tokens")
    if len(_ranks) > 1:
        _slope_ref = _freq_sorted[0] / _ranks**1.0
        _ax1.loglog(
            _ranks, _slope_ref, "--", color="gray", alpha=0.6, label="Zipf (α=1)"
        )
    _ax1.set_xlabel("Rank")
    _ax1.set_ylabel("Frequency")
    _ax1.set_title("Zipf Distribution")
    _ax1.legend(fontsize=9)

    _labels = ["NCA", "English\n(ref)"]
    _gzip_vals = [_gzip_r, 0.4]
    _colors = ["#1f77b4", "#aec7e8"]
    _ax2.bar(_labels, _gzip_vals, color=_colors, edgecolor="black", linewidth=0.5)
    _ax2.set_ylabel("Gzip Ratio")
    _ax2.set_title("Compression Ratio")
    _ax2.set_ylim(0, 1.1)
    _ax2.axhspan(0.3, 0.5, alpha=0.15, color="green", label="English range")
    _ax2.legend(fontsize=9)

    _bigram_labels = ["NCA", "English\n(ref)"]
    _bigram_vals = [_bigram_e, 4.0]
    _ax3.bar(
        _bigram_labels, _bigram_vals, color=_colors, edgecolor="black", linewidth=0.5
    )
    _ax3.set_ylabel("Bits")
    _ax3.set_title("Bigram Entropy")
    _ax3.axhspan(3.0, 5.0, alpha=0.15, color="green", label="English range")
    _ax3.legend(fontsize=9)

    nca_gzip_ratio = _gzip_r
    nca_bigram_entropy = _bigram_e
    _fig.tight_layout()
    plt.gca()
    return nca_bigram_entropy, nca_gzip_ratio


@app.cell(hide_code=True)
def _(mo, nca_bigram_entropy, nca_gzip_ratio):
    mo.md(
        f"""**Current NCA statistics:**
    gzip ratio = **{nca_gzip_ratio:.3f}**
    (English ≈ 0.3–0.5) ·
    bigram entropy = **{nca_bigram_entropy:.2f}** bits
    (English ≈ 3–5 bits)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## Complexity vs Domain

    Lee et al. found that NCA-generated data serves different
    downstream tasks depending on its complexity:

    - **Simple dynamics** (majority vote, low entropy) →
      benefits **code** pre-training where repetitive patterns
      and structured syntax dominate.
    - **Complex dynamics** (moderate temperature softmax) →
      benefits **math and web text** where richer statistical
      structure is needed.
    - **Chaotic dynamics** (high temperature) → resembles
      random noise and provides little pre-training value.

    The bar chart below computes the gzip compression ratio for
    each rule family. Lower ratios indicate more structure;
    natural language typically falls in the 0.3–0.5 range.
    """
    )
    return


@app.cell(hide_code=True)
def _(compute_gzip_ratio, grid_to_tokens, np, plt, run_nca):
    _rules = ["simple", "complex", "chaotic", "periodic", "stochastic"]
    _ratios = []
    for _r in _rules:
        _h = run_nca(32, 5, 1.0, 50, rule=_r, seed=42)
        _tok = grid_to_tokens(_h)
        _ratios.append(compute_gzip_ratio(_tok))

    _colors = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#ff7f0e"]
    _fig, _ax = plt.subplots(figsize=(8, 4))
    _bars = _ax.bar(_rules, _ratios, color=_colors, edgecolor="black", linewidth=0.5)
    _ax.axhspan(0.3, 0.5, alpha=0.15, color="green")
    _ax.axhline(0.4, color="green", linestyle="--", alpha=0.5, label="English ~0.4")
    _ax.set_ylabel("Gzip Compression Ratio")
    _ax.set_xlabel("Rule Family")
    _ax.set_title("NCA Complexity by Rule Family")
    _ax.set_ylim(0, np.max(_ratios) * 1.2)
    _ax.legend(fontsize=9)
    for _bar, _val in zip(_bars, _ratios):
        _ax.text(
            _bar.get_x() + _bar.get_width() / 2,
            _val + 0.01,
            f"{_val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    _fig.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## Extension: Which Rule Families Are Most Language-Like?

    The paper used a single NCA variant. We systematically compare all 5 rule
    families across three statistical measures to identify which produces the
    most language-like token sequences.

    The scatter plot below shows each rule family's Zipf exponent vs gzip
    ratio. Points closer to the natural language reference (green star) are
    more language-like.
    """
    )
    return


@app.cell(hide_code=True)
def _(compute_bigram_entropy, compute_gzip_ratio, grid_to_tokens, np, run_nca):
    import pandas as pd

    _rules = ["simple", "complex", "chaotic", "periodic", "stochastic"]
    _rows = []
    for _r in _rules:
        _h = run_nca(32, 5, 1.0, 50, rule=_r, seed=42)
        _tok = grid_to_tokens(_h)
        _gzip = compute_gzip_ratio(_tok)
        _be = compute_bigram_entropy(_tok, 5)
        _freq = np.bincount(_tok, minlength=5)
        _fs = np.sort(_freq)[::-1]
        _fs = _fs[_fs > 0]
        _ranks = np.arange(1, len(_fs) + 1)
        if len(_ranks) > 1:
            _slope = np.polyfit(np.log(_ranks), np.log(_fs), 1)[0]
        else:
            _slope = 0.0
        _rows.append(
            {
                "family": _r,
                "gzip_ratio": round(_gzip, 4),
                "bigram_entropy": round(_be, 4),
                "zipf_exponent": round(_slope, 4),
            }
        )

    family_stats_df = pd.DataFrame(_rows)
    family_stats_df
    return (family_stats_df,)


@app.cell(hide_code=True)
def _(family_stats_df, mo):
    import altair as alt
    import pandas as _pd

    _ref = _pd.DataFrame(
        [
            {
                "family": "English (ref)",
                "gzip_ratio": 0.4,
                "zipf_exponent": -1.0,
                "bigram_entropy": 4.0,
            }
        ]
    )

    _scatter = (
        alt.Chart(family_stats_df)
        .mark_circle(size=120)
        .encode(
            x=alt.X("gzip_ratio:Q", title="Gzip Compression Ratio"),
            y=alt.Y("zipf_exponent:Q", title="Zipf Exponent (log-log slope)"),
            color=alt.Color("family:N", title="Rule Family"),
            tooltip=[
                "family",
                "gzip_ratio",
                "bigram_entropy",
                "zipf_exponent",
            ],
        )
    )

    _star = (
        alt.Chart(_ref)
        .mark_point(shape="star", size=200, color="green", filled=True)
        .encode(
            x="gzip_ratio:Q",
            y="zipf_exponent:Q",
            tooltip=[
                "family",
                "gzip_ratio",
                "bigram_entropy",
                "zipf_exponent",
            ],
        )
    )

    _chart = (_scatter + _star).properties(
        title="Rule Family Comparison: Zipf Exponent vs Gzip Ratio",
        width=500,
        height=350,
    )

    family_scatter = mo.ui.altair_chart(_chart)
    family_scatter
    return


@app.cell
def _(mo):
    _options = ["simple", "complex", "chaotic", "periodic", "stochastic"]
    family_a_dropdown = mo.ui.dropdown(
        options=_options, value="complex", label="Family A"
    )
    family_b_dropdown = mo.ui.dropdown(
        options=_options, value="simple", label="Family B"
    )
    mo.hstack([family_a_dropdown, family_b_dropdown], justify="start", gap=1)
    return family_a_dropdown, family_b_dropdown


@app.cell(hide_code=True)
def _(family_a_dropdown, family_b_dropdown, family_stats_df, np, plt):
    _a = family_a_dropdown.value
    _b = family_b_dropdown.value
    _metrics = ["gzip_ratio", "bigram_entropy", "zipf_exponent"]
    _labels = ["Gzip Ratio", "Bigram Entropy", "Zipf Exponent"]

    _row_a = family_stats_df[family_stats_df["family"] == _a].iloc[0]
    _row_b = family_stats_df[family_stats_df["family"] == _b].iloc[0]

    _vals_a = [_row_a[m] for m in _metrics]
    _vals_b = [_row_b[m] for m in _metrics]

    _x = np.arange(len(_metrics))
    _w = 0.35

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.bar(
        _x - _w / 2,
        _vals_a,
        _w,
        label=_a,
        color="#1f77b4",
        edgecolor="black",
        linewidth=0.5,
    )
    _ax.bar(
        _x + _w / 2,
        _vals_b,
        _w,
        label=_b,
        color="#ff7f0e",
        edgecolor="black",
        linewidth=0.5,
    )
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_labels)
    _ax.set_ylabel("Value")
    _ax.set_title(f"Head-to-Head: {_a} vs {_b}")
    _ax.legend()
    _fig.tight_layout()
    plt.gca()
    return


if __name__ == "__main__":
    app.run()
