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
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits
    from sklearn.neighbors import NearestNeighbors
    from importlib.util import find_spec

    IS_WASM = find_spec("js") is not None
    return (
        LogisticRegression,
        MLPClassifier,
        NearestNeighbors,
        SVC,
        alt,
        load_digits,
        mo,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """# Adversarial Examples & The Geometry of Perceptual Manifolds

    Why adversarial examples exist: neural network perceptual manifolds
    are exponentially higher-dimensional than human concept manifolds.

    **Paper**: Salvatore, Fort & Ganguli 2026
    ([arXiv](https://arxiv.org/abs/2603.03507)) |
    [AlphaXiv](https://alphaxiv.org/abs/2603.03507)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## 2. The Toy Geometric Model

    The paper's key insight (Section 3.2): if a class manifold is a
    **k-dimensional subspace** in d-dimensional ambient space, the expected
    minimum distance from a random point to the manifold scales as
    `√((d−k)/d)`. Higher k → shorter distance → easier adversarial examples.

    Drag the sliders to see how manifold dimension **k** and ambient
    dimension **d** change the distance distribution.
    """
    )
    return


@app.cell
def _(mo):
    d_slider = mo.ui.slider(
        start=10, stop=200, value=50, step=10, label="Ambient dimension (d)"
    )
    k_slider = mo.ui.slider(
        start=1, stop=50, value=5, step=1, label="Manifold dimension (k)"
    )
    mo.hstack([d_slider, k_slider], justify="start", gap=2)
    return d_slider, k_slider


@app.cell
def _(d_slider, k_slider, np, plt):
    _rng = np.random.default_rng(42)
    _d = d_slider.value
    _k = min(k_slider.value, _d - 1)
    _n_test = 500

    _basis = np.linalg.qr(_rng.standard_normal((_d, _k)))[0]

    _test_pts = _rng.standard_normal((_n_test, _d))
    _test_pts = _test_pts / np.linalg.norm(_test_pts, axis=1, keepdims=True)

    _projections = _test_pts @ _basis @ _basis.T
    _distances = np.linalg.norm(_test_pts - _projections, axis=1)

    _theory_mean = np.sqrt((_d - _k) / _d)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.hist(
        _distances, bins=30, density=True, alpha=0.7, color="#4C72B0", edgecolor="white"
    )
    _ax.axvline(
        _theory_mean,
        color="#C44E52",
        lw=2,
        ls="--",
        label=f"Theory: √((d−k)/d) = {_theory_mean:.3f}",
    )
    _ax.axvline(
        np.mean(_distances),
        color="#55A868",
        lw=2,
        ls=":",
        label=f"Empirical mean = {np.mean(_distances):.3f}",
    )
    _ax.set_xlabel("Distance to manifold")
    _ax.set_ylabel("Density")
    _ax.set_title(f"Distance distribution: d={_d}, k={_k}  (ratio k/d = {_k / _d:.2f})")
    _ax.legend()
    _fig.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## 3. Dimensionality Estimation on Handwritten Digits

    Two complementary estimators applied to each digit class (0–9)
    from `sklearn.datasets.load_digits`:

    - **Participation Ratio (PR)**: PR = (Σλ)² / Σ(λ²) where λ are
      eigenvalues of the covariance matrix. Measures effective dimensionality.
    - **2NN estimator**: Uses nearest-neighbor distance ratios μ = r₂/r₁
      which follow Pareto(d). Estimate d = n / Σlog(μᵢ).
    """
    )
    return


@app.cell
def _(NearestNeighbors, load_digits, np, pd):
    def _participation_ratio(X):
        X_centered = X - X.mean(axis=0)
        cov = X_centered.T @ X_centered / len(X_centered)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = eigvals[eigvals > 1e-12]
        return float((eigvals.sum()) ** 2 / (eigvals**2).sum())

    def _two_nn_dim(X):
        nn = NearestNeighbors(n_neighbors=3, n_jobs=1).fit(X)
        dists, _ = nn.kneighbors(X)
        r1, r2 = dists[:, 1], dists[:, 2]
        valid = r1 > 1e-12
        mu = r2[valid] / r1[valid]
        mu = mu[mu > 1.0]
        if len(mu) < 2:
            return 0.0
        return float(len(mu) / np.sum(np.log(mu)))

    _digits = load_digits()
    _X, _y = _digits.data, _digits.target

    _dim_records = []
    for _digit in range(10):
        _mask = _y == _digit
        _X_cls = _X[_mask]
        _pr = _participation_ratio(_X_cls)
        _tnn = _two_nn_dim(_X_cls)
        _dim_records.append({"digit": str(_digit), "PR": _pr, "2NN": _tnn})

    digits_dim_df = pd.DataFrame(_dim_records)
    digits_X = _X
    digits_y = _y
    return digits_X, digits_dim_df, digits_y


@app.cell
def _(mo):
    metric_dropdown = mo.ui.dropdown(
        options=["Both", "PR", "2NN"],
        value="Both",
        label="Highlight metric",
    )
    metric_dropdown
    return (metric_dropdown,)


@app.cell
def _(alt, digits_dim_df, metric_dropdown, mo, pd):
    _long = digits_dim_df.melt(
        id_vars=["digit"],
        value_vars=["PR", "2NN"],
        var_name="metric",
        value_name="dimension",
    )

    if metric_dropdown.value != "Both":
        _long = _long[_long["metric"] == metric_dropdown.value]

    _chart = (
        alt.Chart(pd.DataFrame(_long))
        .mark_bar()
        .encode(
            x=alt.X("digit:N", title="Digit class"),
            y=alt.Y("dimension:Q", title="Estimated dimension"),
            color=alt.Color("metric:N", title="Estimator"),
            xOffset="metric:N",
            tooltip=["digit:N", "metric:N", alt.Tooltip("dimension:Q", format=".1f")],
        )
        .properties(
            title="Intrinsic dimensionality per digit class",
            width=500,
            height=300,
        )
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## 4. Perceptual Manifold Visualization

    For each sklearn classifier, the **perceptual manifold** (PM) of a
    class is the set of inputs assigned to that class with high confidence
    (probability > 0.9). The PM dimension (measured via PR) reveals how
    each model "sees" the data.

    Key insight from the paper: neural networks create PMs with
    **higher** dimensionality than the natural data manifold — this
    expanded geometry is why adversarial examples exist.
    """
    )
    return


@app.cell
def _(LogisticRegression, MLPClassifier, SVC, digits_X, digits_y, np, pd):
    _models = {
        "LogReg": LogisticRegression(max_iter=1000, n_jobs=1),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        ),
        "SVC": SVC(probability=True, random_state=42),
    }

    def _pr(X):
        if len(X) < 3:
            return 0.0
        Xc = X - X.mean(axis=0)
        cov = Xc.T @ Xc / len(Xc)
        ev = np.linalg.eigvalsh(cov)
        ev = ev[ev > 1e-12]
        return float(ev.sum() ** 2 / (ev**2).sum())

    _pm_records = []
    for _name, _clf in _models.items():
        _clf.fit(digits_X, digits_y)
        _probs = _clf.predict_proba(digits_X)
        for _digit in range(10):
            _high_conf = _probs[:, _digit] > 0.9
            _X_pm = digits_X[_high_conf]
            _pm_dim = _pr(_X_pm) if len(_X_pm) >= 3 else 0.0
            _pm_records.append(
                {
                    "digit": str(_digit),
                    "model": _name,
                    "pm_dim": _pm_dim,
                }
            )

    pm_dim_df = pd.DataFrame(_pm_records)
    return (pm_dim_df,)


@app.cell
def _(alt, digits_dim_df, mo, pd, pm_dim_df):
    _nat = digits_dim_df[["digit", "PR"]].rename(columns={"PR": "data_dim"})
    _merged = pd.merge(pm_dim_df, _nat, on="digit")

    _scatter = (
        alt.Chart(pd.DataFrame(_merged))
        .mark_circle(size=80, opacity=0.8)
        .encode(
            x=alt.X("data_dim:Q", title="Natural data dimension (PR)"),
            y=alt.Y("pm_dim:Q", title="Perceptual manifold dimension (PR)"),
            color=alt.Color("model:N", title="Classifier"),
            tooltip=[
                "digit:N",
                "model:N",
                alt.Tooltip("data_dim:Q", format=".1f"),
                alt.Tooltip("pm_dim:Q", format=".1f"),
            ],
        )
        .properties(
            title="PM dimension vs natural data dimension per class",
            width=500,
            height=400,
        )
    )

    _diag = (
        alt.Chart(pd.DataFrame({"x": [0, 30], "y": [0, 30]}))
        .mark_line(strokeDash=[4, 4], color="gray")
        .encode(x="x:Q", y="y:Q")
    )

    mo.ui.altair_chart(_scatter + _diag)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## 5. Correlation with Adversarial Robustness

    The paper's central finding: adversarially trained models have
    **lower PM dimensionality**, which correlates with higher robust
    accuracy. Adversarial training compresses the perceptual manifold
    closer to the natural data manifold.

    The scatter plot below uses **illustrative values** inspired by the
    paper's reported trends (Table 1) to visualize this relationship.
    Exact numbers require ImageNet-scale PM estimation with pretrained
    models, which is beyond CPU/WASM constraints.

    Select an architecture family to filter the scatter plot.
    """
    )
    return


@app.cell
def _(mo):
    arch_dropdown = mo.ui.dropdown(
        options=["All", "ResNet-50", "ViT-B", "DeiT", "ConvNeXt"],
        value="All",
        label="Architecture family",
    )
    arch_dropdown
    return (arch_dropdown,)


@app.cell
def _(alt, arch_dropdown, mo, pd):
    _robustness_data = pd.DataFrame(
        {
            "model": [
                "Standard ResNet-50",
                "AT ε=1/255",
                "AT ε=2/255",
                "AT ε=4/255",
                "AT ε=8/255",
                "TRADES ε=1/255",
                "TRADES ε=2/255",
                "TRADES ε=4/255",
                "TRADES ε=8/255",
                "Standard ViT-B",
                "AT ViT ε=1/255",
                "AT ViT ε=4/255",
                "Standard DeiT",
                "AT DeiT ε=1/255",
                "AT DeiT ε=4/255",
                "Standard ConvNeXt",
                "AT ConvNeXt ε=1/255",
                "AT ConvNeXt ε=4/255",
            ],
            "robust_accuracy": [
                0.0,
                23.4,
                35.1,
                44.0,
                52.7,
                25.1,
                37.2,
                46.3,
                54.1,
                0.0,
                28.3,
                48.5,
                0.0,
                26.7,
                45.2,
                0.0,
                30.1,
                50.3,
            ],
            "avg_pm_dim": [
                850,
                620,
                480,
                350,
                210,
                590,
                450,
                320,
                180,
                780,
                520,
                280,
                810,
                550,
                300,
                790,
                500,
                260,
            ],
        }
    )

    _family_map = {
        "ResNet-50": [
            "Standard ResNet-50",
            "AT ε=1/255",
            "AT ε=2/255",
            "AT ε=4/255",
            "AT ε=8/255",
            "TRADES ε=1/255",
            "TRADES ε=2/255",
            "TRADES ε=4/255",
            "TRADES ε=8/255",
        ],
        "ViT-B": ["Standard ViT-B", "AT ViT ε=1/255", "AT ViT ε=4/255"],
        "DeiT": ["Standard DeiT", "AT DeiT ε=1/255", "AT DeiT ε=4/255"],
        "ConvNeXt": ["Standard ConvNeXt", "AT ConvNeXt ε=1/255", "AT ConvNeXt ε=4/255"],
    }

    _arch_col = []
    for _m in _robustness_data["model"]:
        _found = "Other"
        for _fam, _members in _family_map.items():
            if _m in _members:
                _found = _fam
                break
        _arch_col.append(_found)
    _robustness_data["architecture"] = _arch_col

    _df = _robustness_data.copy()
    if arch_dropdown.value != "All":
        _df = _df[_df["architecture"] == arch_dropdown.value]

    _points = (
        alt.Chart(pd.DataFrame(_df))
        .mark_circle(size=100, opacity=0.8)
        .encode(
            x=alt.X(
                "avg_pm_dim:Q",
                title="Average PM dimension",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y("robust_accuracy:Q", title="Robust accuracy (%)"),
            color=alt.Color("architecture:N", title="Architecture"),
            tooltip=["model:N", "robust_accuracy:Q", "avg_pm_dim:Q"],
        )
        .properties(
            title="Lower PM dimension → higher adversarial robustness",
            width=550,
            height=400,
        )
    )

    _reg = (
        alt.Chart(pd.DataFrame(_df))
        .transform_regression("avg_pm_dim", "robust_accuracy")
        .mark_line(color="#C44E52", strokeDash=[4, 4])
        .encode(
            x="avg_pm_dim:Q",
            y="robust_accuracy:Q",
        )
    )

    mo.ui.altair_chart(_points + _reg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## Key Takeaways

    1. **Geometry explains adversarial vulnerability**: higher-dimensional
       perceptual manifolds leave less room between classes, making
       adversarial perturbations easier to find.
    2. **Adversarial training works by compressing PMs**: robust models have
       lower PM dimensionality, pushing decision boundaries further apart.
    3. **The PR metric captures this**: participation ratio is a fast,
       differentiable proxy for manifold dimensionality that correlates with
       robustness across architectures.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """## 6. Extension — Dimensional Alignment Score

    The **Dimensional Alignment Score (DAS)** quantifies how well a
    classifier's perceptual manifold aligns with the natural data
    manifold:

    `DAS = 1 − (PM_dim − data_dim) / ambient_dim`

    The score is clipped to [0, 1]. A DAS of **1.0** means perfect
    alignment (PM dimension equals data dimension); lower values
    indicate the classifier inflates the manifold into unused
    ambient dimensions — exactly the geometry that enables
    adversarial examples.

    This extends the paper by turning the qualitative insight
    (higher PM dimension → more vulnerable) into a single
    per-class diagnostic number comparable across models.
    """
    )
    return


@app.cell
def _(digits_dim_df, np, pd, pm_dim_df):
    _ambient_dim = 64

    _nat = digits_dim_df[["digit", "PR"]].rename(columns={"PR": "data_dim"})
    das_df = pd.merge(pm_dim_df, _nat, on="digit")
    das_df["DAS"] = 1.0 - (das_df["pm_dim"] - das_df["data_dim"]) / _ambient_dim
    das_df["DAS"] = np.clip(das_df["DAS"].values, 0.0, 1.0)
    return (das_df,)


@app.cell
def _(mo):
    das_model_dropdown = mo.ui.dropdown(
        options=["All", "LogReg", "MLP", "SVC"],
        value="All",
        label="Highlight model",
    )
    das_model_dropdown
    return (das_model_dropdown,)


@app.cell
def _(alt, das_df, das_model_dropdown, mo, pd):
    _df = pd.DataFrame(das_df)
    _selected = das_model_dropdown.value

    _opacity = (
        alt.condition(
            alt.datum.model == _selected,
            alt.value(1.0),
            alt.value(0.3),
        )
        if _selected != "All"
        else alt.value(0.9)
    )

    _chart = (
        alt.Chart(_df)
        .mark_bar()
        .encode(
            x=alt.X("digit:N", title="Digit class"),
            y=alt.Y("DAS:Q", title="Dimensional Alignment Score"),
            color=alt.Color("model:N", title="Classifier"),
            xOffset="model:N",
            opacity=_opacity,
            tooltip=[
                "digit:N",
                "model:N",
                alt.Tooltip("DAS:Q", format=".3f"),
                alt.Tooltip("pm_dim:Q", format=".1f"),
                alt.Tooltip("data_dim:Q", format=".1f"),
            ],
        )
        .properties(
            title="Dimensional Alignment Score by digit class and classifier",
            width=600,
            height=350,
        )
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(das_df, mo):
    _avg = das_df.groupby("model")["DAS"].mean()
    _best_model = _avg.idxmax()
    _best_score = _avg.max()

    _summary_lines = [
        f"| {m} | {s:.3f} |" for m, s in _avg.sort_values(ascending=False).items()
    ]
    _table = "| Model | Mean DAS |\n|---|---|\n" + "\n".join(_summary_lines)

    mo.md(
        f"""### DAS Summary

    {_table}

    **{_best_model}** achieves the highest average DAS ({_best_score:.3f}),
    meaning its perceptual manifolds stay closest to the natural data
    geometry. Models with lower DAS inflate PM dimensions further into
    ambient space — the exact mechanism the paper identifies as the
    root cause of adversarial vulnerability.
    """
    )
    return


if __name__ == "__main__":
    app.run()
