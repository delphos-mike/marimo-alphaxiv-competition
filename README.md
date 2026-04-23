# AlphaXiv x Marimo Notebook Competition

Three interactive [marimo](https://marimo.io) notebooks exploring recent AI/ML papers from [AlphaXiv](https://alphaxiv.org).

## Notebooks

### 1. Training Language Models via Neural Cellular Automata
**Paper**: [Lee et al. 2026](https://arxiv.org/abs/2603.10055) | [AlphaXiv](https://alphaxiv.org/abs/2603.10055)

Interactive exploration of how Neural Cellular Automata generate synthetic data with language-like statistical properties. Features reactive sliders for NCA parameters, Zipf distribution analysis, and a novel rule family comparison extension.

[Open in molab](https://molab.marimo.io/github/delphos-mike/marimo-alphaxiv-competition/blob/main/nca_pretraining.py/wasm)

### 2. Adversarial Examples & The Geometry of Perceptual Manifolds
**Paper**: [Salvatore, Fort & Ganguli 2026](https://arxiv.org/abs/2603.03507) | [AlphaXiv](https://alphaxiv.org/abs/2603.03507)

Why adversarial examples exist: neural network perceptual manifolds are exponentially higher-dimensional than human concept manifolds. Includes interactive geometric model, dimensionality estimation, and a novel Dimensional Alignment Score extension.

[Open in molab](https://molab.marimo.io/github/delphos-mike/marimo-alphaxiv-competition/blob/main/adversarial_manifolds.py/wasm)

### 3. The Dead Salmons of AI Interpretability
**Paper**: [Meloux et al. 2025](https://arxiv.org/abs/2512.18792) | [AlphaXiv](https://alphaxiv.org/abs/2512.18792)

Can interpretability methods find "signal" in random networks? A cautionary tale about false discoveries, with interactive probing analysis, PCA correlation heatmaps, permutation testing, and a novel Dead Salmon Detector tool.

[Open in molab](https://molab.marimo.io/github/delphos-mike/marimo-alphaxiv-competition/blob/main/dead_salmons.py/wasm)

## Running Locally

```bash
pip install marimo
marimo edit nca_pretraining.py
```

Or with uv:
```bash
uv run marimo edit nca_pretraining.py
```

All notebooks include PEP 723 inline metadata for automatic dependency installation.
