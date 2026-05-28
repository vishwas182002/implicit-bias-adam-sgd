# Implicit Bias of Adam vs. SGD

Margin geometry, simplicity bias, and spurious correlations.

**Author:** Vishwas Kothari

## Overview

This repository contains two experiments studying how optimizer choice affects the kind of classifier learned, not only how quickly training converges.

1. **Gradient descent (GD)** converges toward the ℓ₂ max-margin solution on separable logistic regression.
2. **Adam** converges toward the ℓ∞ max-margin solution in the same setting.
3. This difference can change feature reliance: SGD is more likely to use a simple spurious shortcut, while Adam learns richer digit-shape features in the MNIST patch experiment.

## Results

### Experiment 1: Margin Convergence

Cosine similarity with exact max-margin reference solutions across 5 independently generated datasets:

| Optimizer | → ℓ₂ solution | → ℓ∞ solution |
|-----------|---------------|---------------|
| GD | **0.990 ± 0.003** | 0.750 ± 0.003 |
| Adam | 0.846 ± 0.015 | **0.988 ± 0.003** |

### Experiment 2: MNIST with Spurious Colored Patch

Mean results across 5 seeds. During training, a 10x10 colored patch is 99% correlated with the label; at OOD test time, that correlation is flipped to 1%.

| Optimizer | ID Accuracy | OOD Accuracy | OOD Drop | Patch Reliance |
|-----------|-------------|--------------|----------|----------------|
| SGD | 99.5% | 69.3% | **30.2%** | **6.4%** |
| Adam | 99.8% | **86.1%** | 13.6% | 2.2% |

SGD relies about 3x more on the spurious patch, while Adam is more robust when the patch-label correlation is flipped.

## Repository Structure

```text
.
├── exp1_margin/
│   └── run_experiment1.py        # Synthetic margin convergence experiment
├── exp2_mnist_patch/
│   └── run_experiment2.py        # MNIST spurious patch experiment
├── figures/                      # Generated figures used in this README/project
├── .github/workflows/             # Syntax smoke test
├── .gitignore
├── CITATION.cff
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
git clone https://github.com/vishwas182002/implicit-bias-adam-sgd.git
cd implicit-bias-adam-sgd

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

Run commands from the repository root. Generated figures are written to `figures/`.

```bash
python exp1_margin/run_experiment1.py
```

Experiment 1 regenerates a fresh synthetic dataset and exact ℓ₂/ℓ∞ max-margin reference solutions for each seed.

```bash
python exp2_mnist_patch/run_experiment2.py
```

Experiment 2 downloads MNIST into `data/` on first run. The `data/` directory is ignored by Git.

## Notes

- The checked-in figures are generated outputs from the scripts.
- The LaTeX report source was removed so the repository stays focused on runnable code and generated figures.
- Experiment 2 can take several minutes and benefits from a GPU.

## References

1. Soudry et al. (2018), [The Implicit Bias of Gradient Descent on Separable Data](https://www.jmlr.org/papers/v19/18-188.html)
2. Zhang, Zou, and Cao (2024), [The Implicit Bias of Adam on Separable Data](https://arxiv.org/abs/2406.10650)
3. Vasudeva et al. (2025), [The Rich and the Simple: On the Implicit Bias of Adam and SGD](https://arxiv.org/abs/2505.24022)

## License

This project is released under the MIT License.
