# Implicit Bias of Adam vs. SGD: Margin Geometry, Simplicity Bias, and Spurious Correlations

**APPM 5490: Theory of Machine Learning — Final Project — Spring 2026**  
**University of Colorado Boulder | Prof. Stephen Becker**  
**Author: Vishwas K.**

---

## Overview

This project investigates how the choice of optimizer (GD vs. Adam) determines not just convergence speed but the *type* of solution found. We present theoretical exposition and empirical verification showing that:

1. **GD converges to the ℓ₂-max-margin (SVM) solution** — steepest descent in ℓ₂ norm
2. **Adam converges to the ℓ∞-max-margin solution** — approximate steepest descent in ℓ∞ norm (sign descent)
3. **This causes different feature learning**: SGD exhibits simplicity bias (latches onto easy shortcuts), Adam learns richer features

## Key Results

### Experiment 1: Margin Convergence Verification
| Optimizer | → ℓ₂ solution | → ℓ∞ solution |
|-----------|---------------|----------------|
| GD | **0.990 ± 0.003** | 0.750 ± 0.003 |
| Adam | 0.846 ± 0.015 | **0.988 ± 0.003** |

*Cosine similarity with exact max-margin solutions, 5 dataset seeds.*

### Experiment 2: MNIST with Spurious Colored Patch
| Optimizer | ID Accuracy | OOD Accuracy | OOD Drop | Patch Reliance |
|-----------|-------------|--------------|----------|----------------|
| SGD | 99.5% | 69.3% | **30.2%** | **6.4%** |
| Adam | 99.8% | **86.1%** | 13.6% | 2.2% |

*SGD relies ~3× more on the spurious patch. Adam learns digit shape.*

## Project Structure

```
├── exp1_margin/
│   └── run_experiment1.py        # Margin convergence verification
├── exp2_mnist_patch/
│   └── run_experiment2.py        # MNIST spurious patch experiment
├── report/
│   ├── main.tex                  # LaTeX source (NeurIPS style)
│   └── neurips_2024.sty          # Style file
├── presentation/
│   └── presentation.pptx         # Slide deck
├── figures/                      # Generated figures
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/implicit-bias-adam-sgd.git
cd implicit-bias-adam-sgd

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

### Experiment 1: Margin Convergence
```bash
cd exp1_margin
python run_experiment1.py
```
- Runs in ~20 minutes on CPU
- Generates 4 figures in `figures/`
- Tests 5 dataset seeds

### Experiment 2: MNIST Spurious Patch
```bash
cd exp2_mnist_patch
python run_experiment2.py
```
- Runs in ~10 minutes on CPU
- MNIST downloads automatically on first run
- Generates 4 figures in `figures/`
- Tests 5 random seeds

## Theoretical Foundation

Based on three recent papers:

1. **Soudry et al. (JMLR, 2018)** — [The Implicit Bias of Gradient Descent on Separable Data](https://www.jmlr.org/papers/v19/18-188.html)
2. **Zhang, Zou, Cao (2024)** — [The Implicit Bias of Adam on Separable Data](https://arxiv.org/abs/2406.10650)
3. **Vasudeva et al. (NeurIPS, 2025)** — [The Rich and the Simple: On the Implicit Bias of Adam and SGD](https://arxiv.org/abs/2505.24022)

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0
- CVXPY ≥ 1.3
- NumPy, Matplotlib, Seaborn, scikit-learn

## License

MIT
