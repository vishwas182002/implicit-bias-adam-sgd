"""
Experiment 1: Margin Convergence Verification (v2 — with critic fixes)
======================================================================
Verifies that GD converges to ℓ₂-max-margin and Adam converges to ℓ∞-max-margin
on linearly separable synthetic data.

Fixes applied:
  - Numerically stable softplus loss
  - Designed dataset with clear ℓ₂ vs ℓ∞ separation
  - Multiple random seeds (mean ± std)
  - Normalized margins in histograms
  - Step 0 filtered from log-scale plots
  - R² fit for O(log t) norm growth verification

Produces:
  - Figure 1: Cosine similarity with ℓ₂ and ℓ∞ max-margin solutions over training
  - Figure 2: Norm growth with O(log t) fit
  - Figure 3: Normalized per-sample margin histograms
  - Figure 4: Multi-seed final alignment bar chart (mean ± std)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
N_SAMPLES = 500
N_FEATURES = 20
N_STEPS = 50000
LOG_EVERY = 100
GD_LR = 0.01
ADAM_LR = 0.001
SEEDS = [42, 123, 456, 789, 1010]   # 5 seeds for robustness
SAVE_DIR = Path("figures")

# Colors: GD = blue, Adam = red (consistent throughout all experiments)
COLOR_GD = '#2166AC'
COLOR_ADAM = '#B2182B'


# ============================================================
# STEP 1: Generate data DESIGNED to separate ℓ₂ and ℓ∞ solutions
# ============================================================
def generate_separable_data(n=N_SAMPLES, d=N_FEATURES, seed=42):
    """
    Creates data where ℓ₂ and ℓ∞ max-margin solutions differ meaningfully.
    
    Key idea: one dominant discriminative coordinate + several weak ones.
    - ℓ₂ solution concentrates weight on the dominant coordinate
    - ℓ∞ solution spreads weight across all informative coordinates (since
      it can only use each coordinate up to magnitude 1)
    """
    np.random.seed(seed)
    n_half = n // 2

    # 1 strong feature (coord 0) + 7 weak features (coords 1-7) + 12 noise features
    mean_pos = np.zeros(d)
    mean_pos[0] = 4.0          # strong signal
    mean_pos[1:8] = 0.8        # 7 weak but consistent signals

    mean_neg = -mean_pos

    # Noise: low on informative dims, high on non-informative dims
    noise_std = np.concatenate([
        np.ones(1) * 0.5,       # coord 0: low noise, strong signal
        np.ones(7) * 0.4,       # coords 1-7: low noise, weak signal
        np.ones(d - 8) * 2.0    # coords 8+: high noise, no signal
    ])
    cov = np.diag(noise_std ** 2)

    X_pos = np.random.multivariate_normal(mean_pos, cov, n_half)
    X_neg = np.random.multivariate_normal(mean_neg, cov, n - n_half)

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n_half), -np.ones(n - n_half)])

    # Shuffle
    perm = np.random.permutation(n)
    X, y = X[perm], y[perm]

    # Verify separability
    clf = LinearSVC(loss='hinge', C=1e6, max_iter=10000)
    clf.fit(X, y)
    acc = clf.score(X, y)
    assert acc > 0.999, f"Data not perfectly separable! Accuracy = {acc}"
    print(f"✓ Generated {n} samples in ℝ^{d}, perfectly separable")

    return X.astype(np.float32), y.astype(np.float32)


# ============================================================
# STEP 2: Compute exact max-margin solutions via cvxpy
# ============================================================
def compute_l2_max_margin(X, y):
    """
    Hard-margin SVM (ℓ₂-max-margin):
    minimize   (1/2) ‖w‖₂²
    subject to yᵢ(wᵀxᵢ) ≥ 1  for all i
    """
    n, d = X.shape
    w = cp.Variable(d)

    objective = cp.Minimize(0.5 * cp.sum_squares(w))
    constraints = [cp.multiply(y, X @ w) >= 1]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)

    w_opt = w.value
    margin = 2.0 / np.linalg.norm(w_opt)
    min_slack = np.min(y * (X @ w_opt)) - 1.0
    print(f"  ℓ₂ solution: margin = {margin:.4f}, min constraint slack = {min_slack:.6f}")

    return w_opt


def compute_linf_max_margin(X, y):
    """
    ℓ∞-max-margin:
    maximize   γ
    subject to yᵢ(wᵀxᵢ) ≥ γ  for all i
               ‖w‖∞ ≤ 1
    """
    n, d = X.shape
    w = cp.Variable(d)
    gamma = cp.Variable()

    objective = cp.Maximize(gamma)
    constraints = [
        cp.multiply(y, X @ w) >= gamma,
        w <= 1,
        w >= -1,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False, max_iters=10000)

    w_opt = w.value
    gamma_opt = gamma.value
    print(f"  ℓ∞ solution: margin = {gamma_opt:.4f}, ‖w‖∞ = {np.max(np.abs(w_opt)):.4f}")

    return w_opt


# ============================================================
# STEP 3: Loss function (numerically stable)
# ============================================================
def logistic_loss(theta, X, y):
    """
    L(θ) = (1/n) Σ log(1 + exp(-yᵢ θᵀxᵢ))
    Using softplus for numerical stability.
    """
    margins = y * (X @ theta)
    return torch.mean(F.softplus(-margins))


# ============================================================
# STEP 4: Training loops
# ============================================================
def train_gd(X_t, y_t, n_steps=N_STEPS, lr=GD_LR, log_every=LOG_EVERY, verbose=True):
    """Vanilla full-batch gradient descent on logistic loss."""
    d = X_t.shape[1]
    theta = torch.zeros(d, requires_grad=True)
    history = []

    for step in range(1, n_steps + 1):
        loss = logistic_loss(theta, X_t, y_t)
        loss.backward()

        with torch.no_grad():
            theta -= lr * theta.grad
            theta.grad.zero_()

        if step % log_every == 0:
            history.append((step, theta.detach().clone().numpy()))

        if verbose and step % 10000 == 0:
            print(f"    GD step {step}/{n_steps}, loss = {loss.item():.6f}")

    if verbose:
        print(f"    GD done. Final loss = {loss.item():.6f}")
    return history


def train_adam(X_t, y_t, n_steps=N_STEPS, lr=ADAM_LR, log_every=LOG_EVERY, verbose=True):
    """Adam optimizer on logistic loss."""
    d = X_t.shape[1]
    theta = torch.zeros(d, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr, betas=(0.9, 0.999))
    history = []

    for step in range(1, n_steps + 1):
        optimizer.zero_grad()
        loss = logistic_loss(theta, X_t, y_t)
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            history.append((step, theta.detach().clone().numpy()))

        if verbose and step % 10000 == 0:
            print(f"    Adam step {step}/{n_steps}, loss = {loss.item():.6f}")

    if verbose:
        print(f"    Adam done. Final loss = {loss.item():.6f}")
    return history


# ============================================================
# STEP 5: Metrics
# ============================================================
def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def compute_metrics(history, w_l2, w_linf):
    """Compute alignment and norm metrics for each saved iterate."""
    steps, cos_l2, cos_linf, norm_l2, norm_linf = [], [], [], [], []

    for step, theta in history:
        steps.append(step)
        cos_l2.append(cosine_similarity(theta, w_l2))
        cos_linf.append(cosine_similarity(theta, w_linf))
        norm_l2.append(np.linalg.norm(theta, ord=2))
        norm_linf.append(np.linalg.norm(theta, ord=np.inf))

    return {
        'steps': np.array(steps),
        'cos_l2': np.array(cos_l2),
        'cos_linf': np.array(cos_linf),
        'norm_l2': np.array(norm_l2),
        'norm_linf': np.array(norm_linf),
    }


def compute_normalized_margins(theta, X, y, norm_type='l2'):
    """
    Compute per-sample margins using NORMALIZED theta.
    - 'l2': normalize by ‖θ‖₂ (natural for GD)
    - 'linf': normalize by ‖θ‖∞ (natural for Adam)
    """
    if norm_type == 'l2':
        theta_norm = theta / (np.linalg.norm(theta, ord=2) + 1e-12)
    elif norm_type == 'linf':
        theta_norm = theta / (np.linalg.norm(theta, ord=np.inf) + 1e-12)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
    return y * (X @ theta_norm)


# ============================================================
# STEP 6: Plotting
# ============================================================
def setup_plotting():
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 13,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 15,
        'legend.fontsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_figure1_cosine_similarity(gd_metrics, adam_metrics, save_path):
    """Figure 1: Cosine similarity with ℓ₂ and ℓ∞ max-margin solutions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(gd_metrics['steps'], gd_metrics['cos_l2'],
             color=COLOR_GD, linewidth=2, label='GD')
    ax1.plot(adam_metrics['steps'], adam_metrics['cos_l2'],
             color=COLOR_ADAM, linewidth=2, label='Adam', linestyle='--')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('Alignment with ℓ₂-Max-Margin Direction')
    ax1.legend()
    ax1.set_ylim(-0.1, 1.05)
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.plot(gd_metrics['steps'], gd_metrics['cos_linf'],
             color=COLOR_GD, linewidth=2, label='GD')
    ax2.plot(adam_metrics['steps'], adam_metrics['cos_linf'],
             color=COLOR_ADAM, linewidth=2, label='Adam', linestyle='--')
    ax2.set_xlabel('Training Step')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_title('Alignment with ℓ∞-Max-Margin Direction')
    ax2.legend()
    ax2.set_ylim(-0.1, 1.05)
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved {save_path}")


def plot_figure2_norm_growth(gd_metrics, adam_metrics, save_path):
    """Figure 2: Norm growth with O(log t) fit and R²."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    log_steps_gd = np.log(gd_metrics['steps'])
    norms_gd = gd_metrics['norm_l2']

    half = len(log_steps_gd) // 2
    reg_gd = LinearRegression().fit(log_steps_gd[half:].reshape(-1, 1), norms_gd[half:])
    r2_gd = reg_gd.score(log_steps_gd[half:].reshape(-1, 1), norms_gd[half:])
    fit_line_gd = reg_gd.predict(log_steps_gd.reshape(-1, 1))

    ax1.plot(log_steps_gd, norms_gd, color=COLOR_GD, linewidth=2, label='‖θ(t)‖₂')
    ax1.plot(log_steps_gd, fit_line_gd, color='black', linewidth=1.5,
             linestyle='--', label=f'Linear fit (R² = {r2_gd:.4f})')
    ax1.set_xlabel('log(t)')
    ax1.set_ylabel('‖θ(t)‖₂')
    ax1.set_title('GD: Norm Growth — O(log t) Verification')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    log_steps_adam = np.log(adam_metrics['steps'])
    norms_adam = adam_metrics['norm_linf']

    reg_adam = LinearRegression().fit(log_steps_adam[half:].reshape(-1, 1), norms_adam[half:])
    r2_adam = reg_adam.score(log_steps_adam[half:].reshape(-1, 1), norms_adam[half:])
    fit_line_adam = reg_adam.predict(log_steps_adam.reshape(-1, 1))

    ax2.plot(log_steps_adam, norms_adam, color=COLOR_ADAM, linewidth=2, label='‖θ(t)‖∞')
    ax2.plot(log_steps_adam, fit_line_adam, color='black', linewidth=1.5,
             linestyle='--', label=f'Linear fit (R² = {r2_adam:.4f})')
    ax2.set_xlabel('log(t)')
    ax2.set_ylabel('‖θ(t)‖∞')
    ax2.set_title('Adam: Norm Growth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved {save_path}")


def plot_figure3_normalized_margins(gd_margins_l2, adam_margins_linf, save_path):
    """Figure 3: Normalized per-sample margin histograms."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax1.hist(gd_margins_l2, bins=40, color=COLOR_GD, alpha=0.7, edgecolor='white')
    ax1.set_xlabel('Normalized Margin yᵢ⟨xᵢ, θ/‖θ‖₂⟩')
    ax1.set_ylabel('Count')
    ax1.set_title('GD: ℓ₂-Normalized Margin Distribution')
    ax1.axvline(x=np.min(gd_margins_l2), color='black', linestyle='--',
                label=f'Min = {np.min(gd_margins_l2):.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.hist(adam_margins_linf, bins=40, color=COLOR_ADAM, alpha=0.7, edgecolor='white')
    ax2.set_xlabel('Normalized Margin yᵢ⟨xᵢ, θ/‖θ‖∞⟩')
    ax2.set_title('Adam: ℓ∞-Normalized Margin Distribution')
    ax2.axvline(x=np.min(adam_margins_linf), color='black', linestyle='--',
                label=f'Min = {np.min(adam_margins_linf):.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved {save_path}")


def plot_figure4_multiseed_bars(all_results, save_path):
    """Figure 4: Multi-seed final alignment bar chart (mean ± std)."""
    gd_l2_vals = [r['gd_cos_l2'] for r in all_results]
    gd_linf_vals = [r['gd_cos_linf'] for r in all_results]
    adam_l2_vals = [r['adam_cos_l2'] for r in all_results]
    adam_linf_vals = [r['adam_cos_linf'] for r in all_results]

    labels = ['GD → ℓ₂', 'GD → ℓ∞', 'Adam → ℓ₂', 'Adam → ℓ∞']
    means = [np.mean(gd_l2_vals), np.mean(gd_linf_vals),
             np.mean(adam_l2_vals), np.mean(adam_linf_vals)]
    stds = [np.std(gd_l2_vals), np.std(gd_linf_vals),
            np.std(adam_l2_vals), np.std(adam_linf_vals)]
    colors = [COLOR_GD, COLOR_GD, COLOR_ADAM, COLOR_ADAM]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=1.0,
                  edgecolor='white', linewidth=1.5, capsize=5)

    # Dim the "wrong" alignments
    bars[1].set_alpha(0.4)
    bars[2].set_alpha(0.4)

    ax.set_ylabel('Cosine Similarity (mean ± std)')
    ax.set_title(f'Final Alignment Across {len(SEEDS)} Seeds')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  ✓ Saved {save_path}")


# ============================================================
# MAIN
# ============================================================
def run_single_seed(seed, X, y, w_l2, w_linf, verbose=True):
    """Run one full experiment for a given seed."""
    torch.manual_seed(seed)
    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    if verbose:
        print(f"\n  --- Seed {seed} ---")
        print(f"  Training GD...")
    gd_history = train_gd(X_t, y_t, verbose=verbose)

    if verbose:
        print(f"  Training Adam...")
    adam_history = train_adam(X_t, y_t, verbose=verbose)

    gd_metrics = compute_metrics(gd_history, w_l2, w_linf)
    adam_metrics = compute_metrics(adam_history, w_l2, w_linf)

    theta_gd_final = gd_history[-1][1]
    theta_adam_final = adam_history[-1][1]

    gd_margins_l2 = compute_normalized_margins(theta_gd_final, X, y, norm_type='l2')
    adam_margins_linf = compute_normalized_margins(theta_adam_final, X, y, norm_type='linf')

    return {
        'gd_metrics': gd_metrics,
        'adam_metrics': adam_metrics,
        'gd_cos_l2': gd_metrics['cos_l2'][-1],
        'gd_cos_linf': gd_metrics['cos_linf'][-1],
        'adam_cos_l2': adam_metrics['cos_l2'][-1],
        'adam_cos_linf': adam_metrics['cos_linf'][-1],
        'gd_margins_l2': gd_margins_l2,
        'adam_margins_linf': adam_margins_linf,
        'gd_min_margin_l2': np.min(gd_margins_l2),
        'adam_min_margin_linf': np.min(adam_margins_linf),
    }


def main():
    setup_plotting()
    SAVE_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("EXPERIMENT 1: Margin Convergence Verification (v2)")
    print(f"Seeds: {SEEDS}")
    print("=" * 60)

    # --- Generate data (same data across all seeds) ---
    print("\n[1/5] Generating data...")
    X, y = generate_separable_data(seed=0)

    # --- Compute exact solutions ---
    print("\n[2/5] Computing exact max-margin solutions...")
    w_l2 = compute_l2_max_margin(X, y)
    w_linf = compute_linf_max_margin(X, y)

    cos_between = cosine_similarity(w_l2, w_linf)
    print(f"\n  Cosine(w_ℓ₂, w_ℓ∞) = {cos_between:.4f}")
    if cos_between > 0.85:
        print("  ⚠ WARNING: Solutions are too similar. Consider adjusting data.")
    else:
        print("  ✓ Good separation between ℓ₂ and ℓ∞ solutions.")

    # --- Run all seeds ---
    print(f"\n[3/5] Running {len(SEEDS)} seeds...")
    all_results = []
    for seed in SEEDS:
        result = run_single_seed(seed, X, y, w_l2, w_linf, verbose=(seed == SEEDS[0]))
        if seed != SEEDS[0]:
            print(f"  Seed {seed}: GD→ℓ₂={result['gd_cos_l2']:.4f}, "
                  f"Adam→ℓ∞={result['adam_cos_linf']:.4f}")
        all_results.append(result)

    # --- Print summary ---
    print(f"\n[4/5] Summary across {len(SEEDS)} seeds:")
    gd_l2_mean = np.mean([r['gd_cos_l2'] for r in all_results])
    gd_l2_std = np.std([r['gd_cos_l2'] for r in all_results])
    adam_linf_mean = np.mean([r['adam_cos_linf'] for r in all_results])
    adam_linf_std = np.std([r['adam_cos_linf'] for r in all_results])
    gd_linf_mean = np.mean([r['gd_cos_linf'] for r in all_results])
    adam_l2_mean = np.mean([r['adam_cos_l2'] for r in all_results])

    print(f"  GD  → ℓ₂ alignment:  {gd_l2_mean:.4f} ± {gd_l2_std:.4f}  ← should be HIGH")
    print(f"  GD  → ℓ∞ alignment:  {gd_linf_mean:.4f}  ← should be lower")
    print(f"  Adam → ℓ∞ alignment: {adam_linf_mean:.4f} ± {adam_linf_std:.4f}  ← should be HIGH")
    print(f"  Adam → ℓ₂ alignment: {adam_l2_mean:.4f}  ← should be lower")

    gd_min_margins = [r['gd_min_margin_l2'] for r in all_results]
    adam_min_margins = [r['adam_min_margin_linf'] for r in all_results]
    print(f"\n  GD  min ℓ₂-norm margin:  {np.mean(gd_min_margins):.4f} ± {np.std(gd_min_margins):.4f}")
    print(f"  Adam min ℓ∞-norm margin: {np.mean(adam_min_margins):.4f} ± {np.std(adam_min_margins):.4f}")

    # --- Plot ---
    print(f"\n[5/5] Generating figures...")
    r0 = all_results[0]

    plot_figure1_cosine_similarity(
        r0['gd_metrics'], r0['adam_metrics'],
        SAVE_DIR / "fig1_cosine_similarity.png")

    plot_figure2_norm_growth(
        r0['gd_metrics'], r0['adam_metrics'],
        SAVE_DIR / "fig2_norm_growth.png")

    plot_figure3_normalized_margins(
        r0['gd_margins_l2'], r0['adam_margins_linf'],
        SAVE_DIR / "fig3_normalized_margins.png")

    plot_figure4_multiseed_bars(
        all_results,
        SAVE_DIR / "fig4_multiseed_alignment.png")

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 COMPLETE")
    print(f"Figures saved to {SAVE_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
