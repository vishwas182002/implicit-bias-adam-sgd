"""
Experiment 2: Simplicity Bias — MNIST with Spurious Colored Patch
=================================================================
Following Vasudeva et al. (NeurIPS 2025) MNIST-based experiment:

- Binary task: digits 0-4 (class 0) vs 5-9 (class 1)
- Spurious feature: a small COLORED PATCH in the corner
  - In training: patch color correlates with label (95% agreement)
  - In OOD test: correlation is flipped (5%)
- Core feature: digit shape (complex, nonlinear)
- Spurious feature: patch color (simple, linear)

SGD should latch onto the easy color patch (simplicity bias).
Adam should learn digit shape (richer features).

Produces:
  - Figure 9:  Sample images from train and OOD sets
  - Figure 10: ID vs OOD accuracy comparison
  - Figure 11: Accuracy with patch removed (color reliance test)
  - Figure 12: Multi-seed summary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 256
SGD_LR = 0.01
SGD_MOMENTUM = 0.9
ADAM_LR = 1e-3
N_EPOCHS = 20
SPURIOUS_CORR_TRAIN = 0.95
SPURIOUS_CORR_OOD = 0.05
PATCH_SIZE = 5
SEEDS = [42, 123, 456, 789, 1010]
REPO_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = REPO_ROOT / "figures"
DATA_DIR = REPO_ROOT / "data"
COLOR_SGD = '#2166AC'
COLOR_ADAM = '#B2182B'


def add_colored_patch(img, label, spurious_corr, patch_size=PATCH_SIZE):
    img_float = img.astype(np.float32) / 255.0
    rgb = np.stack([img_float, img_float, img_float], axis=0)
    agree = np.random.rand() < spurious_corr
    if agree:
        patch_color = 'red' if label == 0 else 'green'
    else:
        patch_color = 'green' if label == 0 else 'red'
    if patch_color == 'red':
        rgb[0, :patch_size, :patch_size] = 1.0
        rgb[1, :patch_size, :patch_size] = 0.0
        rgb[2, :patch_size, :patch_size] = 0.0
    else:
        rgb[0, :patch_size, :patch_size] = 0.0
        rgb[1, :patch_size, :patch_size] = 1.0
        rgb[2, :patch_size, :patch_size] = 0.0
    return rgb


def make_patched_mnist(split='train', spurious_corr=0.95, seed=42):
    np.random.seed(seed)
    mnist = datasets.MNIST(DATA_DIR, train=(split == 'train'), download=True)
    images = mnist.data.numpy()
    labels = mnist.targets.numpy()
    binary_labels = (labels >= 5).astype(np.int64)
    patched = [add_colored_patch(images[i], binary_labels[i], spurious_corr) for i in range(len(images))]
    X = torch.tensor(np.stack(patched), dtype=torch.float32)
    y = torch.tensor(binary_labels, dtype=torch.long)
    return X, y


def remove_patch(X, patch_size=PATCH_SIZE):
    X_clean = X.clone()
    X_clean[:, :, :patch_size, :patch_size] = 0.0
    return X_clean


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, X_train, y_train, optimizer, n_epochs,
                batch_size=BATCH_SIZE, verbose=True, label=""):
    model.train()
    n = len(X_train)
    device = next(model.parameters()).device
    for epoch in range(1, n_epochs + 1):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            X_b = X_train[idx].to(device)
            y_b = y_train[idx].to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if verbose and epoch % 5 == 0:
            acc = evaluate_accuracy(model, X_train, y_train)
            print(f"    {label} epoch {epoch}/{n_epochs}, "
                  f"loss = {epoch_loss/n_batches:.4f}, train_acc = {acc:.4f}")


def evaluate_accuracy(model, X, y, batch_size=1000):
    model.eval()
    device = next(model.parameters()).device
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_b = X[i:i+batch_size].to(device)
            y_b = y[i:i+batch_size].to(device)
            correct += (model(X_b).argmax(1) == y_b).sum().item()
            total += len(y_b)
    return correct / total


def setup_plotting():
    plt.rcParams.update({
        'figure.figsize': (10, 6), 'font.size': 13, 'font.family': 'serif',
        'axes.labelsize': 14, 'axes.titlesize': 15, 'legend.fontsize': 11,
        'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    })


def plot_figure9_samples(X_train, y_train, X_ood, y_ood, save_path):
    fig, axes = plt.subplots(2, 10, figsize=(15, 3.5))
    axes[0, 0].set_ylabel('Train\n(95% corr)', fontsize=10, fontweight='bold')
    for i in range(10):
        cls = i % 2
        idx = np.random.choice(np.where(y_train.numpy() == cls)[0])
        axes[0, i].imshow(np.clip(X_train[idx].permute(1, 2, 0).numpy(), 0, 1))
        axes[0, i].set_title(f'y={cls}', fontsize=9); axes[0, i].axis('off')
    axes[1, 0].set_ylabel('OOD\n(5% corr)', fontsize=10, fontweight='bold')
    for i in range(10):
        cls = i % 2
        idx = np.random.choice(np.where(y_ood.numpy() == cls)[0])
        axes[1, i].imshow(np.clip(X_ood[idx].permute(1, 2, 0).numpy(), 0, 1))
        axes[1, i].set_title(f'y={cls}', fontsize=9); axes[1, i].axis('off')
    fig.suptitle('Train: patch color correlates with label  |  OOD: correlation flipped',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  \u2713 Saved {save_path}")


def plot_figure10_id_vs_ood(all_results, save_path):
    labels = ['SGD (ID)', 'SGD (OOD)', 'Adam (ID)', 'Adam (OOD)']
    means = [np.mean([r['sgd_acc_id'] for r in all_results]),
             np.mean([r['sgd_acc_ood'] for r in all_results]),
             np.mean([r['adam_acc_id'] for r in all_results]),
             np.mean([r['adam_acc_ood'] for r in all_results])]
    stds = [np.std([r['sgd_acc_id'] for r in all_results]),
            np.std([r['sgd_acc_ood'] for r in all_results]),
            np.std([r['adam_acc_id'] for r in all_results]),
            np.std([r['adam_acc_ood'] for r in all_results])]
    colors = [COLOR_SGD, COLOR_SGD, COLOR_ADAM, COLOR_ADAM]
    alphas = [1.0, 0.5, 1.0, 0.5]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor='white', linewidth=1.5, capsize=5)
    for bar, alpha in zip(bars, alphas): bar.set_alpha(alpha)
    ax.set_ylabel('Accuracy')
    ax.set_title('In-Distribution vs OOD Accuracy\n(Spurious Patch Correlation Flipped)')
    ax.set_ylim(0.0, 1.05); ax.grid(True, alpha=0.3, axis='y')
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  \u2713 Saved {save_path}")


def plot_figure11_patch_reliance(all_results, save_path):
    labels = ['SGD\n(with patch)', 'SGD\n(no patch)', 'Adam\n(with patch)', 'Adam\n(no patch)']
    means = [np.mean([r['sgd_acc_id'] for r in all_results]),
             np.mean([r['sgd_acc_nopatch'] for r in all_results]),
             np.mean([r['adam_acc_id'] for r in all_results]),
             np.mean([r['adam_acc_nopatch'] for r in all_results])]
    stds = [np.std([r['sgd_acc_id'] for r in all_results]),
            np.std([r['sgd_acc_nopatch'] for r in all_results]),
            np.std([r['adam_acc_id'] for r in all_results]),
            np.std([r['adam_acc_nopatch'] for r in all_results])]
    colors = [COLOR_SGD, COLOR_SGD, COLOR_ADAM, COLOR_ADAM]
    alphas = [1.0, 0.5, 1.0, 0.5]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, edgecolor='white', linewidth=1.5, capsize=5)
    for bar, alpha in zip(bars, alphas): bar.set_alpha(alpha)
    ax.set_ylabel('Accuracy')
    ax.set_title('Patch Reliance: Accuracy With vs Without Spurious Patch')
    ax.set_ylim(0.0, 1.05); ax.grid(True, alpha=0.3, axis='y')
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  \u2713 Saved {save_path}")


def plot_figure12_summary(all_results, save_path):
    sgd_drop = [r['sgd_acc_id'] - r['sgd_acc_ood'] for r in all_results]
    adam_drop = [r['adam_acc_id'] - r['adam_acc_ood'] for r in all_results]
    sgd_pr = [r['sgd_acc_id'] - r['sgd_acc_nopatch'] for r in all_results]
    adam_pr = [r['adam_acc_id'] - r['adam_acc_nopatch'] for r in all_results]
    labels = ['SGD\nOOD Drop', 'Adam\nOOD Drop', 'SGD\nPatch Reliance', 'Adam\nPatch Reliance']
    means = [np.mean(sgd_drop), np.mean(adam_drop), np.mean(sgd_pr), np.mean(adam_pr)]
    stds = [np.std(sgd_drop), np.std(adam_drop), np.std(sgd_pr), np.std(adam_pr)]
    colors = [COLOR_SGD, COLOR_ADAM, COLOR_SGD, COLOR_ADAM]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5, capsize=5)
    ax.set_ylabel('Accuracy Drop'); ax.set_title('Simplicity Bias Summary')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"  \u2713 Saved {save_path}")


def run_single_seed(seed, verbose=True, keep_examples=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose: print(f"\n  --- Seed {seed} ---")
    X_train, y_train = make_patched_mnist('train', SPURIOUS_CORR_TRAIN, seed=seed)
    X_test_id, y_test_id = make_patched_mnist('test', SPURIOUS_CORR_TRAIN, seed=seed + 10000)
    X_test_ood, y_test_ood = make_patched_mnist('test', SPURIOUS_CORR_OOD, seed=seed + 20000)
    X_test_nopatch = remove_patch(X_test_id)
    if verbose: print(f"  Train: {len(X_train)}, Test: {len(X_test_id)}")

    torch.manual_seed(seed)
    model_sgd = SimpleCNN().to(device)
    optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=SGD_LR, momentum=SGD_MOMENTUM)
    if verbose: print(f"  Training SGD...")
    train_model(model_sgd, X_train, y_train, optimizer_sgd, N_EPOCHS, verbose=verbose, label="SGD")

    torch.manual_seed(seed)
    model_adam = SimpleCNN().to(device)
    optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=ADAM_LR)
    if verbose: print(f"  Training Adam...")
    train_model(model_adam, X_train, y_train, optimizer_adam, N_EPOCHS, verbose=verbose, label="Adam")

    sgd_acc_id = evaluate_accuracy(model_sgd, X_test_id, y_test_id)
    sgd_acc_ood = evaluate_accuracy(model_sgd, X_test_ood, y_test_ood)
    sgd_acc_nopatch = evaluate_accuracy(model_sgd, X_test_nopatch, y_test_id)
    adam_acc_id = evaluate_accuracy(model_adam, X_test_id, y_test_id)
    adam_acc_ood = evaluate_accuracy(model_adam, X_test_ood, y_test_ood)
    adam_acc_nopatch = evaluate_accuracy(model_adam, X_test_nopatch, y_test_id)

    if verbose:
        print(f"\n  Results:")
        print(f"    {'':15s} {'ID':>8s} {'OOD':>8s} {'NoPatch':>8s} {'OOD Drop':>9s} {'Patch Rel':>10s}")
        print(f"    {'SGD':15s} {sgd_acc_id:8.4f} {sgd_acc_ood:8.4f} {sgd_acc_nopatch:8.4f} "
              f"{sgd_acc_id-sgd_acc_ood:9.4f} {sgd_acc_id-sgd_acc_nopatch:10.4f}")
        print(f"    {'Adam':15s} {adam_acc_id:8.4f} {adam_acc_ood:8.4f} {adam_acc_nopatch:8.4f} "
              f"{adam_acc_id-adam_acc_ood:9.4f} {adam_acc_id-adam_acc_nopatch:10.4f}")

    result = {
        'sgd_acc_id': sgd_acc_id, 'sgd_acc_ood': sgd_acc_ood, 'sgd_acc_nopatch': sgd_acc_nopatch,
        'adam_acc_id': adam_acc_id, 'adam_acc_ood': adam_acc_ood, 'adam_acc_nopatch': adam_acc_nopatch,
    }
    if keep_examples:
        result.update({
            'X_train': X_train, 'y_train': y_train,
            'X_test_ood': X_test_ood, 'y_test_ood': y_test_ood,
        })
    return result


def main():
    setup_plotting()
    SAVE_DIR.mkdir(exist_ok=True)
    print("=" * 60)
    print("EXPERIMENT 2: MNIST with Spurious Colored Patch")
    print(f"Seeds: {SEEDS}")
    print(f"Spurious corr: train={SPURIOUS_CORR_TRAIN}, OOD={SPURIOUS_CORR_OOD}")
    print("=" * 60)

    all_results = []
    example_result = None
    for idx, seed in enumerate(SEEDS):
        result = run_single_seed(
            seed,
            verbose=(seed == SEEDS[0]),
            keep_examples=(idx == 0),
        )
        if idx == 0:
            example_result = result
        if seed != SEEDS[0]:
            print(f"  Seed {seed}: SGD ID={result['sgd_acc_id']:.4f} OOD={result['sgd_acc_ood']:.4f}, "
                  f"Adam ID={result['adam_acc_id']:.4f} OOD={result['adam_acc_ood']:.4f}")
        all_results.append(result)

    print(f"\n{'='*60}")
    print(f"Summary across {len(SEEDS)} seeds:")
    print(f"  {'':15s} {'ID':>12s} {'OOD':>12s} {'NoPatch':>12s} {'OOD Drop':>12s} {'Patch Rel':>12s}")
    sgd_id = np.mean([r['sgd_acc_id'] for r in all_results])
    sgd_ood = np.mean([r['sgd_acc_ood'] for r in all_results])
    sgd_np = np.mean([r['sgd_acc_nopatch'] for r in all_results])
    adam_id = np.mean([r['adam_acc_id'] for r in all_results])
    adam_ood = np.mean([r['adam_acc_ood'] for r in all_results])
    adam_np = np.mean([r['adam_acc_nopatch'] for r in all_results])
    print(f"  {'SGD':15s} {sgd_id:12.4f} {sgd_ood:12.4f} {sgd_np:12.4f} {sgd_id-sgd_ood:12.4f} {sgd_id-sgd_np:12.4f}")
    print(f"  {'Adam':15s} {adam_id:12.4f} {adam_ood:12.4f} {adam_np:12.4f} {adam_id-adam_ood:12.4f} {adam_id-adam_np:12.4f}")

    print(f"\nGenerating figures...")
    plot_figure9_samples(example_result['X_train'], example_result['y_train'],
                         example_result['X_test_ood'], example_result['y_test_ood'],
                         SAVE_DIR / "fig9_patched_mnist_samples.png")
    plot_figure10_id_vs_ood(all_results, SAVE_DIR / "fig10_id_vs_ood.png")
    plot_figure11_patch_reliance(all_results, SAVE_DIR / "fig11_patch_reliance.png")
    plot_figure12_summary(all_results, SAVE_DIR / "fig12_summary.png")

    print(f"\n{'='*60}")
    print("EXPERIMENT 2 COMPLETE")
    print(f"Figures saved to {SAVE_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
