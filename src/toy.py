"""
TarDiff – Toy 2D Experiment (standalone script).

Reproduces the full notebook pipeline in one shot:
  1. Imbalanced make_moons data (95/5)
  2. Classifier + conditional DDPM training
  3. Influence cache computation
  4. Guidance sweep (w values)
  5. TSTR evaluation
  6. TSRTR evaluation
  7. Ratio experiments (variable & fixed total)
  8. Visualization (decision boundaries, scatter, metrics)
"""

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
)

# Imports locaux
from data.toy import ToyConfig, make_imbalanced_moons
from models.classifier import SimpleClassifier
from models.diffusion_net import SimpleDiffusionNet
from diffusion.scheduler import DDPMScheduler
from diffusion.tardiff import compute_influence_cache, tardiff_sample

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Configs ──────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    clf_epochs: int = 30
    diff_epochs: int = 50
    clf_lr: float = 1e-3
    diff_lr: float = 1e-3
    timesteps: int = 50


# ── Training helpers ─────────────────────────────────────────────────────────
def train_classifier(model, loader, cfg: TrainConfig):
    model.train()
    opt = optim.Adam(model.parameters(), lr=cfg.clf_lr)
    crit = nn.CrossEntropyLoss()
    for _ in range(cfg.clf_epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()


def train_diffusion(model, loader, scheduler: DDPMScheduler, cfg: TrainConfig):
    model.train()
    opt = optim.Adam(model.parameters(), lr=cfg.diff_lr)
    crit = nn.MSELoss()
    for _ in range(cfg.diff_epochs):
        for x, y in loader:
            opt.zero_grad()
            t = scheduler.sample_timesteps(x.shape[0]).view(-1, 1)
            x_t, noise = scheduler.noise(x, t)
            noise_pred = model(x_t, t, y)
            loss = crit(noise_pred, noise)
            loss.backward()
            opt.step()


def eval_classifier(model, X, y):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
    y_np, p_np, pr_np = y.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy()
    return dict(
        acc=accuracy_score(y_np, p_np),
        recall_0=recall_score(y_np, p_np, pos_label=0),
        precision_0=precision_score(y_np, p_np, pos_label=0),
        f1_0=f1_score(y_np, p_np, pos_label=0),
        recall_1=recall_score(y_np, p_np, pos_label=1),
        precision_1=precision_score(y_np, p_np, pos_label=1),
        f1_1=f1_score(y_np, p_np, pos_label=1),
        auroc=roc_auc_score(y_np, pr_np),
        auprc=average_precision_score(y_np, pr_np),
    )


# ── Sampling helpers ─────────────────────────────────────────────────────────
def guidance_sweep(model, scheduler, classifier, G_cache, w_values, n_samples, target_class):
    samples_by_w = {}
    for w in w_values:
        samples_by_w[w] = tardiff_sample(
            model, scheduler, classifier, G_cache,
            n_samples=n_samples, target_class=target_class, w=w, device=DEVICE,
        )
    return samples_by_w


def sample_pool(pool, n):
    if n <= 0:
        return pool[:0]
    idx = torch.randperm(len(pool), device=pool.device)[:n]
    return pool[idx]


# ── Visualization ────────────────────────────────────────────────────────────
def plot_boundary(model, X, y, title, ax):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]).to(DEVICE)
    with torch.no_grad():
        Z = torch.argmax(model(grid), dim=1).cpu().numpy().reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', alpha=0.6, cmap='coolwarm')
    ax.set_title(title, fontsize=10)


def plot_guidance_sweep(X_train, samples_dict, w_values, color, class_label, title_prefix=""):
    fig, axes = plt.subplots(1, len(w_values), figsize=(5 * len(w_values), 4))
    if len(w_values) == 1:
        axes = [axes]
    for ax, w in zip(axes, w_values):
        ax.scatter(X_train[:, 0].cpu(), X_train[:, 1].cpu(), c='gray', s=5, alpha=0.1, label='Real Train')
        s = samples_dict[w]
        ax.scatter(s[:, 0].cpu(), s[:, 1].cpu(), c=color, s=15, alpha=0.6, label=f'Generated (Class {class_label})')
        ax.set_title(f"{title_prefix}w={w}")
        ax.legend()
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"Device: {DEVICE}")

    # ── 1. Data ──────────────────────────────────────────────────────────────
    cfg = ToyConfig()
    train_loader, guide_loader, (X_train, y_train), (X_test, y_test), stats = \
        make_imbalanced_moons(cfg, device=DEVICE)
    print(f"Train size: {X_train.shape}  Guide size: {next(iter(guide_loader))[0].shape}")
    print(f"Test size: {X_test.shape}")

    cfg_train = TrainConfig()
    scheduler = DDPMScheduler(num_timesteps=cfg_train.timesteps, device=DEVICE)

    # ── 2. Train classifier ──────────────────────────────────────────────────
    print("\n--- Training Base Classifier ---")
    clf = SimpleClassifier().to(DEVICE)
    train_classifier(clf, train_loader, cfg_train)
    metrics_real = eval_classifier(clf, X_test, y_test)
    print(f"Downstream on real : {metrics_real}")

    # ── 3. Train diffusion ───────────────────────────────────────────────────
    print("\n--- Training Diffusion Model ---")
    diff_model = SimpleDiffusionNet().to(DEVICE)
    train_diffusion(diff_model, train_loader, scheduler, cfg_train)
    print("Diffusion trained.")

    # ── 4. Influence cache ───────────────────────────────────────────────────
    print("\n--- Computing Influence Cache ---")
    G_cache = compute_influence_cache(clf, guide_loader, device=DEVICE)
    print("Influence cache norms :", {k: float(v.norm().item()) for k, v in G_cache.items()})

    # ── 5. Guidance sweep ────────────────────────────────────────────────────
    N_GEN = 5000
    w_values = [0.0, 1, 5, 10, 20, 50, 100, 500, 1000, 2000]
    print(f"\nSampling {N_GEN} points per w ...")

    samples_0 = guidance_sweep(diff_model, scheduler, clf, G_cache,
                               w_values=w_values, n_samples=N_GEN, target_class=0)
    samples_1 = guidance_sweep(diff_model, scheduler, clf, G_cache,
                               w_values=w_values, n_samples=N_GEN, target_class=1)

    plot_guidance_sweep(X_train, samples_0, w_values, "blue", 0)
    plot_guidance_sweep(X_train, samples_1, w_values, "red", 1)

    # ── 6. TSTR ──────────────────────────────────────────────────────────────
    print("\n=== Downstream Utility – TSTR ===")
    rows_tstr = [{
        "w": "real-only",
        "acc": metrics_real["acc"],
        "precision": metrics_real["precision_1"],
        "recall": metrics_real["recall_1"],
        "f1": metrics_real["f1_1"],
        "auroc": metrics_real["auroc"],
        "auprc": metrics_real["auprc"],
    }]
    results_tstr = {}
    preds_tstr = {}

    for w in w_values:
        X_syn_0 = samples_0[w]
        X_syn_1 = samples_1[w]
        y_syn_0 = torch.zeros(len(X_syn_0), dtype=torch.long).to(DEVICE)
        y_syn_1 = torch.ones(len(X_syn_1), dtype=torch.long).to(DEVICE)
        X_syn = torch.cat([X_syn_0, X_syn_1])
        y_syn = torch.cat([y_syn_0, y_syn_1])

        loader_syn = DataLoader(TensorDataset(X_syn, y_syn), batch_size=64, shuffle=True)
        clf_syn = SimpleClassifier().to(DEVICE)
        train_classifier(clf_syn, loader_syn, TrainConfig(clf_epochs=30))
        metrics_syn = eval_classifier(clf_syn, X_test, y_test)

        with torch.no_grad():
            logits = clf_syn(X_test)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            preds = torch.argmax(logits, dim=1).detach().cpu()

        preds_tstr[w] = {"preds": preds, "probs": probs}
        results_tstr[w] = metrics_syn

        rows_tstr.append({
            "w": w if w != 0.0 else "DDPM",
            "acc": metrics_syn["acc"],
            "precision": metrics_syn["precision_1"],
            "recall": metrics_syn["recall_1"],
            "f1": metrics_syn["f1_1"],
            "auroc": metrics_syn["auroc"],
            "auprc": metrics_syn["auprc"],
        })

    results_tstr_df = pd.DataFrame(rows_tstr)
    best_w_tstr = max(results_tstr, key=lambda k: results_tstr[k]["auroc"])
    print(f"\nTSTR results:\n{results_tstr_df.to_string(index=False)}")
    print(f"Best w (AUROC): {best_w_tstr}  →  {results_tstr[best_w_tstr]}")

    # ── 7. TSRTR ─────────────────────────────────────────────────────────────
    print("\n=== Downstream Utility – TSRTR ===")
    rows_tsrtr = [{
        "w": "real-only",
        "acc": metrics_real["acc"],
        "precision": metrics_real["precision_1"],
        "recall": metrics_real["recall_1"],
        "f1": metrics_real["f1_1"],
        "auroc": metrics_real["auroc"],
        "auprc": metrics_real["auprc"],
    }]
    results_tsrtr = {}
    preds_tsrtr = {}

    for w in w_values:
        X_syn_0 = samples_0[w]
        X_syn_1 = samples_1[w]
        y_syn_0 = torch.zeros(len(X_syn_0), dtype=torch.long).to(DEVICE)
        y_syn_1 = torch.ones(len(X_syn_1), dtype=torch.long).to(DEVICE)
        X_syn = torch.cat([X_syn_0, X_syn_1])
        y_syn = torch.cat([y_syn_0, y_syn_1])

        X_aug = torch.cat([X_train, X_syn])
        y_aug = torch.cat([y_train, y_syn])

        loader_aug = DataLoader(TensorDataset(X_aug, y_aug), batch_size=64, shuffle=True)
        clf_aug = SimpleClassifier().to(DEVICE)
        train_classifier(clf_aug, loader_aug, TrainConfig(clf_epochs=30))
        metrics_aug = eval_classifier(clf_aug, X_test, y_test)

        with torch.no_grad():
            logits = clf_aug(X_test)
            probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
            preds = torch.argmax(logits, dim=1).detach().cpu()

        preds_tsrtr[w] = {"preds": preds, "probs": probs}
        results_tsrtr[w] = metrics_aug

        rows_tsrtr.append({
            "w": w,
            "acc": metrics_aug["acc"],
            "precision": metrics_aug["precision_1"],
            "recall": metrics_aug["recall_1"],
            "f1": metrics_aug["f1_1"],
            "auroc": metrics_aug["auroc"],
            "auprc": metrics_aug["auprc"],
        })

    results_tsrtr_df = pd.DataFrame(rows_tsrtr)
    best_w_tsrtr = max(results_tsrtr, key=lambda k: results_tsrtr[k]["f1_1"])
    print(f"\nTSRTR results:\n{results_tsrtr_df.to_string(index=False)}")
    print(f"Best w (F1): {best_w_tsrtr}  →  {results_tsrtr[best_w_tsrtr]}")

    # ── 8. Ratio experiment (variable total) ─────────────────────────────────
    print("\n=== Impact proportion synthétique (TSRTR) ===")
    w_ratio = best_w_tsrtr
    ratios = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    n_real_0 = int((y_train == 0).sum().item())
    n_real_1 = int((y_train == 1).sum().item())
    n_runs = 10

    rows_ratio = []
    for run in range(n_runs):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        for r in ratios:
            n_syn_0 = min(int(n_real_0 * r), len(samples_0[w_ratio]))
            n_syn_1 = min(int(n_real_1 * r), len(samples_1[w_ratio]))
            X_syn_0 = sample_pool(samples_0[w_ratio], n_syn_0)
            X_syn_1 = sample_pool(samples_1[w_ratio], n_syn_1)

            if n_syn_0 + n_syn_1 > 0:
                X_syn = torch.cat([X_syn_0, X_syn_1])
                y_syn = torch.cat([
                    torch.zeros(len(X_syn_0), dtype=torch.long).to(DEVICE),
                    torch.ones(len(X_syn_1), dtype=torch.long).to(DEVICE),
                ])
                X_aug = torch.cat([X_train, X_syn])
                y_aug = torch.cat([y_train, y_syn])
            else:
                X_aug, y_aug = X_train, y_train

            loader_aug = DataLoader(TensorDataset(X_aug, y_aug), batch_size=64, shuffle=True)
            clf_aug = SimpleClassifier().to(DEVICE)
            train_classifier(clf_aug, loader_aug, TrainConfig(clf_epochs=30))
            metrics_aug = eval_classifier(clf_aug, X_test, y_test)

            rows_ratio.append({
                "run": run,
                "ratio_syn/real": r,
                "n_syn": int(n_syn_0 + n_syn_1),
                "acc": metrics_aug["acc"],
                "precision": metrics_aug["precision_1"],
                "recall": metrics_aug["recall_1"],
                "f1": metrics_aug["f1_1"],
                "auroc": metrics_aug["auroc"],
                "auprc": metrics_aug["auprc"],
                "n_total": len(X_aug),
            })

    ratio_df = pd.DataFrame(rows_ratio)
    ratio_summary = ratio_df.groupby("ratio_syn/real")[
        ["acc", "precision", "recall", "f1", "auroc", "auprc", "n_total"]
    ].agg(["mean", "std"]).reset_index()
    print(f"\nRatio summary (variable total):\n{ratio_summary.to_string()}")

    # Plot ratio metrics
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    for ax, metric in zip(axes, ["acc", "precision", "recall", "f1", "auroc", "auprc"]):
        mean_vals = ratio_summary[(metric, "mean")].values
        std_vals = ratio_summary[(metric, "std")].values
        x = ratio_summary[("ratio_syn/real", "")].to_numpy()
        ax.errorbar(x, mean_vals, yerr=std_vals, marker="o", capsize=3)
        ax.set_title(metric.upper())
        ax.set_xlabel("ratio synth/real")
        ax.grid(alpha=0.3)
    plt.suptitle("Variable total – TSRTR metrics vs ratio", y=1.02)
    plt.tight_layout()
    plt.show()

    # Best run
    best_idx = ratio_df["f1"].idxmax()
    best_row = ratio_df.loc[best_idx]
    print(f"\nBest run: ratio={best_row['ratio_syn/real']}, f1={best_row['f1']:.4f}, "
          f"acc={best_row['acc']:.4f}, auroc={best_row['auroc']:.4f}")

    # ── 9. Ratio experiment (fixed total) ────────────────────────────────────
    print("\n=== Impact proportion synthétique (TSRTR) – TOTAL FIXE ===")
    cfg_ratio = ToyConfig(n_samples=18000)
    train_loader_ratio, guide_loader_ratio, (X_train_ratio, y_train_ratio), \
        (X_test_ratio, y_test_ratio), stats_ratio = make_imbalanced_moons(cfg_ratio, device=DEVICE)

    n_runs_fixed = 5
    N_total = len(X_train_ratio)
    p_real_1 = float((y_train_ratio == 1).float().mean().item())
    p_real_0 = 1.0 - p_real_1

    idx_real_0 = torch.where(y_train_ratio == 0)[0]
    idx_real_1 = torch.where(y_train_ratio == 1)[0]

    def sample_from_indices(idxs, n):
        if n <= 0:
            return idxs[:0]
        perm = torch.randperm(len(idxs), device=idxs.device)[:n]
        return idxs[perm]

    rows_ratio_fixed = []
    for run in range(n_runs_fixed):
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        for r in ratios:
            if r == 0:
                n_syn = 0
                n_real_keep = N_total
            else:
                n_real_keep = int(round(N_total / (1.0 + r)))
                n_syn = N_total - n_real_keep

            n_real_keep_1 = int(round(n_real_keep * p_real_1))
            n_real_keep_0 = n_real_keep - n_real_keep_1

            sel0 = sample_from_indices(idx_real_0, n_real_keep_0)
            sel1 = sample_from_indices(idx_real_1, n_real_keep_1)

            X_real_keep = torch.cat([X_train_ratio[sel0], X_train_ratio[sel1]], dim=0)
            y_real_keep = torch.cat([
                torch.zeros(len(sel0), dtype=torch.long, device=DEVICE),
                torch.ones(len(sel1), dtype=torch.long, device=DEVICE),
            ], dim=0)

            n_syn_1 = int(round(n_syn * p_real_1))
            n_syn_0 = n_syn - n_syn_1
            n_syn_0 = min(n_syn_0, len(samples_0[w_ratio]))
            n_syn_1 = min(n_syn_1, len(samples_1[w_ratio]))

            X_syn_0 = sample_pool(samples_0[w_ratio], n_syn_0)
            X_syn_1 = sample_pool(samples_1[w_ratio], n_syn_1)

            if n_syn_0 + n_syn_1 > 0:
                X_syn = torch.cat([X_syn_0, X_syn_1], dim=0)
                y_syn = torch.cat([
                    torch.zeros(len(X_syn_0), dtype=torch.long, device=DEVICE),
                    torch.ones(len(X_syn_1), dtype=torch.long, device=DEVICE),
                ], dim=0)
                X_aug = torch.cat([X_real_keep, X_syn], dim=0)
                y_aug = torch.cat([y_real_keep, y_syn], dim=0)
            else:
                X_aug, y_aug = X_real_keep, y_real_keep

            loader_aug = DataLoader(TensorDataset(X_aug, y_aug), batch_size=64, shuffle=True)
            clf_aug = SimpleClassifier().to(DEVICE)
            train_classifier(clf_aug, loader_aug, TrainConfig(clf_epochs=30))
            metrics_aug = eval_classifier(clf_aug, X_test_ratio, y_test_ratio)

            rows_ratio_fixed.append({
                "run": run,
                "ratio_syn/real": r,
                "n_total": int(len(X_aug)),
                "n_real_keep": int(len(X_real_keep)),
                "n_syn": int(n_syn_0 + n_syn_1),
                "acc": metrics_aug["acc"],
                "precision": metrics_aug["precision_1"],
                "recall": metrics_aug["recall_1"],
                "f1": metrics_aug["f1_1"],
                "auroc": metrics_aug["auroc"],
                "auprc": metrics_aug["auprc"],
            })

    ratio_fixed_df = pd.DataFrame(rows_ratio_fixed)
    ratio_fixed_summary = ratio_fixed_df.groupby("ratio_syn/real")[
        ["acc", "precision", "recall", "f1", "auroc", "auprc", "n_total"]
    ].agg(["mean", "std"]).reset_index()
    print(f"\nRatio summary (fixed total):\n{ratio_fixed_summary.to_string()}")

    # Plot fixed-total ratio metrics
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()
    for ax, metric in zip(axes, ["acc", "precision", "recall", "f1", "auroc", "auprc"]):
        mean_vals = ratio_fixed_summary[(metric, "mean")].values
        std_vals = ratio_fixed_summary[(metric, "std")].values
        x = ratio_fixed_summary[("ratio_syn/real", "")].to_numpy()
        ax.errorbar(x, mean_vals, yerr=std_vals, marker="o", capsize=3)
        ax.set_title(metric.upper())
        ax.set_xlabel("ratio synth/real")
        ax.grid(alpha=0.3)
    plt.suptitle("Fixed total – TSRTR metrics vs ratio", y=1.02)
    plt.tight_layout()
    plt.show()

    print("\n✅ Done.")


if __name__ == "__main__":
    main()