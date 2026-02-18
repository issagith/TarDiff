import torch
import numpy as np
from dataclasses import dataclass
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader


@dataclass
class ToyConfig:
    n_samples: int = 10000
    batch_size: int = 128
    test_samples: int = 2000


def make_imbalanced_moons(cfg: ToyConfig, device='cpu'):
    """
    Dataset 2D non-linéaire et fortement déséquilibré (95/5).
    Classe 0 : grosse lune (make_moons)
    Classe 1 : petit cluster proche de la classe 0
    """
    minority_ratio = 0.05
    n_min = int(cfg.n_samples * minority_ratio)
    n_maj = cfg.n_samples - n_min

    # Majority class: moon-shaped distribution
    X_maj, _ = make_moons(n_samples=n_maj, noise=0.1)
    y_maj = np.zeros(n_maj, dtype=int)

    # Minority class: small cluster near the moon
    mean_min = np.array([0.5, 0.2])
    cov_min = np.eye(2) * 0.02
    X_min = np.random.multivariate_normal(mean_min, cov_min, size=n_min)
    y_min = np.ones(n_min, dtype=int)

    X = np.vstack([X_maj, X_min])
    y = np.hstack([y_maj, y_min])

    # global normalization
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std

    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.long).to(device)

    # Split train/guide
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_guide, y_guide = X[n_train:], y[n_train:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    guide_loader = DataLoader(TensorDataset(X_guide, y_guide), batch_size=len(X_guide), shuffle=False)

    # Test set
    n_min_t = cfg.test_samples // 20  # 5%
    n_maj_t = cfg.test_samples - n_min_t
    X_maj_t, _ = make_moons(n_samples=n_maj_t, noise=0.1)
    y_maj_t = np.zeros(n_maj_t, dtype=int)
    X_min_t = np.random.multivariate_normal(mean_min, cov_min, size=n_min_t)
    y_min_t = np.ones(n_min_t, dtype=int)

    X_test = np.vstack([X_maj_t, X_min_t])
    y_test = np.hstack([y_maj_t, y_min_t])
    X_test = (X_test - mean) / std

    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    stats = dict(mean=mean, std=std)
    return train_loader, guide_loader, (X_train, y_train), (X_test, y_test), stats