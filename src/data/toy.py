import torch
import numpy as np
from sklearn.datasets import make_blobs
from torch.utils.data import TensorDataset, DataLoader

def get_imbalanced_data(n_samples=2000, minority_ratio=0.01, batch_size=64, device='cpu'):
    """Génère un dataset d'entraînement déséquilibré."""
    n_minority = int(n_samples * minority_ratio)
    n_majority = n_samples - n_minority
    
    # Génération
    X_maj, y_maj = make_blobs(n_samples=n_majority, centers=[[0, 0]], cluster_std=1)
    X_min, y_min = make_blobs(n_samples=n_minority, centers=[[1, 1]], cluster_std=1)
    
    y_min[:] = 1 
    y_maj[:] = 0
    
    X = np.vstack([X_maj, X_min])
    y = np.hstack([y_maj, y_min])
    
    # Normalisation
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X = (X - mean) / std
    
    # Conversion Tensors
    tensor_x = torch.FloatTensor(X).to(device)
    tensor_y = torch.LongTensor(y).to(device)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    
    # Split Train / Guidance
    train_size = int(0.8 * len(dataset))
    train_set, guidance_set = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # Le guidance loader charge tout en un seul batch pour simplifier le calcul de G
    guidance_loader = DataLoader(guidance_set, batch_size=len(guidance_set), shuffle=False)
    
    return train_loader, guidance_loader, (mean, std)

def get_test_data(mean, std, n=1000, device='cpu'):
    """Génère un test set équilibré normalisé avec les stats du train."""
    X_maj, y_maj = make_blobs(n_samples=int(n*0.5), centers=[[-2, -2]], cluster_std=0.8)
    X_min, y_min = make_blobs(n_samples=int(n*0.5), centers=[[2, 2]], cluster_std=0.8)
    y_min[:] = 1; y_maj[:] = 0
    X = np.vstack([X_maj, X_min])
    y = np.hstack([y_maj, y_min])
    
    # Appliquer la même normalisation que le train
    X = (X - mean) / std
    
    tensor_x = torch.FloatTensor(X).to(device)
    tensor_y = torch.LongTensor(y).to(device)
    
    return tensor_x, tensor_y