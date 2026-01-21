import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

# Imports locaux
from data.toy import get_imbalanced_data, get_test_data
from models.classifier import SimpleClassifier
from models.diffusion_net import SimpleDiffusionNet
from diffusion.scheduler import DDPMScheduler
from diffusion.tardiff import compute_influence_cache, tardiff_sample

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#SEED = 42
#torch.manual_seed(SEED)
#np.random.seed(SEED)

def train_clf(model, loader, epochs=100, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

def train_diff(model, loader, scheduler, epochs=300, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    print("--- Training Diffusion Model ---")
    for _ in range(epochs):
        for x, y in loader:
            opt.zero_grad()
            t = scheduler.sample_timesteps(x.shape[0]).view(-1, 1)
            x_t, noise = scheduler.noise_images(x, t)
            noise_pred = model(x_t, t, y)
            loss = crit(noise_pred, noise)
            loss.backward()
            opt.step()

def eval_pipeline(name, train_x, train_y, test_x, test_y):
    """Entraîne un classifieur vierge sur les données fournies et évalue."""
    model = SimpleClassifier().to(DEVICE)
    train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    
    # Train new classifier
    train_clf(model, train_dl, epochs=50, lr=0.01)
    
    # Eval
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        test_dl = DataLoader(TensorDataset(test_x, test_y), batch_size=64)
        for x, y in test_dl:
            pred = torch.argmax(model(x), dim=1)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
            
    f1 = f1_score(targets, preds, average='binary')
    print(f"[{name}] \t F1-Score (Min. Class): {f1:.4f}")
    return model

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

def main():
    # 1. Data
    train_loader, guidance_loader, (mean, std) = get_imbalanced_data(device=DEVICE)
    X_test, y_test = get_test_data(mean, std, device=DEVICE)
    
    # 2. Train Base Models
    clf_base = SimpleClassifier().to(DEVICE)
    print("--- Training Base Classifier ---")
    train_clf(clf_base, train_loader)
    
    diff_model = SimpleDiffusionNet().to(DEVICE)
    scheduler = DDPMScheduler(device=DEVICE)
    train_diff(diff_model, train_loader, scheduler)
    
    # 3. Compute Influence (Stage 2)
    G_cache = compute_influence_cache(clf_base, guidance_loader, device=DEVICE)
    
    # 4. Generate Synthetic Data
    n_gen = 1000
    
    # > Generation Standard (w=0)
    x_std_c0 = tardiff_sample(diff_model, scheduler, clf_base, G_cache, n_samples=n_gen, target_class=0, w=0.0, device=DEVICE)
    x_std_c1 = tardiff_sample(diff_model, scheduler, clf_base, G_cache, n_samples=n_gen, target_class=1, w=0.0, device=DEVICE)
    X_syn_std = torch.FloatTensor(np.vstack([x_std_c0, x_std_c1])).to(DEVICE)
    y_syn = torch.cat([torch.zeros(n_gen), torch.ones(n_gen)]).long().to(DEVICE)
    
    # > Generation TarDiff (w=100)
    x_tar_c0 = tardiff_sample(diff_model, scheduler, clf_base, G_cache, n_samples=n_gen, target_class=0, w=100.0, device=DEVICE)
    x_tar_c1 = tardiff_sample(diff_model, scheduler, clf_base, G_cache, n_samples=n_gen, target_class=1, w=100.0, device=DEVICE)
    X_syn_tar = torch.FloatTensor(np.vstack([x_tar_c0, x_tar_c1])).to(DEVICE)
    
    # 5. Evaluation & Plotting
    
    # CORRECTION : On ne peut pas faire .dataset.tensors sur un Subset.
    # On reconstruit X_real et y_real en itérant sur le loader d'entraînement.
    X_real_list = []
    y_real_list = []
    
    # On itère sur le train_loader pour récupérer toutes les données d'entraînement (mélangées)
    for x_batch, y_batch in train_loader:
        X_real_list.append(x_batch)
        y_real_list.append(y_batch)
        
    X_real = torch.cat(X_real_list).to(DEVICE)
    y_real = torch.cat(y_real_list).to(DEVICE)
    
    scenarios = {
        'Real Only': (X_real, y_real),
        # On ajoute les données synthétiques à la fin des données réelles
        'Real + DDPM': (torch.cat([X_real, X_syn_std[n_gen:]]), torch.cat([y_real, y_syn[n_gen:]])),
        'Real + TarDiff': (torch.cat([X_real, X_syn_tar[n_gen:]]), torch.cat([y_real, y_syn[n_gen:]])),
        'Synth Only (DDPM)': (X_syn_std, y_syn),
        'Synth Only (TarDiff)': (X_syn_tar, y_syn)
    }
    
    models = {}
    print("\n=== PERFORMANCE EVALUATION ===")
    for name, (tx, ty) in scenarios.items():
        models[name] = eval_pipeline(name, tx, ty, X_test, y_test)
        
    # Visualization
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    # Passage en numpy pour l'affichage
    X_test_np = X_test.cpu().numpy()
    y_test_np = y_test.cpu().numpy()
    
    for ax, name in zip(axes, scenarios.keys()):
        plot_boundary(models[name], X_test_np, y_test_np, name, ax)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()