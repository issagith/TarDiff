import torch
import torch.nn as nn

def compute_influence_cache(classifier, loader, device='cpu'):
    """Stage 2: Calcule le gradient d'influence moyen (G) sur le dataset de guidance."""
    classifier.eval()
    criterion = nn.CrossEntropyLoss()
    
    G_accum = {name: torch.zeros_like(param) for name, param in classifier.named_parameters()}
    total_samples = 0
    
    print("--- Computing Influence Gradient Cache (G) ---")
    for x_guide, y_guide in loader:
        x_guide, y_guide = x_guide.to(device), y_guide.to(device)
        pred = classifier(x_guide)
        loss = criterion(pred, y_guide)
        
        # Calcul des gradients
        grads = torch.autograd.grad(loss, classifier.parameters())
        
        for (name, _), g in zip(classifier.named_parameters(), grads):
            G_accum[name] += g
        
        total_samples += x_guide.size(0)
            
    # Normalisation
    for name in G_accum:
        G_accum[name] /= total_samples
        
    return G_accum

def tardiff_sample(model, scheduler, classifier, G_cache, n_samples=200, target_class=1, w=10.0, device='cpu'):
    """Stage 3: Sampling avec guidage par influence."""
    model.eval()
    classifier.eval()
    criterion_clf = nn.CrossEntropyLoss()
    
    print(f"Sampling with TarDiff Guidance (Class={target_class}, w={w})...")
    
    with torch.no_grad():
        x = torch.randn((n_samples, 2)).to(device)
        y = torch.full((n_samples,), target_class).to(device)
        
        for i in reversed(range(1, scheduler.num_timesteps)):
            t = (torch.ones(n_samples) * i).long().to(device).view(-1, 1)
            
            # 1. Prédiction du bruit standard
            predicted_noise = model(x, t, y)
            
            alpha = scheduler.alpha[i]
            alpha_hat = scheduler.alpha_hat[i]
            beta = scheduler.beta[i]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            
            x_recon = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise)
            
            # 2. Calcul du Guidage TarDiff (J)
            J = 0
            if w != 0:
                with torch.enable_grad():
                    x_in = x.detach().clone()
                    x_in.requires_grad = True
                    
                    out_clf = classifier(x_in)
                    loss_clf = criterion_clf(out_clf, y)
                    
                    # Gradient de second ordre
                    grads_x = torch.autograd.grad(loss_clf, classifier.parameters(), create_graph=True)
                    
                    influence_score = 0
                    for (name, param), g_x in zip(classifier.named_parameters(), grads_x):
                        if name in G_cache:
                            influence_score += torch.sum(g_x * G_cache[name])
                    
                    # Gradient de l'influence par rapport à l'input x
                    J = torch.autograd.grad(influence_score, x_in)[0]
                    J = J.detach()
            
            sigma_t = torch.sqrt(beta)
            
            # 3. Mise à jour guidée
            x = x_recon + (w * J) + sigma_t * noise
            
    return x.detach().cpu().numpy()