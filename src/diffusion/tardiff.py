import torch
import torch.nn as nn


def _normalize_grads(grads, eps=1e-6):
    total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    return [g / (total_norm + eps) for g in grads]

def compute_influence_cache(classifier, loader, device='cpu'):
    """Stage 2: Calcule le gradient d'influence (G) normalisé sur le dataset de guidance."""
    classifier.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    params = list(classifier.parameters())
    total_loss = torch.tensor(0.0, device=device)

    print("--- Computing Influence Gradient Cache (G) ---")
    for x_guide, y_guide in loader:
        x_guide, y_guide = x_guide.to(device), y_guide.to(device)
        pred = classifier(x_guide)
        total_loss += criterion(pred, y_guide)

    grads = torch.autograd.grad(total_loss, params, allow_unused=True)
    filtered = [(p, g) for p, g in zip(params, grads) if g is not None]
    if not filtered:
        raise ValueError("No parameter received gradient!")

    _, filtered_grads = zip(*filtered)
    normed_grads = _normalize_grads(filtered_grads)

    G_accum = {
        name: g
        for (name, _), g in zip(classifier.named_parameters(), normed_grads)
        if g is not None
    }

    return G_accum

def tardiff_sample(model, scheduler, classifier, G_cache, n_samples=200, target_class=1, w=10.0, device='cpu'):
    """Stage 3: Sampling avec guidage par influence (gradients normalisés)."""
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
                    grads_x = torch.autograd.grad(loss_clf,
                                                  classifier.parameters(),
                                                  create_graph=True,
                                                  allow_unused=True)
                    filtered = [(p, g) for p, g in zip(classifier.parameters(), grads_x) if g is not None]
                    if not filtered:
                        raise ValueError("No parameter received gradient!")
                    _, grads_x = zip(*filtered)
                    grads_x = _normalize_grads(grads_x)

                    influence_score = 0
                    for (name, _), g_x in zip(classifier.named_parameters(), grads_x):
                        if name in G_cache:
                            influence_score += torch.sum(g_x * G_cache[name])
                    
                    # Gradient de l'influence par rapport à l'input x
                    J = torch.autograd.grad(influence_score, x_in)[0]
                    J = J.detach()
            
            sigma_t = torch.sqrt(beta)
            
            # 3. Mise à jour guidée
            x = x_recon + (w * J) + sigma_t * noise
            
    return x.detach().cpu().numpy()