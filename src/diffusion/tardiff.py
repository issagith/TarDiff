import torch
import torch.nn as nn


def _normalize_grads(grads, eps=1e-6):
    """L2 normalization across all parameter gradients (as in official code)."""
    total_norm = torch.sqrt(sum((g ** 2).sum() for g in grads))
    return [g / (total_norm + eps) for g in grads]


def compute_influence_cache(classifier, loader, device='cpu'):
    """Compute the normalized influence gradient cache G on the guidance set."""
    classifier.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")

    params = list(classifier.parameters())
    total_loss = torch.tensor(0.0, device=device)

    for x_g, y_g in loader:
        x_g, y_g = x_g.to(device), y_g.to(device)
        logits = classifier(x_g)
        total_loss += crit(logits, y_g)

    grads = torch.autograd.grad(total_loss, params, allow_unused=True)
    filtered = [(p, g) for p, g in zip(params, grads) if g is not None]
    if not filtered:
        raise ValueError("No parameter received gradient!")

    filtered_params, filtered_grads = zip(*filtered)
    normed_grads = _normalize_grads(filtered_grads)

    G_cache = {
        name: g
        for (name, _), g in zip(classifier.named_parameters(), normed_grads)
        if g is not None
    }
    return G_cache


def tardiff_sample(model, scheduler, classifier, G_cache, n_samples=200,
                   target_class=1, w=10.0, device='cpu', input_dim=2):
    """
    TarDiff sampling with influence guidance.
    J = grad_x [ <norm(grad_phi l(x,y)), norm(G)> ]
    """
    model.eval()
    classifier.eval()
    crit_clf = nn.CrossEntropyLoss()

    with torch.no_grad():
        x = torch.randn((n_samples, input_dim)).to(device)
        y_cond = torch.full((n_samples,), target_class, dtype=torch.long).to(device)

        for i in reversed(range(1, scheduler.num_timesteps)):
            t = (torch.ones(n_samples) * i).long().to(device).view(-1, 1)
            predicted_noise = model(x, t, y_cond)

            alpha = scheduler.alpha[i]
            alpha_hat = scheduler.alpha_hat[i]
            beta = scheduler.beta[i]

            noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
            x_recon = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise)

            J = torch.zeros_like(x)
            if w != 0:
                with torch.enable_grad():
                    x_in = x.detach().clone()
                    x_in.requires_grad = True

                    out_clf = classifier(x_in)
                    loss = crit_clf(out_clf, y_cond)

                    grads_phi = torch.autograd.grad(loss, classifier.parameters(),
                                                    create_graph=True, allow_unused=True)
                    filtered = [(p, g) for p, g in zip(classifier.parameters(), grads_phi) if g is not None]
                    if not filtered:
                        raise ValueError("No parameter received gradient!")
                    _, grads_phi = zip(*filtered)
                    grads_phi = _normalize_grads(grads_phi)

                    influence_score = 0
                    for (name, _), g_phi in zip(classifier.named_parameters(), grads_phi):
                        if name in G_cache:
                            influence_score += torch.sum(g_phi * G_cache[name])

                    J = torch.autograd.grad(influence_score, x_in)[0]

                J = J.detach()

            sigma_t = torch.sqrt(beta)
            x = x_recon + (w * J) + sigma_t * noise

    return x.detach()