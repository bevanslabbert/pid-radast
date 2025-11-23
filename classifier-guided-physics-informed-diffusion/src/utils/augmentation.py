import torch
import torch.nn.functional as F

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=10):
    device = next(model.parameters()).device

    images = images.clone().detach().to(device).float()
    labels = labels.clone().detach().to(device)
    ori_images = images.clone().detach()

    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = images.grad.sign()

        images = images + alpha * grad
        delta = torch.clamp(images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + delta, min=0, max=1).detach()

    return images

def pgd_attack_early_stop(model, x_t, t, y,
                          epsilon=0.03,   # max l_inf perturbation in pixel space [0,1]
                          alpha=0.01,     # step size (pixel space)
                          num_steps=10,
                          random_start=True,
                          clamp=(0.0, 1.0),
                          device=None,
                          verbose=False):
    """
    PGD attack (untargeted) for a time-dependent classifier model(x, t).
    Returns adversarial image tensor of same shape as x_t.

    Args:
      model: callable (x, t) -> logits. model should be in eval() mode.
      x_t: noisy image at timestep t (tensor, shape [B,C,H,W], values in [0,1])
      t: timestep (can be scalar or tensor broadcastable to batch)
      y: true labels (long tensor, shape [B])
      epsilon, alpha: in pixel scale [0..1] (if your model uses normalized input, convert accordingly)
    """
    # before attack
    model.eval()
    with torch.no_grad():
        clean_logits = model(x_t, t)
        clean_preds = clean_logits.argmax(dim=1)

    if device is None:
        device = x_t.device
    model = model.to(device)
    model.eval()

    x_t = x_t.clone().detach().to(device)
    y = y.clone().detach().to(device)
    t = t if isinstance(t, torch.Tensor) else torch.tensor(t, device=device)

    # initialize delta (perturbation) in pixel space
    if random_start:
        delta = torch.empty_like(x_t).uniform_(-epsilon, epsilon).to(device)
        # make sure we stay in valid pixel range
        delta = torch.clamp(x_t + delta, clamp[0], clamp[1]) - x_t
    else:
        delta = torch.zeros_like(x_t).to(device)
    delta.requires_grad_(True)

    # keep track of which samples are already successful (so diagnostics can show progress)
    successful = torch.zeros(x_t.size(0), dtype=torch.bool, device=device)

    for step in range(num_steps):
        # forward on (x_t + delta)
        inp = x_t + delta
        logits = model(inp, t)
        preds = logits.detach().argmax(dim=1)

        # early stopping check (per-batch: stop if all are fooled)
        newly_success = (preds != y)
        if newly_success.all():
            print(f"[PGD] All samples fooled at step {step}")
            if verbose:
                print(f"[PGD] All samples fooled at step {step}")
            break

        # compute scalar loss (we want to maximize the classification loss)
        loss = F.cross_entropy(logits, y)

        # compute gradient of loss wrt delta only (avoids model param grads)
        grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]

        # diagnostics
        if verbose:
            grad_norm = grad.view(grad.shape[0], -1).norm(p=2, dim=1).mean().item()
            delta_inf = delta.detach().view(delta.shape[0], -1).abs().max(dim=1)[0].mean().item()
            batch_acc = (preds == y).float().mean().item()
            print(f"[PGD] step={step} loss={loss.item():.4f} grad_norm={grad_norm:.6f} "
                  f"delta_inf={delta_inf:.6f} batch_acc={batch_acc*100:.2f}%")

        # gradient ascent update on delta, projection to l_inf ball, and clamp image range
        with torch.no_grad():
            delta += alpha * torch.sign(grad)
            # project to l_inf epsilon ball
            delta.clamp_(-epsilon, epsilon)
            # ensure valid image range after adding delta
            delta = torch.clamp(x_t + delta, clamp[0], clamp[1]) - x_t
            # re-enable grad for next iter
            delta.requires_grad_(True)

    x_adv = torch.clamp(x_t + delta.detach(), clamp[0], clamp[1])
    return x_adv

def get_noisy_image(x, t, alphas_cumprod):
    """
    Add Gaussian noise to image x at timestep t
    x_t = sqrt(alpha_bar_t) * x + sqrt(1 - alpha_bar_t) * epsilon
    """
    # Get alpha_bar for timestep t
    alpha_bar_t = alphas_cumprod[t].view(-1, 1, 1, 1)

    # Sample noise
    epsilon = torch.randn_like(x)

    # Create noisy image
    x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * epsilon

    return x_t

def get_max_timestep(epoch, total_epochs, max_t=1000):
    """Linearly increase max timestep during train"""
    return int((epoch / total_epochs) * max_t)
