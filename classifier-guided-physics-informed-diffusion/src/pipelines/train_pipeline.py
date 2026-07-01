import gc
import json
import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.datasets.mirabest.MiraBestFITS import MiraBestFITS
from src.models.diffusion import build_diffusion_components, eval_epoch
from src.models.pid import estimate_x0, symmetry_loss, nonnegativity_loss
from torchvision.models import resnet18
from src.models.simple_cnn import SimpleCNN
from src.models.time_dependent_resnet import TimeDependentResNet
from src.utils.augmentation import pgd_attack_early_stop, get_max_timestep, get_noisy_image
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.utils.metrics import generate_class_samples, compute_fid_kid, compute_pixel_pdf
from src.utils.visualization import (
    save_training_plot, save_generative_metrics_plot, save_pixel_pdf_history_plot,
    save_pid_training_plots, save_comparison_grid,
)

CHECKPOINT_DIR = 'checkpoints'


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def evaluate_loss(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            total_loss += criterion(model(inputs), labels).item()
    return total_loss / len(dataloader)


def fits_to_linear(x_fits, dataset):
    """Convert FITS log-SNR normalised tensor to linear-normalised [-1, 1].

    Fully differentiable — safe to use inside a training step where gradients
    must flow back through x_0_pred to the UNet.
    """
    peak_log = dataset.median_peak_log
    return torch.sign(x_fits) * torch.expm1(torch.abs(x_fits) * peak_log) / math.expm1(peak_log)


def _restore_rng(ckpt):
    if ckpt.get('rng_state') is not None:
        torch.set_rng_state(ckpt['rng_state'].to('cpu').to(torch.uint8))
    if ckpt.get('cuda_rng_state') is not None:
        cuda_state = ckpt['cuda_rng_state']
        if isinstance(cuda_state, torch.Tensor):
            torch.cuda.set_rng_state(cuda_state.to('cpu').to(torch.uint8))
        else:
            torch.cuda.set_rng_state_all([s.to('cpu').to(torch.uint8) for s in cuda_state])


def _save_fits_dir(images_by_class, dataset, result_dir, epoch_suffix=''):
    fits_dir = os.path.join(result_dir, 'generated_fits')
    os.makedirs(fits_dir, exist_ok=True)
    for class_idx, imgs in images_by_class:
        for i, img in enumerate(imgs):
            norm_array = img.squeeze(0).cpu().numpy()
            jy_array = dataset.denormalise(norm_array)
            fname = os.path.join(fits_dir, f'generated_class{class_idx}_{i:03d}{epoch_suffix}.fits')
            MiraBestFITS.write_fits(jy_array, fname)
    print(f"FITS files saved to {fits_dir}")


def sample_from_model(model, scheduler, class_emb, num_samples, num_classes, device,
                      shape=(1, 150, 150)):
    """Generate random-class images (no CFG). Used for post-training previews only."""
    model.eval()
    labels = torch.randint(0, num_classes, (num_samples,), device=device)
    class_embeddings = class_emb(labels).unsqueeze(1)
    scheduler.set_timesteps(1000)
    images = torch.randn((num_samples, *shape), device=device)
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(images, t, encoder_hidden_states=class_embeddings).sample
            images = scheduler.step(noise_pred, t, images).prev_sample
    return images


def _post_train_save(unet, scheduler, class_emb, config, result_dir, dataset, include_random=True):
    """Generate final images, save PNGs (and FITS if applicable), save final_weights.pt."""
    num_classes = config['data']['num_classes']
    num_samples = config['data']['batch_size']
    device = next(unet.parameters()).device
    guidance_scale = float(config['training'].get('guidance_scale', 7.5))

    gen_0, gen_1 = generate_class_samples(unet, scheduler, class_emb, num_classes, num_samples, device,
                                           guidance_scale=guidance_scale)
    torchvision.utils.save_image(gen_0, f'{result_dir}/generated_images_class_0.png',
                                  nrow=2, normalize=True, value_range=(-1, 1))
    torchvision.utils.save_image(gen_1, f'{result_dir}/generated_images_class_1.png',
                                  nrow=2, normalize=True, value_range=(-1, 1))

    if include_random:
        random_images = sample_from_model(unet, scheduler, class_emb, num_samples, num_classes, device)
        torchvision.utils.save_image(random_images,
                                      f'{result_dir}/generated_images_random_all_classes.png',
                                      nrow=2, normalize=True, value_range=(-1, 1))

    if isinstance(dataset, MiraBestFITS):
        _save_fits_dir([(0, gen_0), (1, gen_1)], dataset, result_dir)

    torch.save(
        {'model_state_dict': unet.state_dict(), 'class_emb_state_dict': class_emb.state_dict(),
         'config': config},
        os.path.join(result_dir, 'final_weights.pt'),
    )
    print("Generated images saved.")


# ---------------------------------------------------------------------------
# Classification trainers
# ---------------------------------------------------------------------------

def train_classification(config, trainloader, valloader, device, result_directory, resume, checkpoint):
    num_classes = config['data']['num_classes']
    model = resnet18(pretrained=True)

    # Freeze layer1 and layer2 — preserve generic low-level features.
    for name, param in model.named_parameters():
        if name.startswith('layer1') or name.startswith('layer2'):
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    num_epochs = config['training']['epochs']
    lr = float(config['training']['learning_rate'])
    wd = float(config['training']['weight_decay'])
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('fc')], 'lr': lr * 0.1},
        {'params': model.fc.parameters(), 'lr': lr},
    ], weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    epoch_losses, val_losses = [], []
    best_val_loss = torch.inf
    best_state = None
    start_epoch = 0

    if resume is not None:
        ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/classification', device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from checkpoint (epoch {start_epoch})")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        model.train()
        print(f"Epoch {epoch}")
        for batch in trainloader:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)
        epoch_losses.append(avg_loss)
        avg_val_loss = evaluate_loss(model, valloader, criterion, device)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f'Epoch {epoch}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f} (best)')
        else:
            print(f'Epoch {epoch}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        if checkpoint is not None:
            save_checkpoint(
                {'epoch': epoch, 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'loss': loss, 'config': config},
                f'{CHECKPOINT_DIR}/classification',
            )

    # Restore best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Restored best model (val loss {best_val_loss:.4f})")

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{result_directory}/classifier_loss_plot.png')
    plt.close()

    return model


def train_robust_classification(config, trainloader, valloader, device, result_directory, resume, checkpoint):
    num_classes = config['data']['num_classes']
    rob_model = TimeDependentResNet(num_classes)
    rob_model.to(device)

    num_epochs = config['training']['epochs']
    warmup_epochs = config['training'].get('warmup_epochs', 20)
    transition_epochs = config['training'].get('transition_epochs', 15)
    label_smoothing = float(config['training'].get('label_smoothing', 0.1))
    num_timesteps = config['training'].get('num_timesteps', 1000)

    optimizer = torch.optim.SGD(
        rob_model.parameters(),
        lr=float(config['training']['learning_rate']),
        momentum=0.9,
        weight_decay=float(config['training']['weight_decay']),
    )
    warmup_lr_epochs = 5

    def lr_lambda(epoch):
        if epoch < warmup_lr_epochs:
            return (epoch + 1) / warmup_lr_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_lr_epochs
    )

    pgd_cfg = config['training'].get('pgd', {})
    pgd_epsilon = float(pgd_cfg.get('epsilon', 0.03))
    pgd_alpha = float(pgd_cfg.get('alpha', 0.01))
    pgd_num_steps = int(pgd_cfg.get('num_steps', 20))
    pgd_random_start = bool(pgd_cfg.get('random_start', True))
    trades_beta = float(config['training'].get('trades_beta', 6.0))

    betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)

    epoch_losses, val_losses = [], []
    val_acc_history, adv_val_acc_history, adv_epochs = [], [], []
    start_epoch = 0
    best_val_acc = 0.0
    loss = None

    if resume is not None:
        ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/robust_classification', device)
        rob_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('warmup_scheduler_state_dict') is not None:
            warmup_scheduler.load_state_dict(ckpt['warmup_scheduler_state_dict'])
        if ckpt.get('cosine_scheduler_state_dict') is not None:
            cosine_scheduler.load_state_dict(ckpt['cosine_scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"Resumed from checkpoint (epoch {start_epoch}, best val acc {best_val_acc:.1f}%)")

    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        rob_model.train()
        max_t = get_max_timestep(epoch, num_epochs, num_timesteps)
        in_warmup = epoch < warmup_epochs
        in_transition = warmup_epochs <= epoch < warmup_epochs + transition_epochs

        if in_transition:
            eps_scale = (epoch - warmup_epochs) / transition_epochs
            current_epsilon = pgd_epsilon * (0.25 + 0.75 * eps_scale)
        else:
            current_epsilon = pgd_epsilon

        for batch_idx, batch in enumerate(trainloader):
            inputs, labels = batch[0].to(device), batch[1].to(device)
            batch_size = inputs.shape[0]
            t = torch.randint(0, max(1, max_t), (batch_size,), device=device)
            x_t = get_noisy_image(inputs, t, alphas_cumprod)

            if in_warmup:
                t_train = t
                optimizer.zero_grad()
                logits = rob_model(x_t, t_train)
                loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
                loss.backward()
                optimizer.step()
            else:
                x_adv = pgd_attack_early_stop(
                    rob_model, x_t, t, labels,
                    epsilon=current_epsilon, alpha=pgd_alpha,
                    num_steps=pgd_num_steps, random_start=pgd_random_start,
                    clamp=(-1.0, 1.0), training_mode=True,
                )
                t_train = t
                optimizer.zero_grad()
                logits_clean = rob_model(x_t, t)
                logits_adv = rob_model(x_adv, t)
                loss_ce = F.cross_entropy(logits_clean, labels, label_smoothing=label_smoothing)
                loss_kl = F.kl_div(
                    F.log_softmax(logits_adv, dim=1),
                    F.softmax(logits_clean.detach(), dim=1),
                    reduction='batchmean',
                )
                loss = loss_ce + trades_beta * loss_kl
                loss.backward()
                optimizer.step()
                logits = logits_adv

            total_loss += loss.item()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                wrong_mask = preds != labels
                wrong_indices = wrong_mask.nonzero(as_tuple=True)[0].tolist()
                wrong_preds = preds[wrong_mask].tolist()
                wrong_labels = labels[wrong_mask].tolist()
                wrong_t = t_train[wrong_mask].tolist()

            t_min, t_max = t_train.min().item(), t_train.max().item()
            status = f"  Batch {batch_idx:>3} | t=[{t_min},{t_max}] | loss={loss.item():.4f} | acc={correct}/{batch_size}"
            if wrong_indices:
                misses = ', '.join(
                    f'[{i}] pred={p} true={l} t={tv}'
                    for i, p, l, tv in zip(wrong_indices, wrong_preds, wrong_labels, wrong_t)
                )
                status += f' | MISCLASSIFIED: {misses}'
            print(status)

        current_lr = optimizer.param_groups[0]['lr']
        if epoch < warmup_lr_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        avg_loss = total_loss / len(trainloader)
        epoch_losses.append(avg_loss)

        rob_model.eval()
        val_loss_accum = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for val_batch in valloader:
                val_inputs, val_labels = val_batch[0].to(device), val_batch[1].to(device)
                t_val = torch.zeros(val_inputs.size(0), dtype=torch.long, device=device)
                val_logits = rob_model(val_inputs, t_val)
                val_loss_accum += F.cross_entropy(val_logits, val_labels,
                                                   label_smoothing=label_smoothing).item()
                val_correct += (val_logits.argmax(dim=1) == val_labels).sum().item()
                val_total += val_labels.size(0)
        avg_val_loss = val_loss_accum / len(valloader)
        val_acc = 100.0 * val_correct / val_total
        val_losses.append(avg_val_loss)

        adv_val_acc = None
        if not in_warmup and epoch % 5 == 0:
            adv_correct, adv_total = 0, 0
            for val_batch in valloader:
                val_inputs, val_labels = val_batch[0].to(device), val_batch[1].to(device)
                t_val = torch.zeros(val_inputs.size(0), dtype=torch.long, device=device)
                val_adv = pgd_attack_early_stop(
                    rob_model, val_inputs, t_val, val_labels,
                    epsilon=pgd_epsilon, alpha=pgd_alpha,
                    num_steps=10, random_start=True, clamp=(-1.0, 1.0),
                )
                with torch.no_grad():
                    adv_logits = rob_model(val_adv, t_val)
                adv_correct += (adv_logits.argmax(dim=1) == val_labels).sum().item()
                adv_total += val_labels.size(0)
            adv_val_acc = 100.0 * adv_correct / adv_total

        val_acc_history.append(val_acc)
        if adv_val_acc is not None:
            adv_val_acc_history.append(adv_val_acc)
            adv_epochs.append(epoch)

        phase = 'warmup' if in_warmup else (f'transition(ε={current_epsilon:.3f})' if in_transition else 'adversarial')
        adv_str = f' | Adv Val Acc: {adv_val_acc:.1f}%' if adv_val_acc is not None else ''
        print(f'Epoch {epoch} [{phase}] | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}'
              f' | Val Acc: {val_acc:.1f}%{adv_str} | LR: {current_lr:.2e}')

        best_metric = adv_val_acc if adv_val_acc is not None else val_acc
        is_new_best = best_metric > best_val_acc
        if is_new_best:
            best_val_acc = best_metric

        ckpt_payload = {
            'epoch': epoch, 'model_state_dict': rob_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
            'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
            'loss': loss, 'config': config, 'best_val_acc': best_val_acc,
        }
        if checkpoint is not None or resume is not None:
            save_checkpoint(ckpt_payload, f'{CHECKPOINT_DIR}/robust_classification')

        if is_new_best:
            save_checkpoint(ckpt_payload, f'{CHECKPOINT_DIR}/robust_classification_best')
            print(f'  ** New best metric: {best_val_acc:.1f}% — saved to checkpoints/robust_classification_best')

    epochs_range = list(range(start_epoch, start_epoch + len(epoch_losses)))

    save_training_plot(epochs_range, epoch_losses, val_losses, result_directory)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, val_acc_history, color='tab:blue', linewidth=2, label='Val Acc (%)')
    if adv_val_acc_history:
        ax.plot(adv_epochs, adv_val_acc_history, color='tab:orange', linewidth=2,
                linestyle='--', label='Adv Val Acc (%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Robust Classifier: Validation Accuracy')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(result_directory, 'robust_classifier_accuracy.png'), dpi=150)
    plt.close()

    with open(os.path.join(result_directory, 'metrics.json'), 'w') as f:
        json.dump({
            'epochs': epochs_range,
            'train_loss': epoch_losses,
            'val_loss': val_losses,
            'val_acc': val_acc_history,
            'adv_epochs': adv_epochs,
            'adv_val_acc': adv_val_acc_history,
            'best_val_acc': best_val_acc,
        }, f, indent=2)

    torch.save(
        {'model_state_dict': rob_model.state_dict(), 'config': config},
        os.path.join(result_directory, 'final_weights.pt'),
    )
    return rob_model


# ---------------------------------------------------------------------------
# Unified diffusion training loop
# ---------------------------------------------------------------------------

def _train_diffusion_loop(
    config, trainloader, valloader, testloader, device, result_dir,
    resume, do_checkpoint, ckpt_dir,
    loss_fn,
    dataset=None,
    extra_keys=None,
    compliance_fn=None,
    init_fn=None,
):
    """Core training loop shared by all four diffusion model variants.

    Args:
        loss_fn: Callable(**kwargs) -> (loss: Tensor, extras: dict[str, float])
            Receives noise_pred, noise, noisy_images, images, labels,
            training_labels, t, alphas_cumprod, device. Use **_ to ignore unused args.
        extra_keys: list[str] — keys expected in loss_fn's extras dict;
            each accumulates into a per-epoch history list.
        compliance_fn: (gen_0, gen_1) -> dict[str, float] — optional physics
            compliance metrics computed every 5 epochs on generated images (PID).
        init_fn: (unet, class_emb, device) -> None — optional hook called once
            before training starts when not resuming (e.g. load pretrained weights).

    Returns:
        (unet, scheduler, class_emb, histories: dict)
    """
    torch.cuda.empty_cache()
    gc.collect()

    unet, scheduler, class_emb, optimizer = build_diffusion_components(config, {}, device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    num_classes = config['data']['num_classes']
    num_epochs = config['training']['epochs']
    label_dropout = config['training']['label_dropout']
    guidance_scale = float(config['training'].get('guidance_scale', 7.5))
    eval_interval = int(config['training'].get('eval_interval', 5))
    eval_num_samples = int(config['training'].get('eval_num_samples', 16))
    extra_keys = extra_keys or []

    loss_history, val_loss_history, epochs_range = [], [], []
    fid_history, kid_history, fid_epochs, pdf_history = [], [], [], []
    extra_hist = {k: [] for k in extra_keys}
    compliance_epochs, compliance_hist = [], {}
    start_epoch = 0
    loss = None

    if resume is not None:
        ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/{ckpt_dir}', device)
        _restore_rng(ckpt)
        unet.load_state_dict(ckpt['model_state_dict'])
        class_emb.load_state_dict(ckpt['class_emb_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        loss_history = ckpt['loss_history']
        val_loss_history = ckpt['val_loss_history']
        epochs_range = ckpt['epochs_range']
        fid_history = ckpt['fid_history']
        kid_history = ckpt.get('kid_history', [])
        fid_epochs = ckpt.get('fid_epochs', [])
        pdf_history = ckpt.get('pdf_history', [])
        for k in extra_keys:
            extra_hist[k] = ckpt.get(f'{k}_history', [])
        if compliance_fn is not None:
            compliance_epochs = ckpt.get('compliance_epochs', [])
            for k in list(ckpt.keys()):
                if k.endswith('_history') and k[:-8] not in extra_keys and k not in (
                    'loss_history', 'val_loss_history', 'fid_history', 'kid_history', 'pdf_history',
                ):
                    compliance_hist[k[:-8]] = ckpt[k]
        print(f"Resumed from checkpoint: {ckpt_dir} (epoch {start_epoch})")
    elif init_fn is not None:
        init_fn(unet, class_emb, device)

    for epoch in range(start_epoch, num_epochs):
        unet.train()
        epoch_loss = 0.0
        epoch_extra = {k: 0.0 for k in extra_keys}
        batch_count = 0

        print(f'Epoch {epoch}')
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            drop_mask = torch.rand(labels.shape, device=device) < label_dropout
            training_labels = labels.clone()
            training_labels[drop_mask] = num_classes

            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, noise, t)
            class_embeddings = class_emb(training_labels).unsqueeze(1)
            noise_pred = unet(noisy_images, t, encoder_hidden_states=class_embeddings).sample

            loss, extras = loss_fn(
                noise_pred=noise_pred, noise=noise, noisy_images=noisy_images,
                images=images, labels=labels, training_labels=training_labels,
                t=t, alphas_cumprod=alphas_cumprod, device=device,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            for k, v in extras.items():
                epoch_extra[k] += v
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        epochs_range.append(epoch)
        for k in extra_keys:
            extra_hist[k].append(epoch_extra[k] / batch_count)

        avg_val_loss = eval_epoch(unet, scheduler, class_emb, testloader, num_classes, device)
        val_loss_history.append(avg_val_loss)

        extra_str = '  '.join(f'{k}: {extra_hist[k][-1]:.6f}' for k in extra_keys)
        log = f'Epoch {epoch} | Loss: {avg_loss:.6f} | Val: {avg_val_loss:.6f}'
        if extra_str:
            log += f'  |  {extra_str}'
        print(log)

        if epoch % eval_interval == 0:
            unet.eval()
            with torch.no_grad():
                gen_0, gen_1 = generate_class_samples(
                    unet, scheduler, class_emb, num_classes, eval_num_samples, device,
                    guidance_scale=guidance_scale,
                )

            save_comparison_grid(gen_0[:4], gen_1[:4], epoch, result_dir)

            if isinstance(dataset, MiraBestFITS):
                _save_fits_dir([(0, gen_0), (1, gen_1)], dataset, result_dir, f'_{epoch}')

            print(f'Computing FID/KID/PDF at epoch {epoch}...')
            fid_score, kid_score = compute_fid_kid(gen_0, gen_1, valloader, device)
            fid_history.append(fid_score)
            kid_history.append(kid_score)
            fid_epochs.append(epoch)
            print(f'  FID: {fid_score:.4f} | KID: {kid_score:.6f}')

            pdf_score = compute_pixel_pdf(gen_0, gen_1, valloader, num_classes, result_dir, epoch)
            pdf_history.append(pdf_score)
            print(f'  Pixel PDF W-dist: {pdf_score:.4f}')

            if compliance_fn is not None:
                c_metrics = compliance_fn(gen_0, gen_1)
                compliance_epochs.append(epoch)
                for k, v in c_metrics.items():
                    compliance_hist.setdefault(k, []).append(v)
                print('  Compliance — ' + '  '.join(f'{k}: {v:.4f}' for k, v in c_metrics.items()))

            unet.train()

        if do_checkpoint is not None or resume is not None:
            payload = {
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_emb_state_dict': class_emb.state_dict(),
                'loss': loss,
                'config': config,
                'loss_history': loss_history,
                'val_loss_history': val_loss_history,
                'epochs_range': epochs_range,
                'fid_history': fid_history,
                'kid_history': kid_history,
                'fid_epochs': fid_epochs,
                'pdf_history': pdf_history,
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            }
            for k in extra_keys:
                payload[f'{k}_history'] = extra_hist[k]
            if compliance_fn is not None:
                payload['compliance_epochs'] = compliance_epochs
                for k, v in compliance_hist.items():
                    payload[f'{k}_history'] = v
            save_checkpoint(payload, f'{CHECKPOINT_DIR}/{ckpt_dir}')

    return unet, scheduler, class_emb, {
        'loss_history': loss_history,
        'val_loss_history': val_loss_history,
        'epochs_range': epochs_range,
        'fid_history': fid_history,
        'kid_history': kid_history,
        'fid_epochs': fid_epochs,
        'pdf_history': pdf_history,
        'extra': extra_hist,
        'compliance_epochs': compliance_epochs,
        'compliance': compliance_hist,
    }


# ---------------------------------------------------------------------------
# Public training wrappers
# ---------------------------------------------------------------------------

def train_diffusion(config, trainloader, valloader, testloader, device, result_directory,
                    resume, checkpoint, dataset=None):

    def loss_fn(noise_pred, noise, **_):
        return F.mse_loss(noise_pred, noise), {}

    unet, scheduler, class_emb, hist = _train_diffusion_loop(
        config, trainloader, valloader, testloader, device, result_directory,
        resume, checkpoint, 'diffusion', loss_fn, dataset=dataset,
    )

    _post_train_save(unet, scheduler, class_emb, config, result_directory, dataset, include_random=True)
    save_training_plot(hist['epochs_range'], hist['loss_history'], hist['val_loss_history'],
                       result_directory)
    save_generative_metrics_plot(hist['fid_epochs'], hist['fid_history'], hist['kid_history'],
                                  result_directory)
    save_pixel_pdf_history_plot(hist['fid_epochs'], hist['pdf_history'], result_directory)
    return unet


def train_pid(config, trainloader, valloader, testloader, device, result_directory,
              resume, checkpoint, dataset=None):
    """Physics-informed diffusion: DDPM MSE + symmetry + non-negativity penalties."""
    lambda_sym = float(config['training'].get('lambda_sym', 0.1))
    lambda_neg = float(config['training'].get('lambda_neg', 0.1))

    def loss_fn(noise_pred, noise, noisy_images, t, alphas_cumprod, **_):
        mse = F.mse_loss(noise_pred, noise)
        x0 = estimate_x0(noisy_images, noise_pred, alphas_cumprod, t)
        sym = lambda_sym * symmetry_loss(x0)
        neg = lambda_neg * nonnegativity_loss(x0)
        return mse + sym + neg, {'mse': mse.item(), 'sym': sym.item(), 'neg': neg.item()}

    def compliance_fn(gen_0, gen_1):
        all_gen = torch.cat([gen_0, gen_1], dim=0)
        return {
            'pct_negative': (all_gen < 0).float().mean().item() * 100,
            'sym_score': symmetry_loss(all_gen).item(),
        }

    unet, scheduler, class_emb, hist = _train_diffusion_loop(
        config, trainloader, valloader, testloader, device, result_directory,
        resume, checkpoint, 'pid', loss_fn, dataset=dataset,
        extra_keys=['mse', 'sym', 'neg'], compliance_fn=compliance_fn,
    )

    _post_train_save(unet, scheduler, class_emb, config, result_directory, dataset,
                     include_random=False)

    with open(os.path.join(result_directory, 'metrics.json'), 'w') as f:
        json.dump({
            'epochs': hist['epochs_range'],
            'loss': hist['loss_history'],
            'val_loss': hist['val_loss_history'],
            'mse': hist['extra'].get('mse', []),
            'sym': hist['extra'].get('sym', []),
            'neg': hist['extra'].get('neg', []),
            'compliance_epochs': hist['compliance_epochs'],
            'pct_negative': hist['compliance'].get('pct_negative', []),
            'sym_score': hist['compliance'].get('sym_score', []),
            'fid_epochs': hist['fid_epochs'],
            'fid': hist['fid_history'],
            'kid': hist['kid_history'],
        }, f, indent=2)

    save_pid_training_plots(
        hist['epochs_range'], hist['loss_history'], hist['val_loss_history'],
        hist['extra'].get('mse', []), hist['extra'].get('sym', []),
        hist['extra'].get('neg', []),
        hist['compliance_epochs'],
        hist['compliance'].get('pct_negative', []),
        hist['compliance'].get('sym_score', []),
        result_directory,
    )
    save_generative_metrics_plot(hist['fid_epochs'], hist['fid_history'], hist['kid_history'],
                                  result_directory)
    save_pixel_pdf_history_plot(hist['fid_epochs'], hist['pdf_history'], result_directory)
    return unet


def train_classifier_guided_diffusion(config, trainloader, valloader, testloader, device,
                                       result_directory, resume, checkpoint, dataset=None):
    """Fine-tunes a pre-trained diffusion model with frozen classifier guidance.

    loss = MSE + lambda_cls * (1 - p_correct).mean()
    """
    num_classes = config['data']['num_classes']
    lambda_cls = float(config['training'].get('lambda_cls', 0.1))

    classifier = TimeDependentResNet(num_classes, pretrained=False)
    cls_ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/robust_classification', device)
    classifier.load_state_dict(cls_ckpt['model_state_dict'])
    classifier.to(device).eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    def loss_fn(noise_pred, noise, noisy_images, labels, training_labels, t, alphas_cumprod, device, **_):
        mse = F.mse_loss(noise_pred, noise)
        cls_loss = torch.tensor(0.0, device=device)
        cls_mask = training_labels != num_classes
        if cls_mask.any():
            # generate image
            x0 = estimate_x0(noisy_images, noise_pred, alphas_cumprod, t)
            cls_input = x0[cls_mask]

            # convert to png if fits
            if isinstance(dataset, MiraBestFITS):
                cls_input = fits_to_linear(cls_input, dataset)

            # classify
            t_clean = torch.zeros(cls_mask.sum(), dtype=torch.long, device=device)
            p_correct = F.softmax(classifier(cls_input, t_clean), dim=1).gather(
                1, labels[cls_mask].unsqueeze(1)).squeeze(1)
            cls_loss = lambda_cls * (1.0 - p_correct).mean()
        return mse + cls_loss, {'cls_loss': cls_loss.item()}

    def init_fn(unet, class_emb, device):
        pretrained_dir = config['model'].get('pretrained_checkpoint', f'{CHECKPOINT_DIR}/diffusion')
        ckpt_path = os.path.join(pretrained_dir, 'state.pt')
        if os.path.exists(ckpt_path):
            ckpt = load_checkpoint(pretrained_dir, device)
            unet.load_state_dict(ckpt['model_state_dict'])
            class_emb.load_state_dict(ckpt['class_emb_state_dict'])
            print(f"Loaded pre-trained diffusion weights from {pretrained_dir}")
        else:
            print(f"No pre-trained checkpoint at {pretrained_dir}, training from scratch")

    unet, scheduler, class_emb, hist = _train_diffusion_loop(
        config, trainloader, valloader, testloader, device, result_directory,
        resume, checkpoint, 'classifier_guided_diffusion', loss_fn, dataset=dataset,
        extra_keys=['cls_loss'], init_fn=init_fn,
    )

    _post_train_save(unet, scheduler, class_emb, config, result_directory, dataset, include_random=True)
    save_training_plot(hist['epochs_range'], hist['loss_history'], hist['val_loss_history'],
                       result_directory)
    save_generative_metrics_plot(hist['fid_epochs'], hist['fid_history'], hist['kid_history'],
                                  result_directory)
    save_pixel_pdf_history_plot(hist['fid_epochs'], hist['pdf_history'], result_directory)
    return unet


def train_robust_classifier_guided_diffusion(config, trainloader, valloader, testloader, device,
                                              result_directory, resume, checkpoint, dataset=None):
    """Fine-tunes a pre-trained diffusion model with robust classifier guidance.

    Identical to train_classifier_guided_diffusion but uses cross-entropy as the
    guidance loss (more numerically stable when classifier confidence is poorly calibrated).
    loss = MSE + lambda_cls * CE(classifier(x_0_pred), labels)
    """
    num_classes = config['data']['num_classes']
    lambda_cls = float(config['training'].get('lambda_cls', 0.1))

    classifier = TimeDependentResNet(num_classes, pretrained=False)
    cls_ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/robust_classification', device)
    classifier.load_state_dict(cls_ckpt['model_state_dict'])
    classifier.to(device).eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    def loss_fn(noise_pred, noise, noisy_images, labels, training_labels, t, alphas_cumprod, device, **_):
        mse = F.mse_loss(noise_pred, noise)
        cls_loss = torch.tensor(0.0, device=device)
        cls_mask = training_labels != num_classes
        if cls_mask.any():
            x0 = estimate_x0(noisy_images, noise_pred, alphas_cumprod, t)
            cls_input = x0[cls_mask]
            if isinstance(dataset, MiraBestFITS):
                cls_input = fits_to_linear(cls_input, dataset)
            t_clean = torch.zeros(cls_mask.sum(), dtype=torch.long, device=device)
            cls_loss = lambda_cls * F.cross_entropy(classifier(cls_input, t_clean), labels[cls_mask])
        return mse + cls_loss, {'cls_loss': cls_loss.item()}

    def init_fn(unet, class_emb, device):
        pretrained_dir = config['model'].get('pretrained_checkpoint', f'{CHECKPOINT_DIR}/diffusion')
        ckpt_path = os.path.join(pretrained_dir, 'state.pt')
        if os.path.exists(ckpt_path):
            ckpt = load_checkpoint(pretrained_dir, device)
            unet.load_state_dict(ckpt['model_state_dict'])
            class_emb.load_state_dict(ckpt['class_emb_state_dict'])
            print(f"Loaded pre-trained diffusion weights from {pretrained_dir}")
        else:
            print(f"No pre-trained checkpoint at {pretrained_dir}, training from scratch")

    unet, scheduler, class_emb, hist = _train_diffusion_loop(
        config, trainloader, valloader, testloader, device, result_directory,
        resume, checkpoint, 'robust_classifier_guided_diffusion', loss_fn, dataset=dataset,
        extra_keys=['cls_loss'], init_fn=init_fn,
    )

    _post_train_save(unet, scheduler, class_emb, config, result_directory, dataset, include_random=True)
    save_training_plot(hist['epochs_range'], hist['loss_history'], hist['val_loss_history'],
                       result_directory)
    save_generative_metrics_plot(hist['fid_epochs'], hist['fid_history'], hist['kid_history'],
                                  result_directory)
    save_pixel_pdf_history_plot(hist['fid_epochs'], hist['pdf_history'], result_directory)
    return unet


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def train_model(model, config, trainloader, valloader, testloader, device, result_directory,
                resume, checkpoint, dataset=None):
    print(f"Training {model} for {config['training']['epochs']} epochs")
    if model == 'classification':
        return train_classification(config, trainloader, valloader, device,
                                    result_directory, resume, checkpoint)
    elif model == 'robust_classification':
        return train_robust_classification(config, trainloader, valloader, device,
                                           result_directory, resume, checkpoint)
    elif model == 'diffusion':
        return train_diffusion(config, trainloader, valloader, testloader, device,
                               result_directory, resume, checkpoint, dataset=dataset)
    elif model == 'pid':
        return train_pid(config, trainloader, valloader, testloader, device,
                         result_directory, resume, checkpoint, dataset=dataset)
    elif model == 'classifier_guided_diffusion':
        return train_classifier_guided_diffusion(config, trainloader, valloader, testloader, device,
                                                  result_directory, resume, checkpoint, dataset=dataset)
    elif model == 'robust_classifier_guided_diffusion':
        return train_robust_classifier_guided_diffusion(config, trainloader, valloader, testloader,
                                                         device, result_directory, resume, checkpoint,
                                                         dataset=dataset)
    else:
        raise ValueError(
            f'Model {model} not supported. '
            f'Choose from: classification, robust_classification, diffusion, pid, '
            f'classifier_guided_diffusion, robust_classifier_guided_diffusion'
        )
