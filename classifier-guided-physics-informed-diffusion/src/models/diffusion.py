import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusers import UNet2DConditionModel, DDPMScheduler


def build_diffusion_components(cfg: dict, params: dict, device):
    """
    Build UNet, DDPMScheduler, class embedding, and AdamW optimizer.

    ``params`` values override config defaults, enabling hyperparameter search
    to inject trial values without touching the config.
    """
    num_classes = cfg['data']['num_classes']
    training_cfg = cfg['training']

    cross_attention_dim = params.get('cross_attention_dim', 256)
    layers_per_block = params.get('layers_per_block', 2)

    unet = UNet2DConditionModel(
        sample_size=cfg['data']['input_size'],
        in_channels=1,
        out_channels=1,
        layers_per_block=layers_per_block,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        cross_attention_dim=cross_attention_dim,
    ).to(device)

    beta_schedule = params.get('beta_schedule', 'linear')
    beta_start = params.get('beta_start', 0.0001)
    beta_end = params.get('beta_end', 0.02)
    num_train_timesteps = params.get('num_train_timesteps', 1000)

    scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        beta_start=beta_start,
        beta_end=beta_end,
    )

    # Index `num_classes` is the null/unconditional embedding for CFG.
    class_emb = nn.Embedding(num_classes + 1, cross_attention_dim).to(device)

    lr = float(params.get('learning_rate', training_cfg['learning_rate']))
    weight_decay = float(params.get('weight_decay', training_cfg.get('weight_decay', 0.01)))

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(class_emb.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    return unet, scheduler, class_emb, optimizer


def train_epoch(unet, scheduler, class_emb, optimizer, loader, num_classes: int, label_dropout: float, device) -> float:
    """Run one training epoch. Returns average MSE loss over the epoch."""
    unet.train()
    total_loss = 0.0
    count = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        drop_mask = torch.rand(labels.shape, device=device) < label_dropout
        training_labels = labels.clone()
        training_labels[drop_mask] = num_classes

        t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
        noise = torch.randn_like(images)
        noisy = scheduler.add_noise(images, noise, t)

        pred = unet(noisy, t, encoder_hidden_states=class_emb(training_labels).unsqueeze(1)).sample
        loss = F.mse_loss(pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / count


def eval_epoch(unet, scheduler, class_emb, loader, num_classes: int, device) -> float:
    """Evaluate diffusion val loss over a dataloader. Returns average MSE loss."""
    unet.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)
            pred = unet(noisy, t, encoder_hidden_states=class_emb(labels).unsqueeze(1)).sample
            total_loss += F.mse_loss(pred, noise).item()

    return total_loss / len(loader)
