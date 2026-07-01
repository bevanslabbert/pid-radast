import json
import math
import os

import pyhopper
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
import yaml
from diffusers import UNet2DConditionModel, DDPMScheduler

from src.datasets.mirabest.MiraBestFITS import MiraBestFITS
from src.models.simple_cnn import SimpleCNN
from src.models.time_dependent_resnet import TimeDependentResNet
from src.models.pid import estimate_x0, physics_loss
from src.utils.augmentation import pgd_attack_early_stop, get_noisy_image, get_max_timestep
from src.utils.checkpoint import load_checkpoint

# Short training proxy used during each trial — not a full training run.
TRIAL_EPOCHS = 5
CHECKPOINT_DIR = 'checkpoints'


def _build_search_space(opt_config: dict) -> dict:
    """Convert the YAML optimization.parameters spec into a PyHopper search space."""
    space = {}
    for name, spec in opt_config.get('parameters', {}).items():
        if 'choices' in spec:
            space[name] = pyhopper.choice(spec['choices'])
        elif 'min' in spec and 'max' in spec:
            log = spec.get('log', False)
            if isinstance(spec['min'], int) and isinstance(spec['max'], int) and not log:
                space[name] = pyhopper.int(spec['min'], spec['max'])
            else:
                space[name] = pyhopper.float(spec['min'], spec['max'], log=log)
    return space


def _write_yaml_params(params: dict, out_path: str) -> None:
    """Write best params to YAML, expanding dotted keys to nested dicts.

    Keys like 'pgd.epsilon' become pgd: {epsilon: <value>} so the output
    can be copy-pasted directly into the training: section of a config file.
    """
    nested = {}
    for k, v in params.items():
        if '.' in k:
            parent, child = k.split('.', 1)
            nested.setdefault(parent, {})[child] = v
        else:
            nested[k] = v
    with open(out_path, 'w') as f:
        yaml.dump({'training': nested}, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Per-model objective functions — each returns a scalar PyHopper maximises.
# ---------------------------------------------------------------------------

def _objective_classification(params, cfg, trainloader, valloader, device, dataset=None):
    num_classes = cfg['data']['num_classes']

    batch_size = int(params.get('batch_size', cfg['data']['batch_size']))
    if batch_size != trainloader.batch_size:
        trial_loader = tud.DataLoader(
            trainloader.dataset, batch_size=batch_size, shuffle=True, num_workers=0,
        )
    else:
        trial_loader = trainloader

    model = SimpleCNN(num_classes=num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params.get('learning_rate', cfg['training']['learning_rate'])),
        weight_decay=float(params.get('weight_decay', cfg['training']['weight_decay'])),
    )

    for _ in range(TRIAL_EPOCHS):
        model.train()
        for inputs, labels in trial_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            F.cross_entropy(model(inputs), labels).backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            correct += (model(inputs).argmax(1) == labels).sum().item()
            total += labels.size(0)

    return correct / total  # maximise val accuracy


def _objective_robust_classification(params, cfg, trainloader, valloader, device, dataset=None):
    num_classes = cfg['data']['num_classes']
    num_timesteps = 1000
    model = TimeDependentResNet(num_classes)
    model.to(device)

    momentum = float(params.get('momentum', cfg['training'].get('momentum', 0.9)))
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(params.get('learning_rate', cfg['training']['learning_rate'])),
        momentum=momentum,
        weight_decay=float(params.get('weight_decay', cfg['training']['weight_decay'])),
    )

    betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)

    pgd_cfg = cfg['training'].get('pgd', {})
    epsilon   = float(params.get('pgd.epsilon', pgd_cfg.get('epsilon', 0.03)))
    alpha     = float(params.get('pgd.alpha',   pgd_cfg.get('alpha', 0.01)))
    num_steps = int(params.get('pgd.num_steps', pgd_cfg.get('num_steps', 20)))

    for epoch in range(TRIAL_EPOCHS):
        model.train()
        max_t = get_max_timestep(epoch, TRIAL_EPOCHS, num_timesteps)
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            t = torch.randint(0, max(1, max_t), (inputs.shape[0],), device=device)
            x_t = get_noisy_image(inputs, t, alphas_cumprod)
            x_adv = pgd_attack_early_stop(model, x_t, t, labels,
                                          epsilon=epsilon, alpha=alpha, num_steps=num_steps,
                                          clamp=(-1.0, 1.0))
            optimizer.zero_grad()
            F.cross_entropy(model(x_adv, t), labels).backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            t = torch.zeros(inputs.shape[0], dtype=torch.long, device=device)
            correct += (model(inputs, t).argmax(1) == labels).sum().item()
            total += labels.size(0)

    return correct / total  # maximise clean val accuracy


def _objective_diffusion(params, cfg, trainloader, valloader, device, dataset=None):
    num_classes = cfg['data']['num_classes']
    label_dropout = params.get('label_dropout', cfg['training']['label_dropout'])
    embedding_dim = int(params.get('embedding_dim', cfg['model']['embedding_dim']))
    weight_decay = float(params.get('weight_decay', cfg['training']['weight_decay']))

    unet = UNet2DConditionModel(
        sample_size=cfg['data']['input_size'],
        in_channels=1, out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D",
            "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
            "UpBlock2D", "UpBlock2D",
        ),
        cross_attention_dim=embedding_dim,
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=cfg['training']['num_train_timesteps'])
    class_emb = nn.Embedding(num_classes + 1, embedding_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(class_emb.parameters()),
        lr=float(params.get('learning_rate', cfg['training']['learning_rate'])),
        weight_decay=weight_decay,
    )

    for _ in range(TRIAL_EPOCHS):
        unet.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            drop_mask = torch.rand(labels.shape, device=device) < label_dropout
            training_labels = labels.clone()
            training_labels[drop_mask] = num_classes

            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)
            pred = unet(noisy, t,
                        encoder_hidden_states=class_emb(training_labels).unsqueeze(1)).sample

            optimizer.zero_grad()
            F.mse_loss(pred, noise).backward()
            optimizer.step()

    unet.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)
            pred = unet(noisy, t,
                        encoder_hidden_states=class_emb(labels).unsqueeze(1)).sample
            val_loss += F.mse_loss(pred, noise).item()

    # Negate because PyHopper maximises — lower val loss is better.
    return -(val_loss / len(valloader))


def _objective_pid(params, cfg, trainloader, valloader, device, dataset=None):
    num_classes = cfg['data']['num_classes']
    label_dropout = params.get('label_dropout', cfg['training']['label_dropout'])
    embedding_dim = int(params.get('embedding_dim', cfg['model']['embedding_dim']))
    lambda_sym = float(params.get('lambda_sym', cfg['training'].get('lambda_sym', 0.1)))
    lambda_neg = float(params.get('lambda_neg', cfg['training'].get('lambda_neg', 0.5)))

    unet = UNet2DConditionModel(
        sample_size=cfg['data']['input_size'],
        in_channels=1, out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D",
            "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
            "UpBlock2D", "UpBlock2D",
        ),
        cross_attention_dim=embedding_dim,
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=cfg['training']['num_train_timesteps'])
    alphas_cumprod = scheduler.alphas_cumprod.to(device)
    class_emb = nn.Embedding(num_classes + 1, embedding_dim).to(device)

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(class_emb.parameters()),
        lr=float(params.get('learning_rate', cfg['training']['learning_rate'])),
        weight_decay=float(params.get('weight_decay', cfg['training'].get('weight_decay', 0.01))),
    )

    for _ in range(TRIAL_EPOCHS):
        unet.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            drop_mask = torch.rand(labels.shape, device=device) < label_dropout
            training_labels = labels.clone()
            training_labels[drop_mask] = num_classes

            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)

            class_embeddings = class_emb(training_labels).unsqueeze(1)
            noise_pred = unet(noisy, t, encoder_hidden_states=class_embeddings).sample

            mse = F.mse_loss(noise_pred, noise)
            x_0_pred = estimate_x0(noisy, noise_pred, alphas_cumprod, t)
            loss = mse + physics_loss(x_0_pred, lambda_sym, lambda_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    unet.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            batch_sz = images.size(0)

            t = torch.linspace(0, scheduler.num_train_timesteps - 1, batch_sz, dtype=torch.long, device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)

            class_embeddings = class_emb(labels).unsqueeze(1)
            noise_pred = unet(noisy, t, encoder_hidden_states=class_embeddings).sample

            x_0_val = estimate_x0(noisy, noise_pred, alphas_cumprod, t)
            val_loss += (F.mse_loss(noise_pred, noise) + physics_loss(x_0_val, lambda_sym, lambda_neg)).item()

    return -(val_loss / len(valloader))


def _build_guided_diffusion_components(cfg, params, device):
    """Build UNet + scheduler + class_emb + optimizer for guided diffusion objectives."""
    num_classes = cfg['data']['num_classes']
    embedding_dim = int(params.get('embedding_dim', cfg['model'].get('embedding_dim', 256)))

    unet = UNet2DConditionModel(
        sample_size=cfg['data']['input_size'],
        in_channels=1, out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D",
            "CrossAttnDownBlock2D", "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",
            "UpBlock2D", "UpBlock2D",
        ),
        cross_attention_dim=embedding_dim,
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=cfg['training']['num_train_timesteps'])
    class_emb = nn.Embedding(num_classes + 1, embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(class_emb.parameters()),
        lr=float(params.get('learning_rate', cfg['training']['learning_rate'])),
        weight_decay=float(params.get('weight_decay', cfg['training'].get('weight_decay', 0.01))),
    )
    return unet, scheduler, class_emb, optimizer


def _load_pretrained_diffusion(unet, class_emb, cfg, device):
    """Load pre-trained diffusion weights if checkpoint exists."""
    pretrained_dir = cfg['model'].get('pretrained_checkpoint', f'{CHECKPOINT_DIR}/diffusion')
    ckpt_path = os.path.join(pretrained_dir, 'state.pt')
    if os.path.exists(ckpt_path):
        ckpt = load_checkpoint(pretrained_dir, device)
        unet.load_state_dict(ckpt['model_state_dict'])
        class_emb.load_state_dict(ckpt['class_emb_state_dict'])
        print(f"[optimize] Loaded pre-trained diffusion from {pretrained_dir}")
    else:
        print(f"[optimize] No checkpoint at {pretrained_dir} — training guidance from scratch")


def _load_frozen_classifier(cfg, device):
    """Load and freeze the robust classifier used for guidance."""
    num_classes = cfg['data']['num_classes']
    ckpt_path = os.path.join(CHECKPOINT_DIR, 'robust_classification', 'state.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Robust classifier checkpoint not found at {ckpt_path}. "
            "Train robust_classification first before optimising guided diffusion."
        )
    classifier = TimeDependentResNet(num_classes, pretrained=False)
    ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/robust_classification', device)
    classifier.load_state_dict(ckpt['model_state_dict'])
    classifier.to(device).eval()
    for p in classifier.parameters():
        p.requires_grad_(False)
    return classifier


def _objective_classifier_guided_diffusion(params, cfg, trainloader, valloader, device, dataset=None):
    num_classes = cfg['data']['num_classes']
    label_dropout = float(params.get('label_dropout', cfg['training']['label_dropout']))
    lambda_cls    = float(params.get('lambda_cls',    cfg['training'].get('lambda_cls', 0.1)))

    unet, scheduler, class_emb, optimizer = _build_guided_diffusion_components(cfg, params, device)
    _load_pretrained_diffusion(unet, class_emb, cfg, device)
    classifier = _load_frozen_classifier(cfg, device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    for _ in range(TRIAL_EPOCHS):
        unet.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            drop_mask = torch.rand(labels.shape, device=device) < label_dropout
            training_labels = labels.clone()
            training_labels[drop_mask] = num_classes

            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)
            class_embeddings = class_emb(training_labels).unsqueeze(1)
            noise_pred = unet(noisy, t, encoder_hidden_states=class_embeddings).sample

            mse = F.mse_loss(noise_pred, noise)
            cls_loss = torch.tensor(0.0, device=device)
            cls_mask = training_labels != num_classes
            if cls_mask.any():
                x0 = estimate_x0(noisy, noise_pred, alphas_cumprod, t)
                cls_input = x0[cls_mask]
                if isinstance(dataset, MiraBestFITS):
                    peak_log = dataset.median_peak_log
                    cls_input = (torch.sign(cls_input)
                                 * torch.expm1(torch.abs(cls_input) * peak_log)
                                 / math.expm1(peak_log))
                t_clean = torch.zeros(cls_mask.sum(), dtype=torch.long, device=device)
                p_correct = F.softmax(classifier(cls_input, t_clean), dim=1).gather(
                    1, labels[cls_mask].unsqueeze(1)).squeeze(1)
                cls_loss = lambda_cls * (1.0 - p_correct).mean()

            optimizer.zero_grad()
            (mse + cls_loss).backward()
            optimizer.step()

    unet.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)
            noise_pred = unet(noisy, t,
                              encoder_hidden_states=class_emb(labels).unsqueeze(1)).sample
            val_loss += F.mse_loss(noise_pred, noise).item()

    return -(val_loss / len(valloader))


def _objective_robust_classifier_guided_diffusion(params, cfg, trainloader, valloader, device, dataset=None):
    num_classes = cfg['data']['num_classes']
    label_dropout = float(params.get('label_dropout', cfg['training']['label_dropout']))
    lambda_cls    = float(params.get('lambda_cls',    cfg['training'].get('lambda_cls', 0.1)))

    unet, scheduler, class_emb, optimizer = _build_guided_diffusion_components(cfg, params, device)
    _load_pretrained_diffusion(unet, class_emb, cfg, device)
    classifier = _load_frozen_classifier(cfg, device)
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    for _ in range(TRIAL_EPOCHS):
        unet.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            drop_mask = torch.rand(labels.shape, device=device) < label_dropout
            training_labels = labels.clone()
            training_labels[drop_mask] = num_classes

            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)
            class_embeddings = class_emb(training_labels).unsqueeze(1)
            noise_pred = unet(noisy, t, encoder_hidden_states=class_embeddings).sample

            mse = F.mse_loss(noise_pred, noise)
            cls_loss = torch.tensor(0.0, device=device)
            cls_mask = training_labels != num_classes
            if cls_mask.any():
                x0 = estimate_x0(noisy, noise_pred, alphas_cumprod, t)
                cls_input = x0[cls_mask]
                if isinstance(dataset, MiraBestFITS):
                    peak_log = dataset.median_peak_log
                    cls_input = (torch.sign(cls_input)
                                 * torch.expm1(torch.abs(cls_input) * peak_log)
                                 / math.expm1(peak_log))
                t_clean = torch.zeros(cls_mask.sum(), dtype=torch.long, device=device)
                cls_loss = lambda_cls * F.cross_entropy(
                    classifier(cls_input, t_clean), labels[cls_mask]
                )

            optimizer.zero_grad()
            (mse + cls_loss).backward()
            optimizer.step()

    unet.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy = scheduler.add_noise(images, noise, t)
            noise_pred = unet(noisy, t,
                              encoder_hidden_states=class_emb(labels).unsqueeze(1)).sample
            val_loss += F.mse_loss(noise_pred, noise).item()

    return -(val_loss / len(valloader))


# ---------------------------------------------------------------------------
# Registry — register a new model type by adding one entry here.
# ---------------------------------------------------------------------------

_OBJECTIVES = {
    'classification':                     _objective_classification,
    'robust_classification':              _objective_robust_classification,
    'diffusion':                          _objective_diffusion,
    'pid':                                _objective_pid,
    'classifier_guided_diffusion':        _objective_classifier_guided_diffusion,
    'robust_classifier_guided_diffusion': _objective_robust_classifier_guided_diffusion,
}


def optimize_parameters(model_type, cfg, trainloader, valloader, device, result_directory,
                        dataset=None):
    if model_type not in _OBJECTIVES:
        raise ValueError(
            f"No optimization objective registered for '{model_type}'. "
            f"Available: {list(_OBJECTIVES)}"
        )

    opt_cfg = cfg.get('optimization', {})
    max_steps = opt_cfg.get('max_trials', 25)
    space = _build_search_space(opt_cfg)

    if not space:
        raise ValueError(
            "No parameters defined under optimization.parameters in config. "
            "Add at least one parameter with min/max or choices."
        )

    objective_fn = _OBJECTIVES[model_type]
    objective = lambda params: objective_fn(params, cfg, trainloader, valloader, device,
                                            dataset=dataset)

    print(f"Starting {model_type} optimization: {max_steps} trials over {list(space)}")
    search = pyhopper.Search(space)
    best = search.run(objective, direction="max", steps=max_steps)

    out_json = os.path.join(result_directory, 'best_params.json')
    with open(out_json, 'w') as f:
        json.dump(dict(best), f, indent=2)

    out_yaml = os.path.join(result_directory, 'best_params.yaml')
    _write_yaml_params(dict(best), out_yaml)

    print(f"Best params for {model_type}: {best}")
    print(f"Saved to {out_json} and {out_yaml}")

    return best
