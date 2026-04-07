import json
import os

import pyhopper
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
from torchvision.models import resnet50

from src.models.time_dependent_resnet import TimeDependentResNet
from src.utils.augmentation import pgd_attack_early_stop, get_noisy_image, get_max_timestep

# Short training proxy used during each trial — not a full training run.
TRIAL_EPOCHS = 5


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


# ---------------------------------------------------------------------------
# Per-model objective functions — each returns a scalar PyHopper maximises.
# ---------------------------------------------------------------------------

def _objective_classification(params, cfg, trainloader, valloader, device):
    num_classes = cfg['data']['num_classes']
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.get('learning_rate', cfg['training']['learning_rate']),
        weight_decay=params.get('weight_decay', cfg['training']['weight_decay']),
    )

    for _ in range(TRIAL_EPOCHS):
        model.train()
        for inputs, labels in trainloader:
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


def _objective_robust_classification(params, cfg, trainloader, valloader, device):
    num_classes = cfg['data']['num_classes']
    num_timesteps = 1000
    model = TimeDependentResNet(num_classes)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params.get('learning_rate', cfg['training']['learning_rate']),
        weight_decay=params.get('weight_decay', cfg['training']['weight_decay']),
    )

    betas = torch.linspace(0.0001, 0.02, num_timesteps, device=device)
    alphas_cumprod = torch.cumprod(1 - betas, dim=0)

    epsilon   = params.get('epsilon',    0.03)
    alpha     = params.get('alpha',      0.01)
    num_steps = params.get('num_steps',  10)

    for epoch in range(TRIAL_EPOCHS):
        model.train()
        max_t = get_max_timestep(epoch, TRIAL_EPOCHS, num_timesteps)
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            t = torch.randint(0, max(1, max_t), (inputs.shape[0],), device=device)
            x_t = get_noisy_image(inputs, t, alphas_cumprod)
            x_adv = pgd_attack_early_stop(model, x_t, t, labels,
                                          epsilon=epsilon, alpha=alpha, num_steps=num_steps)
            optimizer.zero_grad()
            F.cross_entropy(model(x_adv, t), labels).backward()
            optimizer.step()

    # Evaluate clean accuracy on val set at timestep 0
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            t = torch.zeros(inputs.shape[0], dtype=torch.long, device=device)
            correct += (model(inputs, t).argmax(1) == labels).sum().item()
            total += labels.size(0)

    return correct / total  # maximise clean val accuracy


def _objective_diffusion(params, cfg, trainloader, valloader, device):
    num_classes = cfg['data']['num_classes']
    label_dropout = params.get('label_dropout', 0.15)

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
        cross_attention_dim=256,
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)
    class_emb = nn.Embedding(num_classes + 1, 256).to(device)

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(class_emb.parameters()),
        lr=float(params.get('learning_rate', cfg['training']['learning_rate'])),
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


# ---------------------------------------------------------------------------
# Registry — register a new model type by adding one entry here.
# ---------------------------------------------------------------------------

_OBJECTIVES = {
    'classification':        _objective_classification,
    'robust_classification': _objective_robust_classification,
    'diffusion':             _objective_diffusion,
}


def optimize_parameters(model_type, cfg, trainloader, valloader, device, result_directory):
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
    objective = lambda params: objective_fn(params, cfg, trainloader, valloader, device)

    print(f"Starting {model_type} optimization: {max_steps} trials over {list(space)}")
    search = pyhopper.Search(space)
    best = search.run(objective, direction="max", steps=max_steps)

    out_path = os.path.join(result_directory, 'best_params.json')
    with open(out_path, 'w') as f:
        json.dump(dict(best), f, indent=2)

    print(f"Best params for {model_type}: {best}")
    print(f"Saved to {out_path}")

    return best
