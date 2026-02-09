from src.utils.data import get_data_loaders
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.models.time_dependent_resnet import TimeDependentResNet
from src.utils.augmentation import pgd_attack_early_stop, get_max_timestep, get_noisy_image
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt

CHECKPOINT_DIR = 'checkpoints'
 
def evaluate_loss(model, dataloader, criterion, device='cpu'):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def train_classification(config, trainloader, valloader, device, result_directory, resume, checkpoint):

    # model definition
    model = resnet50(pretrained=True)

    num_classes = config['data']['num_classes']

    # replace last layer to match number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # initialize values
    print(f"Setting model to {device}")
    model.to(device)
    num_epochs = config['training']['epochs']
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    epoch_losses = []
    val_losses = []

    # for early stopping
    best_val_loss = torch.inf
    patience = 10
    patience_counter = 0

    start_epoch = 0

    if resume is not None:
        checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/classification', device)
        model.loa. _state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {resume} (epoch {start_epoch})")

    # train model
    for epoch in range(start_epoch, num_epochs):
        total_loss = 0.0
        model.train()

        print(f"Epoch {epoch}")
        for idx, batch in enumerate(trainloader):
            print(f"Batch {idx}")
            inputs = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)
        epoch_losses.append(avg_loss)

        # Evaluate validation loss after epoch
        avg_val_loss = evaluate_loss(model, valloader, criterion, device)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter > patience:
                break

        print(f'Epoch {epoch}, Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # save checkpoint for resuming
        if not checkpoint == None and not resume == None:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'config': config
                },
                f'{CHECKPOINT_DIR}/classification'
            )

    plt.figure(figsize=(8, 5))
    plt.plot(epoch_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    print(f'saving in {result_directory}/classifier_loss_plot.png')
    plt.savefig(f'{result_directory}/classifier_loss_plot.png') 

    return model

def train_diffusion(config, trainloader, device, result_directory, resume, checkpoint):
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # --- UNet that supports class conditioning ---
    unet = UNet2DConditionModel(
        sample_size=224,
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 64, 128, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=128,   # needed for conditioning
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Embed class labels
    num_classes = config['data']['num_classes']
    num_epochs = config['training']['epochs']
    class_emb = nn.Embedding(num_classes, 128).to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

    start_epoch = 0

    if resume is not None:
        checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/diffusion', device)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        class_emb.load_state_dict(checkpoint['class_emb_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {resume} (epoch {start_epoch})")

    # Initialize (2048 is the standard feature dimension for Inception)
    fid = FrechetInceptionDistance(feature=2048).to(device)

    loss_history = []
    epochs_range = []
    fid_history = []

    # --- Training loop ---
    for epoch in range(start_epoch, num_epochs):
        unet.train()
        epoch_loss = 0
        batch_count = 0

        print(f'Epoch {epoch}')
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, noise, t)

            # get class embeddings and add sequence dimension
            class_embeddings = class_emb(labels).unsqueeze(1)  # (B, 1, D)

            # predict noise conditioned on class
            model_output = unet(noisy_images, t, encoder_hidden_states=class_embeddings).sample

            loss = F.mse_loss(model_output, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        epochs_range.append(epoch)

        # --- Inside your validation block ---
        unet.eval()
        with torch.no_grad():
            # 1. Generate fake images [B, 1, H, W]
            fake_images = sample_from_model(
                model=unet,
                scheduler=scheduler,
                class_emb=class_emb,
                num_samples=config['data']['batch_size'],
                num_classes=num_classes,
                device=device
            )
            
            # 2. Get a batch of real images [B, 1, H, W]
            real_images, _ = next(iter(trainloader))
            real_images = real_images.to(device)

            # 3. Convert both to RGB for the FID metric
            fid.update(prepare_for_fid(real_images), real=True)
            fid.update(prepare_for_fid(fake_images), real=False)

            current_fid = fid.compute().item()
            print(f"FID Score: {current_fid}")
            fid_history.append(current_fid)
            fid.reset()

        print(f"Epoch {epoch+1}, loss={loss.item():.4f}")

        # save checkpoint for resuming
        if not checkpoint == None or not resume == None:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'class_emb_state_dict': class_emb.state_dict(),
                    'loss': loss,
                    'config': config
                },
                f'{CHECKPOINT_DIR}/diffusion'
            )

    images = sample_from_model(
        model=unet,
        scheduler=scheduler,
        class_emb=class_emb,
        num_samples=config['data']['batch_size'],
        num_classes=num_classes,
        device=device
    )

    torchvision.utils.save_image(
        images, 
        f"{result_directory}/generated_images.png", 
        nrow=2, 
        normalize=True, 
        value_range=(-1, 1)
    )

    save_training_plot(epochs_range, loss_history, fid_history, result_directory)

    print(f"✅ Generated images saved to PNG.")

    return unet

def sample_from_model(model, scheduler, class_emb, num_samples, num_classes, device, shape=(1, 224, 224)):
    model.eval()
    # Random target labels for validation
    labels = torch.randint(0, num_classes, (num_samples,), device=device)
    class_embeddings = class_emb(labels).unsqueeze(1)
    
    scheduler.set_timesteps(50) # Use fewer steps for validation to save time
    images = torch.randn((num_samples, *shape), device=device)
    
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(images, t, encoder_hidden_states=class_embeddings).sample
            images = scheduler.step(noise_pred, t, images).prev_sample
    return images

def prepare_for_fid(t):
        t = t.repeat(1, 3, 1, 1)          # 1 channel -> 3 channels
        t = (t + 1.0) / 2.0               # -1..1 -> 0..1
        t = (t * 255).clamp(0, 255)       # 0..1 -> 0..255
        return t.to(torch.uint8)          # Float -> Byte

def save_training_plot(epochs, losses, fids, result_dir):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Primary axis: Loss
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss', color=color)
    ax1.plot(epochs, losses, color=color, linewidth=2, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)

    # Secondary axis: FID
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('FID (Lower is better)', color=color)
    ax2.plot(epochs, fids, color=color, linewidth=2, linestyle='--', label='FID Score')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Diffusion Training Efficiency: Loss vs. FID')
    fig.tight_layout()
    
    plot_path = f"{result_dir}/training_metrics.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"📈 Efficiency graph saved to {plot_path}")

def train_robust_classification(config, trainloader, device, result_directory, resume, checkpoint):
    # model definition
    num_classes = config['data']['num_classes']
    rob_model = TimeDependentResNet(num_classes)

    # initialize values
    rob_model.to(device)
    num_epochs = config['training']['epochs']
    num_timesteps = 1000
    optimizer = torch.optim.Adam(rob_model.parameters(), lr=1e-4, weight_decay=1e-4)
    epoch_losses = []
    val_losses = []

    # Define diffusion noise schedule (linear beta schedule)
    betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    start_epoch = 0

    if resume is not None:
        checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/robust_classification', device)
        rob_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {resume} (epoch {start_epoch})")

    # train model
    for epoch in range(start_epoch, num_epochs):

        total_loss = 0.0
        rob_model.train()

        # Calculate current max timestep
        max_t = get_max_timestep(epoch, num_epochs, num_timesteps)

        for idx, batch in enumerate(trainloader):
            print(f"Index {idx}")
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            batch_size = inputs.shape[0]

            # Step 1: Sample random timesteps for each image
            t = torch.randint(0, max(1, max_t), (batch_size,), device=device)

            # Step 2: Add Gaussian noise to create x_t
            x_t = get_noisy_image(inputs, t, alphas_cumprod)

            # Step 3: Apply adversarial attack with early stopping
            x_tilde = pgd_attack_early_stop(
                rob_model, x_t, t, labels,
                epsilon=0.03,
                alpha=0.01,
                num_steps=10,
                random_start=False
            )

            # Step 4: Train on adversarial examples
            optimizer.zero_grad()
            logits = rob_model(x_tilde, t)
            loss = F.cross_entropy(logits, labels)

            # Step 5: Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(trainloader)
        epoch_losses.append(avg_loss)

        print(f'Epoch {epoch}, Training Loss: {avg_loss:.4f}')

        # save checkpoint for resuming
        if not checkpoint == None and not resume == None:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': rob_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'config': config
                },
                f'{CHECKPOINT_DIR}/robust_classification'
            )

    plt.figure(figsize=(8, 5))
        
    # Ensure we only plot the average losses per epoch
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, label='Training Loss', marker='o')
    
    # Only plot validation if there is actually data in it
    if val_losses:
        plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', marker='s')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)

    print(f'saving in {result_directory}/robust_classifier_loss_plot.png')
    plt.savefig(f'{result_directory}/robust_classifier_loss_plot.png') 

    # torch.save(rob_model.state_dict(), f'{result_directory}/state_dict.pth') # save this config
    return rob_model


def train_model(model, config, trainloader, valloader, device, result_directory, resume, checkpoint):
    print(f"🚀 Training {model} for {config['training']['epochs']} epochs")
    if model == 'classification':
        return train_classification(config, trainloader, valloader, device, result_directory, resume, checkpoint)
    elif model == 'robust_classification':
        return train_robust_classification(config, trainloader, device, result_directory, resume, checkpoint)
    elif model == 'diffusion':
        return train_diffusion(config, trainloader, device, result_directory, resume, checkpoint)
    else:
        raise f'Model {model} not supported ["diffusion", "robust_classification, "classification"]'
