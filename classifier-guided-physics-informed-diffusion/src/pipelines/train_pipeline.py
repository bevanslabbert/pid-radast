from src.utils.data import get_data_loaders
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.models.time_dependent_resnet import TimeDependentResNet
from src.utils.augmentation import pgd_attack_early_stop, get_max_timestep, get_noisy_image
from src.datasets.mirabest.MiraBestFITS import MiraBestFITS
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
from torchmetrics.image.fid import FrechetInceptionDistance
import matplotlib.pyplot as plt
import numpy as np
import os

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

def train_diffusion(config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=None):
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    unet = UNet2DConditionModel(
        sample_size=config['data']['input_size'],
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 512), # Increased capacity for scientific data
        down_block_types=(
            "DownBlock2D",         # 150x150
            "DownBlock2D",         # 75x75
            "CrossAttnDownBlock2D", # 37x37 (Attention helps here)
            "CrossAttnDownBlock2D", # 18x18
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",   # 18x18
            "CrossAttnUpBlock2D",   # 37x37
            "UpBlock2D",           # 75x75
            "UpBlock2D",           # 150x150
        ),
        cross_attention_dim=256,
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Embed class labels
    num_classes = config['data']['num_classes'] # to account for null class
    num_epochs = config['training']['epochs']
    class_emb = nn.Embedding(num_classes + 1, 256).to(device)

    start_epoch = 0

    loss_history = []
    val_loss_history = []
    epochs_range = []
    fid_history = []

    # optimizer resume logic
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(class_emb.parameters()), 
        lr=float(config['training']['learning_rate'])
    )

    if resume is not None:
        checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/diffusion', device)

        if checkpoint.get('rng_state') is not None:
            # Force the state to the correct type for the CPU generator
            rng_state = checkpoint['rng_state'].to('cpu').to(torch.uint8)
            torch.set_rng_state(rng_state)

        if checkpoint.get('cuda_rng_state') is not None:
            # CUDA states can be a list (for multiple GPUs) or a single tensor
            cuda_state = checkpoint['cuda_rng_state']
            if isinstance(cuda_state, torch.Tensor):
                torch.cuda.set_rng_state(cuda_state.to('cpu').to(torch.uint8))
            else:
                # If it's a list of states for multiple GPUs
                torch.cuda.set_rng_state_all([s.to('cpu').to(torch.uint8) for s in cuda_state])

        unet.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        class_emb.load_state_dict(checkpoint['class_emb_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint['loss_history']
        val_loss_history = checkpoint['val_loss_history']
        epochs_range = checkpoint['epochs_range']
        fid_history = checkpoint['fid_history']
        print(f"Resumed from checkpoint: {resume} (epoch {start_epoch})")

    if resume is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # --- Training loop ---
    for epoch in range(start_epoch, num_epochs):
        unet.train()
        epoch_loss = 0
        batch_count = 0

        print(f'Epoch {epoch}')
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # dropout 15% of labels to train on unclassified images
            drop_mask = torch.rand(labels.shape, device=device) < 0.15
            training_labels = labels.clone()
            training_labels[drop_mask] = num_classes

            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, noise, t)

            # get class embeddings and add sequence dimension
            class_embeddings = class_emb(training_labels).unsqueeze(1)  # (B, 1, D)

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
        val_loss_accum = 0
        
        with torch.no_grad():
            for i, (val_images, val_labels) in enumerate(testloader):
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                batch_sz = val_images.size(0)

                # 1. CALCULATE VAL LOSS (Fast)
                t_val = torch.randint(0, scheduler.num_train_timesteps, (batch_sz,), device=device)
                noise_val = torch.randn_like(val_images)
                noisy_val = scheduler.add_noise(val_images, noise_val, t_val)
                
                class_emb_val = class_emb(val_labels).unsqueeze(1)
                model_output = unet(noisy_val, t_val, encoder_hidden_states=class_emb_val).sample
                
                v_loss = F.mse_loss(model_output, noise_val)
                val_loss_accum += v_loss.item()

            # --- Finalize Metrics for the Epoch ---
            avg_val_loss = val_loss_accum / len(testloader)
            
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            
            # Save to history
            val_loss_history.append(avg_val_loss)

        # save a sample image every x epochs
        if epoch % 5 == 0:
            unet.eval()
            with torch.no_grad():
                # 1. Generate images for both classes
                # Assuming these return a batch of images [B, 1, 150, 150]
                zero_images = sample_from_model_zeros(unet, scheduler, class_emb, 4, num_classes, device)
                one_images = sample_from_model_ones(unet, scheduler, class_emb, 4, num_classes, device)

                # 2. Combine into a single comparison plot
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # Helper to process tensors for plotting
                def prep_for_plot(img_tensor):
                    grid = torchvision.utils.make_grid(img_tensor, nrow=2, normalize=True, value_range=(-1, 1))
                    return grid.permute(1, 2, 0).cpu().numpy()

                # Display Class 0
                axes[0].imshow(prep_for_plot(zero_images), cmap='gray')
                axes[0].set_title(f"Class 0 (Epoch {epoch})")
                axes[0].axis('off')

                # Display Class 1
                axes[1].imshow(prep_for_plot(one_images), cmap='gray')
                axes[1].set_title(f"Class 1 (Epoch {epoch})")
                axes[1].axis('off')

                # 3. Save the single comparison file
                plt.tight_layout()
                plt.savefig(f"{result_directory}/comparison_epoch_{epoch}.png")
                plt.close() # Important to avoid memory leaks

                # If trained on FITS data, also save as FITS with inverted scaling
                if isinstance(dataset, MiraBestFITS):
                    fits_dir = os.path.join(result_directory, 'generated_fits')
                    os.makedirs(fits_dir, exist_ok=True)

                    for class_idx, images in [(0, zero_images), (1, one_images)]:
                        for i, img in enumerate(images):
                            # img shape: (1, H, W) tensor in [-1, 1]
                            norm_array = img.squeeze(0).cpu().numpy()          # (H, W)
                            jy_array = dataset.denormalise(norm_array)         # approximate Jy/beam
                            fname = os.path.join(fits_dir, f"generated_class{class_idx}_{i:03d}_{epoch}.fits")
                            MiraBestFITS.write_fits(jy_array, fname)

                    print(f"FITS files saved to {fits_dir}")

            unet.train()

        # save checkpoint for resuming
        if not checkpoint == None or not resume == None:
            save_checkpoint(
                {
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
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                },
                f'{CHECKPOINT_DIR}/diffusion'
            )

    num_samples = config['data']['batch_size']

    class_0_images = sample_from_model_zeros(unet, scheduler, class_emb, num_samples, num_classes, device)
    class_1_images = sample_from_model_ones(unet, scheduler, class_emb, num_samples, num_classes, device)
    random_images  = sample_from_model(unet, scheduler, class_emb, num_samples, num_classes, device)

    # Always save PNG previews
    torchvision.utils.save_image(class_0_images, f"{result_directory}/generated_images_class_0.png", nrow=2, normalize=True, value_range=(-1, 1))
    torchvision.utils.save_image(class_1_images, f"{result_directory}/generated_images_class_1.png", nrow=2, normalize=True, value_range=(-1, 1))
    torchvision.utils.save_image(random_images,  f"{result_directory}/generated_images_random_all_classes.png", nrow=2, normalize=True, value_range=(-1, 1))

    # If trained on FITS data, also save as FITS with inverted scaling
    if isinstance(dataset, MiraBestFITS):
        fits_dir = os.path.join(result_directory, 'generated_fits')
        os.makedirs(fits_dir, exist_ok=True)

        for class_idx, images in [(0, class_0_images), (1, class_1_images)]:
            for i, img in enumerate(images):
                # img shape: (1, H, W) tensor in [-1, 1]
                norm_array = img.squeeze(0).cpu().numpy()          # (H, W)
                jy_array = dataset.denormalise(norm_array)         # approximate Jy/beam
                fname = os.path.join(fits_dir, f"generated_class{class_idx}_{i:03d}.fits")
                MiraBestFITS.write_fits(jy_array, fname)

        print(f"FITS files saved to {fits_dir}")

    save_training_plot(epochs_range, loss_history, val_loss_history, result_directory)

    print(f"Generated images saved.")

    return unet

def sample_from_model(model, scheduler, class_emb, num_samples, num_classes, device, shape=(1, 150, 150)):
    model.eval()
    # Random target labels for validation
    labels = torch.randint(0, num_classes, (num_samples,), device=device)
    print("Labels:")
    print(labels)
    class_embeddings = class_emb(labels).unsqueeze(1)
    
    scheduler.set_timesteps(1000) # Use fewer steps for validation to save time
    images = torch.randn((num_samples, *shape), device=device)
    
    for t in scheduler.timesteps:
        with torch.no_grad():
            noise_pred = model(images, t, encoder_hidden_states=class_embeddings).sample
            images = scheduler.step(noise_pred, t, images).prev_sample
    return images

def sample_from_model_zeros(model, scheduler, class_emb, num_samples, num_classes, device, shape=(1, 150, 150), guidance_scale=7.5):
    print("Generating class 0 images")
    model.eval()

    # 1. Prepare conditional (Class 0) and unconditional (Null Class) labels
    cond_labels = torch.zeros(num_samples, dtype=torch.long, device=device)
    uncond_labels = torch.full((num_samples,), num_classes, dtype=torch.long, device=device)
    
    cond_emb = class_emb(cond_labels).unsqueeze(1)
    uncond_emb = class_emb(uncond_labels).unsqueeze(1)
    
    scheduler.set_timesteps(50) # Use fewer steps for validation to save time
    images = torch.randn((num_samples, *shape), device=device)
    
    for t in scheduler.timesteps:
        # Expand images to run both cond and uncond in one batch
        model_input = torch.cat([images] * 2)
        combined_emb = torch.cat([uncond_emb, cond_emb])

        with torch.no_grad():
            # Predict noise for both versions
            output = model(model_input, t, encoder_hidden_states=combined_emb).sample
            noise_pred_uncond, noise_pred_cond = output.chunk(2)

            # 3. APPLY CFG MATH: Extrapolate away from 'uncond' towards 'cond'
            # This sharpens the image and removes the "grey noise"
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Step the scheduler
            images = scheduler.step(noise_pred, t, images).prev_sample
    return images

def sample_from_model_ones(model, scheduler, class_emb, num_samples, num_classes, device, shape=(1, 150, 150), guidance_scale=7.5):
    print("Generating class 1 images")
    model.eval()

    # 1. Prepare conditional (Class 0) and unconditional (Null Class) labels
    cond_labels = torch.ones(num_samples, dtype=torch.long, device=device)
    uncond_labels = torch.full((num_samples,), num_classes, dtype=torch.long, device=device)
    
    cond_emb = class_emb(cond_labels).unsqueeze(1)
    uncond_emb = class_emb(uncond_labels).unsqueeze(1)
    
    scheduler.set_timesteps(50) # Use fewer steps for validation to save time
    images = torch.randn((num_samples, *shape), device=device)
    
    for t in scheduler.timesteps:
        # Expand images to run both cond and uncond in one batch
        model_input = torch.cat([images] * 2)
        combined_emb = torch.cat([uncond_emb, cond_emb])

        with torch.no_grad():
            # Predict noise for both versions
            output = model(model_input, t, encoder_hidden_states=combined_emb).sample
            noise_pred_uncond, noise_pred_cond = output.chunk(2)

            # 3. APPLY CFG MATH: Extrapolate away from 'uncond' towards 'cond'
            # This sharpens the image and removes the "grey noise"
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # Step the scheduler
            images = scheduler.step(noise_pred, t, images).prev_sample
    return images

def prepare_for_fid(t):
        t = t.repeat(1, 3, 1, 1)          # 1 channel -> 3 channels
        t = (t + 1.0) / 2.0               # -1..1 -> 0..1
        t = (t * 255).clamp(0, 255)       # 0..1 -> 0..255
        return t.to(torch.uint8)          # Float -> Byte

def save_training_plot(epochs, losses, val_losses, result_dir="results"):
    # Initialize the plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Axis labels
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE Loss')
    
    # Plot Training Loss (solid line)
    ax1.plot(epochs, losses, color='tab:blue', linewidth=2, label='Training Loss')
    
    # Plot Validation Loss (dotted line for distinction)
    ax1.plot(epochs, val_losses, color='tab:cyan', linewidth=2, linestyle=':', label='Validation Loss')
    
    # Aesthetics
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend(loc='upper right')
    plt.title('Diffusion Training: MSE Loss Trends')
    
    # Layout and Saving
    fig.tight_layout()
    os.makedirs(result_dir, exist_ok=True)
    
    plot_path = os.path.join(result_dir, "training_metrics.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"📈 Loss graph saved to {plot_path}")

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


def train_model(model, config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=None):
    print(f"Training {model} for {config['training']['epochs']} epochs")
    if model == 'classification':
        return train_classification(config, trainloader, valloader, device, result_directory, resume, checkpoint)
    elif model == 'robust_classification':
        return train_robust_classification(config, trainloader, device, result_directory, resume, checkpoint)
    elif model == 'diffusion':
        return train_diffusion(config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=dataset)
    else:
        raise f'Model {model} not supported ["diffusion", "robust_classification, "classification"]'
