from src.utils.data import get_data_loaders
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.models.time_dependent_resnet import TimeDependentResNet
from src.models.diffusion import build_diffusion_components, train_epoch, eval_epoch
from diffusers import UNet2DConditionModel, DDPMScheduler
from src.models.pid import (
    estimate_x0, physics_loss, symmetry_loss,
    sample_pid_zeros, sample_pid_ones,
)
from src.utils.augmentation import pgd_attack_early_stop, get_max_timestep, get_noisy_image
from src.datasets.mirabest.MiraBestFITS import MiraBestFITS
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
import matplotlib.pyplot as plt
import numpy as np
import math
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
        cross_attention_dim=config['model']['embedding_dim'],
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=config['training']['num_train_timesteps'])

    # Embed class labels
    num_classes = config['data']['num_classes'] # to account for null class
    num_epochs = config['training']['epochs']
    class_emb = nn.Embedding(num_classes + 1, config['model']['embedding_dim']).to(device)

    start_epoch = 0

    loss_history = []
    val_loss_history = []
    epochs_range = []
    fid_history = []
    kid_history = []
    fid_epochs = []

    # optimizer resume logic
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(class_emb.parameters()),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
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
        kid_history = checkpoint.get('kid_history', [])
        fid_epochs = checkpoint.get('fid_epochs', [])
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

            # dropout labels to train on unclassified images (classifier-free guidance)
            drop_mask = torch.rand(labels.shape, device=device) < config['training']['label_dropout']
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
                t_val = torch.linspace(0, scheduler.num_train_timesteps - 1, batch_sz, dtype=torch.long, device=device)
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

                # compute distributional quality metrics against the validation set
                print(f"Computing FID/KID at epoch {epoch}...")
                fid_score, kid_score = compute_fid_kid(
                    unet, scheduler, class_emb, num_classes, valloader, device,
                    sample_from_model_zeros, sample_from_model_ones,
                )
                fid_history.append(fid_score)
                kid_history.append(kid_score)
                fid_epochs.append(epoch)
                print(f"  FID: {fid_score:.4f} | KID: {kid_score:.6f}")

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
                    'kid_history': kid_history,
                    'fid_epochs': fid_epochs,
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
    save_generative_metrics_plot(fid_epochs, fid_history, kid_history, result_directory)

    torch.save(
        {'model_state_dict': unet.state_dict(), 'class_emb_state_dict': class_emb.state_dict(), 'config': config},
        os.path.join(result_directory, 'final_weights.pt')
    )
    print(f"Generated images saved.")

    return unet

def train_classifier_guided_diffusion(config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=None):
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    unet = UNet2DConditionModel(
        sample_size=config['data']['input_size'],
        in_channels=1,
        out_channels=1,
        layers_per_block=2,
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
        cross_attention_dim=config['model']['embedding_dim'],
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=config['training']['num_train_timesteps'])

    num_classes = config['data']['num_classes']
    num_epochs = config['training']['epochs']
    class_emb = nn.Embedding(num_classes + 1, config['model']['embedding_dim']).to(device)

    # load pre-trained diffusion weights as the starting point for classifier-guided fine-tuning.
    # defaults to checkpoints/diffusion but can be overridden via model.pretrained_checkpoint in config.
    pretrained_dir = config['model'].get('pretrained_checkpoint', f'{CHECKPOINT_DIR}/diffusion')
    pretrained_ckpt_path = os.path.join(pretrained_dir, 'state.pt')
    if os.path.exists(pretrained_ckpt_path):
        pretrained = load_checkpoint(pretrained_dir, device)
        unet.load_state_dict(pretrained['model_state_dict'])
        class_emb.load_state_dict(pretrained['class_emb_state_dict'])
        print(f"Loaded pre-trained diffusion weights from {pretrained_dir}")
    else:
        print(f"No pre-trained checkpoint found at {pretrained_dir}, training from scratch")

    start_epoch = 0

    loss_history = []
    val_loss_history = []
    epochs_range = []
    fid_history = []
    kid_history = []
    fid_epochs = []

    # optimizer resume logic
    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(class_emb.parameters()),
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
    )

    if resume is not None:
        checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/classifier_guided_diffusion', device)

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
        kid_history = checkpoint.get('kid_history', [])
        fid_epochs = checkpoint.get('fid_epochs', [])
        print(f"Resumed from checkpoint: {resume} (epoch {start_epoch})")

    if resume is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # --- Training loop ---

    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    classifier = TimeDependentResNet(num_classes, pretrained=False)
    classifier.load_state_dict(load_checkpoint(f'{CHECKPOINT_DIR}/robust_classification', device)['model_state_dict'])
    classifier.to(device)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    lambda_cls = float(config['training'].get('lambda_cls', 0.1))

    for epoch in range(start_epoch, num_epochs):
        unet.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        batch_count = 0

        print(f'Epoch {epoch}')
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # dropout labels to train on unclassified images (classifier-free guidance)
            drop_mask = torch.rand(labels.shape, device=device) < config['training']['label_dropout']
            training_labels = labels.clone()
            training_labels[drop_mask] = num_classes

            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, noise, t)

            # get class embeddings and add sequence dimension
            class_embeddings = class_emb(training_labels).unsqueeze(1)  # (B, 1, D)

            # predict noise conditioned on class
            model_output = unet(noisy_images, t, encoder_hidden_states=class_embeddings).sample

            # classifier guidance: penalise the UNet when the estimated clean image is misclassified.
            # loss = (1 - p_correct), where p_correct = softmax probability assigned to the true class.
            # this is exactly 0 when the classifier is certain and right, and equals the classifier's
            # confidence in the wrong answer when misclassified — matching the intended formulation.
            # only applied to non-dropped samples (null class has no target label).
            cls_loss = torch.tensor(0.0, device=device)
            cls_mask = training_labels != num_classes
            if cls_mask.any():
                x_0_pred = estimate_x0(noisy_images, model_output, alphas_cumprod, t)
                cls_input = x_0_pred[cls_mask]
                if isinstance(dataset, MiraBestFITS):
                    cls_input = fits_to_linear(cls_input, dataset)
                t_clean = torch.zeros(cls_mask.sum(), dtype=torch.long, device=device)
                cls_logits = classifier(cls_input, t_clean)
                p_correct = F.softmax(cls_logits, dim=1).gather(1, labels[cls_mask].unsqueeze(1)).squeeze(1)
                cls_loss = lambda_cls * (1.0 - p_correct).mean()

            loss = F.mse_loss(model_output, noise) + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            epoch_cls_loss += cls_loss.item()

        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        avg_cls_loss = epoch_cls_loss / batch_count
        epochs_range.append(epoch)

        # --- Inside your validation block ---
        unet.eval()
        val_loss_accum = 0
        
        with torch.no_grad():
            for i, (val_images, val_labels) in enumerate(testloader):
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                batch_sz = val_images.size(0)

                # 1. CALCULATE VAL LOSS (Fast)
                t_val = torch.linspace(0, scheduler.num_train_timesteps - 1, batch_sz, dtype=torch.long, device=device)
                noise_val = torch.randn_like(val_images)
                noisy_val = scheduler.add_noise(val_images, noise_val, t_val)

                class_emb_val = class_emb(val_labels).unsqueeze(1)
                model_output = unet(noisy_val, t_val, encoder_hidden_states=class_emb_val).sample

                v_loss = F.mse_loss(model_output, noise_val)
                val_loss_accum += v_loss.item()

            # --- Finalize Metrics for the Epoch ---
            avg_val_loss = val_loss_accum / len(testloader)
            
            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Cls Loss: {avg_cls_loss:.6f}")
            
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

                # compute distributional quality metrics against the validation set
                print(f"Computing FID/KID at epoch {epoch}...")
                fid_score, kid_score = compute_fid_kid(
                    unet, scheduler, class_emb, num_classes, valloader, device,
                    sample_from_model_zeros, sample_from_model_ones,
                )
                fid_history.append(fid_score)
                kid_history.append(kid_score)
                fid_epochs.append(epoch)
                print(f"  FID: {fid_score:.4f} | KID: {kid_score:.6f}")

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
                    'kid_history': kid_history,
                    'fid_epochs': fid_epochs,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                },
                f'{CHECKPOINT_DIR}/classifier_guided_diffusion'
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
    save_generative_metrics_plot(fid_epochs, fid_history, kid_history, result_directory)

    torch.save(
        {'model_state_dict': unet.state_dict(), 'class_emb_state_dict': class_emb.state_dict(), 'config': config},
        os.path.join(result_directory, 'final_weights.pt')
    )
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

def fits_to_linear(x_fits, dataset):
    """Convert a FITS log-SNR normalised tensor to linear-normalised [-1, 1].

    Fully differentiable — safe to use inside a training step where gradients
    must flow back through x_0_pred to the UNet.

    Derivation:
      FITS normalisation: x = sign(snr) * log1p(|snr|) / peak_log
      Inverse to Jy/beam: jy = sign(x) * expm1(|x| * peak_log) * noise_rms
      Linear normalisation: divide by max Jy/beam = expm1(peak_log) * noise_rms
      noise_rms cancels → x_linear = sign(x) * expm1(|x| * peak_log) / expm1(peak_log)
    """
    peak_log = dataset.median_peak_log
    return torch.sign(x_fits) * torch.expm1(torch.abs(x_fits) * peak_log) / math.expm1(peak_log)

def prepare_for_fid(t):
    t = t.repeat(1, 3, 1, 1)          # 1 channel -> 3 channels
    t = (t + 1.0) / 2.0               # -1..1 -> 0..1
    t = (t * 255).clamp(0, 255)       # 0..1 -> 0..255
    return t.to(torch.uint8)          # float -> uint8

def compute_fid_kid(unet, scheduler, class_emb, num_classes, valloader, device,
                    sample_zeros_fn, sample_ones_fn, num_gen_per_class=16):
    """Compute FID and KID between real val images and CFG-generated images.

    Generates `num_gen_per_class` images for each class (total 2×), feeds all
    real validation images as the reference distribution, then returns
    (fid_score, kid_mean).  Both metrics use Inception v3 pool3 features.
    KID is preferred over FID on small datasets like MiraBest (~1 200 images)
    because it is an unbiased estimator.
    """
    n_fake = num_gen_per_class * 2
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid_metric = KernelInceptionDistance(
        feature=2048, normalize=False, subset_size=min(n_fake, 50)
    ).to(device)

    with torch.no_grad():
        for real_imgs, _ in valloader:
            real_u8 = prepare_for_fid(real_imgs.to(device))
            fid_metric.update(real_u8, real=True)
            kid_metric.update(real_u8, real=True)

        gen_0 = sample_zeros_fn(unet, scheduler, class_emb, num_gen_per_class, num_classes, device)
        gen_1 = sample_ones_fn(unet, scheduler, class_emb, num_gen_per_class, num_classes, device)
        fake_u8 = prepare_for_fid(torch.cat([gen_0, gen_1], dim=0))
        fid_metric.update(fake_u8, real=False)
        kid_metric.update(fake_u8, real=False)

    fid_score = fid_metric.compute().item()
    kid_mean, _ = kid_metric.compute()
    return fid_score, kid_mean.item()

def save_generative_metrics_plot(fid_epochs, fid_history, kid_history, result_dir):
    """Save a two-panel figure showing FID and KID over training epochs."""
    if not fid_epochs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Generative Quality Metrics (lower is better)', fontsize=14)

    axes[0].plot(fid_epochs, fid_history, color='tab:purple', linewidth=2, marker='o')
    axes[0].set_title('FID')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('FID score')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(fid_epochs, kid_history, color='tab:orange', linewidth=2, marker='o')
    axes[1].set_title('KID')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('KID mean')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plot_path = os.path.join(result_dir, 'generative_metrics.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Generative metrics plot saved to {plot_path}")

def save_pid_training_plots(epochs, loss_history, val_loss_history,
                            mse_history, sym_history, neg_history,
                            compliance_epochs, pct_negative_history, sym_score_history,
                            result_dir="results"):
    """Save two figures summarising a PID training run.

    Figure 1 — loss decomposition: total train/val loss plus the three
    component losses (MSE, symmetry, non-negativity) on separate subplots so
    it is easy to see which term is dominating and whether each is converging.

    Figure 2 — physics compliance on generated samples: % negative pixels and
    symmetry score measured every few epochs on actual generated images, showing
    whether the constraints are working in practice and not just in the loss.
    """
    # --- Figure 1: loss decomposition ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PID Training: Loss Decomposition', fontsize=14)

    axes[0, 0].plot(epochs, loss_history, color='tab:blue', linewidth=2, label='Train')
    axes[0, 0].plot(epochs, val_loss_history, color='tab:cyan', linewidth=2, linestyle=':', label='Val')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)

    axes[0, 1].plot(epochs, mse_history, color='tab:orange', linewidth=2)
    axes[0, 1].set_title('MSE Loss (noise prediction)')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)

    axes[1, 0].plot(epochs, sym_history, color='tab:green', linewidth=2)
    axes[1, 0].set_title(f'Symmetry Loss (λ={sym_history[0]:.0e} scale)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)

    axes[1, 1].plot(epochs, neg_history, color='tab:red', linewidth=2)
    axes[1, 1].set_title('Non-negativity Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    plt.savefig(os.path.join(result_dir, 'pid_loss_decomposition.png'), dpi=150)
    plt.close()

    # --- Figure 2: physics compliance on generated samples ---
    if compliance_epochs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('PID Physics Compliance (generated images)', fontsize=14)

        axes[0].plot(compliance_epochs, pct_negative_history, color='tab:red', linewidth=2, marker='o')
        axes[0].set_title('% Negative Pixels')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('% of pixels < 0')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        axes[1].plot(compliance_epochs, sym_score_history, color='tab:green', linewidth=2, marker='o')
        axes[1].set_title('Symmetry Score (lower = more symmetric)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MSE(image, flipped)')
        axes[1].grid(True, linestyle='--', alpha=0.5)

        fig.tight_layout()
        plt.savefig(os.path.join(result_dir, 'pid_physics_compliance.png'), dpi=150)
        plt.close()

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

def train_robust_classification(config, trainloader, valloader, device, result_directory, resume, checkpoint):
    # model definition
    num_classes = config['data']['num_classes']
    rob_model = TimeDependentResNet(num_classes)

    # initialize values
    rob_model.to(device)
    num_epochs        = config['training']['epochs']
    warmup_epochs     = config['training'].get('warmup_epochs',    20)
    transition_epochs = config['training'].get('transition_epochs', 15)
    label_smoothing   = float(config['training'].get('label_smoothing', 0.1))
    num_timesteps     = config['training'].get('num_timesteps', 1000)
    optimizer = torch.optim.SGD(
        rob_model.parameters(),
        lr=float(config['training']['learning_rate']),
        momentum=0.9,
        weight_decay=float(config['training']['weight_decay']),
    )
    # Linear warmup for the first 5 epochs, then cosine anneal
    warmup_lr_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_lr_epochs:
            return (epoch + 1) / warmup_lr_epochs
        return 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_lr_epochs
    )
    epoch_losses = []
    val_losses = []

    pgd_cfg = config['training'].get('pgd', {})
    pgd_epsilon      = float(pgd_cfg.get('epsilon',      0.03))
    pgd_alpha        = float(pgd_cfg.get('alpha',        0.01))
    pgd_num_steps    = int(pgd_cfg.get('num_steps',      20))
    pgd_random_start = bool(pgd_cfg.get('random_start',  True))
    trades_beta      = float(config['training'].get('trades_beta', 6.0))

    # Define diffusion noise schedule (linear beta schedule)
    betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    start_epoch = 0
    best_val_acc = 0.0

    if resume is not None:
        checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/robust_classification', device)
        rob_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('warmup_scheduler_state_dict') is not None:
            warmup_scheduler.load_state_dict(checkpoint['warmup_scheduler_state_dict'])
        if checkpoint.get('cosine_scheduler_state_dict') is not None:
            cosine_scheduler.load_state_dict(checkpoint['cosine_scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"Resumed from checkpoint: epoch {start_epoch}, best val acc {best_val_acc:.1f}%")

    # train model
    for epoch in range(start_epoch, num_epochs):

        total_loss = 0.0
        rob_model.train()

        # Calculate current max timestep for the noise curriculum
        max_t = get_max_timestep(epoch, num_epochs, num_timesteps)

        in_warmup     = epoch < warmup_epochs
        in_transition = warmup_epochs <= epoch < warmup_epochs + transition_epochs

        # Progressive epsilon: ramp from 25% to 100% over the transition window.
        # Full epsilon only kicks in after the transition completes.
        if in_transition:
            eps_scale = (epoch - warmup_epochs) / transition_epochs  # 0 → 1
            current_epsilon = pgd_epsilon * (0.25 + 0.75 * eps_scale)
        else:
            current_epsilon = pgd_epsilon

        for batch_idx, batch in enumerate(trainloader):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            batch_size = inputs.shape[0]

            t = torch.randint(0, max(1, max_t), (batch_size,), device=device)
            x_t = get_noisy_image(inputs, t, alphas_cumprod)

            if in_warmup:
                # Noisy images with the curriculum schedule but no adversarial attack.
                x_train = x_t
                t_train = t

                optimizer.zero_grad()
                logits = rob_model(x_train, t_train)
                loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
                loss.backward()
                optimizer.step()
            else:
                # TRADES loss: CE on clean + β * KL(clean || adv)
                # Generate adversarial examples with full-strength PGD (no early stop)
                x_adv = pgd_attack_early_stop(
                    rob_model, x_t, t, labels,
                    epsilon=current_epsilon,
                    alpha=pgd_alpha,
                    num_steps=pgd_num_steps,
                    random_start=pgd_random_start,
                    clamp=(-1.0, 1.0),
                    training_mode=True,
                )
                t_train = t

                optimizer.zero_grad()
                logits_clean = rob_model(x_t, t)
                logits_adv   = rob_model(x_adv, t)
                loss_ce  = F.cross_entropy(logits_clean, labels, label_smoothing=label_smoothing)
                loss_kl  = F.kl_div(
                    F.log_softmax(logits_adv, dim=1),
                    F.softmax(logits_clean.detach(), dim=1),
                    reduction='batchmean',
                )
                loss = loss_ce + trades_beta * loss_kl
                loss.backward()
                optimizer.step()
                logits = logits_adv  # report adversarial accuracy during AT phase

            total_loss += loss.item()

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == labels).sum().item()
                wrong_mask = preds != labels
                wrong_indices = wrong_mask.nonzero(as_tuple=True)[0].tolist()
                wrong_preds  = preds[wrong_mask].tolist()
                wrong_labels = labels[wrong_mask].tolist()
                wrong_t      = t_train[wrong_mask].tolist()

            t_min, t_max = t_train.min().item(), t_train.max().item()
            status = f"  Batch {batch_idx:>3} | t=[{t_min},{t_max}] | loss={loss.item():.4f} | acc={correct}/{batch_size}"
            if wrong_indices:
                misses = ", ".join(
                    f"[{i}] pred={p} true={l} t={tv}"
                    for i, p, l, tv in zip(wrong_indices, wrong_preds, wrong_labels, wrong_t)
                )
                status += f" | MISCLASSIFIED: {misses}"
            print(status)

        # read LR before stepping so the summary reflects this epoch's LR
        current_lr = optimizer.param_groups[0]['lr']
        if epoch < warmup_lr_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        avg_loss = total_loss / len(trainloader)
        epoch_losses.append(avg_loss)

        # Validation — clean accuracy at t=0 on the held-out split
        rob_model.eval()
        val_loss_accum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_batch in valloader:
                val_inputs = val_batch[0].to(device)
                val_labels = val_batch[1].to(device)
                t_val = torch.zeros(val_inputs.size(0), dtype=torch.long, device=device)
                val_logits = rob_model(val_inputs, t_val)
                val_loss_accum += F.cross_entropy(val_logits, val_labels, label_smoothing=label_smoothing).item()
                val_correct += (val_logits.argmax(dim=1) == val_labels).sum().item()
                val_total += val_labels.size(0)
        avg_val_loss = val_loss_accum / len(valloader)
        val_acc = 100.0 * val_correct / val_total
        val_losses.append(avg_val_loss)

        # Adversarial validation every 5 epochs (after warmup) to track robustness
        adv_val_acc = None
        if not in_warmup and epoch % 5 == 0:
            adv_val_correct = 0
            adv_val_total = 0
            for val_batch in valloader:
                val_inputs = val_batch[0].to(device)
                val_labels = val_batch[1].to(device)
                t_val = torch.zeros(val_inputs.size(0), dtype=torch.long, device=device)
                val_adv = pgd_attack_early_stop(
                    rob_model, val_inputs, t_val, val_labels,
                    epsilon=pgd_epsilon,
                    alpha=pgd_alpha,
                    num_steps=10,
                    random_start=True,
                    clamp=(-1.0, 1.0),
                    training_mode=False,
                )
                with torch.no_grad():
                    adv_logits = rob_model(val_adv, t_val)
                adv_val_correct += (adv_logits.argmax(dim=1) == val_labels).sum().item()
                adv_val_total += val_labels.size(0)
            adv_val_acc = 100.0 * adv_val_correct / adv_val_total

        if in_warmup:
            phase = "warmup"
        elif in_transition:
            phase = f"transition(ε={current_epsilon:.3f})"
        else:
            phase = "adversarial"
        adv_str = f" | Adv Val Acc: {adv_val_acc:.1f}%" if adv_val_acc is not None else ""
        print(f'Epoch {epoch} [{phase}] | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.1f}%{adv_str} | LR: {current_lr:.2e}')

        # best-checkpoint criterion: adversarial accuracy when available, else clean accuracy
        best_metric = adv_val_acc if adv_val_acc is not None else val_acc

        # always save latest checkpoint for resuming
        ckpt_payload = {
            'epoch': epoch,
            'model_state_dict': rob_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'warmup_scheduler_state_dict': warmup_scheduler.state_dict(),
            'cosine_scheduler_state_dict': cosine_scheduler.state_dict(),
            'loss': loss,
            'config': config,
            'best_val_acc': best_val_acc,
        }
        if not checkpoint == None or not resume == None:
            save_checkpoint(ckpt_payload, f'{CHECKPOINT_DIR}/robust_classification')

        # save a separate best checkpoint based on adversarial val acc when available
        if best_metric > best_val_acc:
            best_val_acc = best_metric
            save_checkpoint(ckpt_payload, f'{CHECKPOINT_DIR}/robust_classification_best')
            print(f"  ** New best metric: {best_val_acc:.1f}% — saved to checkpoints/robust_classification_best")

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

    torch.save(
        {'model_state_dict': rob_model.state_dict(), 'config': config},
        os.path.join(result_directory, 'final_weights.pt')
    )
    return rob_model

def train_pid(config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=None):
    """Physics-informed diffusion: standard DDPM + symmetry + non-negativity losses.

    On top of the standard noise-prediction MSE, each training step:
      1. Converts the noise prediction back to image space via estimate_x0.
      2. Adds a symmetry penalty (H-flip + V-flip MSE) on that estimated image.
      3. Adds a non-negativity penalty (ReLU on negative pixels) on that estimated image.

    During sampling, the fully denoised image is clamped to >= 0 to remove
    unphysical negative-flux pixels before saving.
    """
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # reuse the same UNet architecture, scheduler, class embedding, and optimizer as train_diffusion
    unet, scheduler, class_emb, optimizer = build_diffusion_components(config, {}, device)

    # alphas_cumprod needed to convert noise predictions back to image space
    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    num_classes = config['data']['num_classes']
    num_epochs = config['training']['epochs']

    # physics loss weights from config
    lambda_sym = float(config['training'].get('lambda_sym', 0.1))

    start_epoch = 0
    loss_history = []
    val_loss_history = []
    mse_history = []        # MSE component only (train)
    sym_history = []        # weighted symmetry loss component (train)
    neg_history = []        # weighted non-negativity loss component (train)
    epochs_range = []
    fid_history = []
    kid_history = []
    fid_epochs = []

    # physics compliance measured on generated samples every 5 epochs
    compliance_epochs = []
    pct_negative_history = []   # % pixels < 0 in generated images
    sym_score_history = []      # raw symmetry MSE on generated images

    if resume is not None:
        checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/pid', device)

        if checkpoint.get('rng_state') is not None:
            rng_state = checkpoint['rng_state'].to('cpu').to(torch.uint8)
            torch.set_rng_state(rng_state)

        if checkpoint.get('cuda_rng_state') is not None:
            cuda_state = checkpoint['cuda_rng_state']
            if isinstance(cuda_state, torch.Tensor):
                torch.cuda.set_rng_state(cuda_state.to('cpu').to(torch.uint8))
            else:
                torch.cuda.set_rng_state_all([s.to('cpu').to(torch.uint8) for s in cuda_state])

        unet.load_state_dict(checkpoint['model_state_dict'])
        class_emb.load_state_dict(checkpoint['class_emb_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss_history = checkpoint['loss_history']
        val_loss_history = checkpoint['val_loss_history']
        mse_history = checkpoint.get('mse_history', [])
        sym_history = checkpoint.get('sym_history', [])
        neg_history = checkpoint.get('neg_history', [])
        epochs_range = checkpoint['epochs_range']
        fid_history = checkpoint['fid_history']
        kid_history = checkpoint.get('kid_history', [])
        fid_epochs = checkpoint.get('fid_epochs', [])
        compliance_epochs = checkpoint.get('compliance_epochs', [])
        pct_negative_history = checkpoint.get('pct_negative_history', [])
        sym_score_history = checkpoint.get('sym_score_history', [])
        print(f"Resumed from checkpoint (epoch {start_epoch})")

    if resume is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    for epoch in range(start_epoch, num_epochs):
        unet.train()
        epoch_loss = 0
        batch_count = 0

        print(f'Epoch {epoch}')

        epoch_loss = 0
        epoch_mse = 0
        epoch_sym = 0
        epoch_neg = 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # classifier-free guidance: randomly replace labels with null class
            drop_mask = torch.rand(labels.shape, device=device) < config['training']['label_dropout']
            training_labels = labels.clone()
            training_labels[drop_mask] = num_classes

            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, noise, t)

            class_embeddings = class_emb(training_labels).unsqueeze(1)  # (B, 1, D)
            noise_pred = unet(noisy_images, t, encoder_hidden_states=class_embeddings).sample

            # standard diffusion MSE on noise prediction
            mse = F.mse_loss(noise_pred, noise)

            # convert noise prediction to image space and apply physics penalties
            x_0_pred = estimate_x0(noisy_images, noise_pred, alphas_cumprod, t)
            sym = lambda_sym * symmetry_loss(x_0_pred)
            loss = mse + sym

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse  += mse.item()
            epoch_sym  += sym.item()
            epoch_neg  += neg.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        loss_history.append(avg_loss)
        mse_history.append(epoch_mse / batch_count)
        sym_history.append(epoch_sym / batch_count)
        neg_history.append(epoch_neg / batch_count)
        epochs_range.append(epoch)

        unet.eval()
        val_loss_accum = 0

        with torch.no_grad():
            for val_images, val_labels in testloader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                batch_sz = val_images.size(0)

                t_val = torch.linspace(0, scheduler.num_train_timesteps - 1, batch_sz, dtype=torch.long, device=device)
                noise_val = torch.randn_like(val_images)
                noisy_val = scheduler.add_noise(val_images, noise_val, t_val)

                class_emb_val = class_emb(val_labels).unsqueeze(1)
                noise_pred_val = unet(noisy_val, t_val, encoder_hidden_states=class_emb_val).sample

                # val loss includes physics terms so the metric is comparable to training loss
                x_0_val = estimate_x0(noisy_val, noise_pred_val, alphas_cumprod, t_val)
                v_loss = F.mse_loss(noise_pred_val, noise_val) + physics_loss(x_0_val, lambda_sym)
                val_loss_accum += v_loss.item()

            avg_val_loss = val_loss_accum / len(testloader)
            val_loss_history.append(avg_val_loss)
            print(
                f"Epoch {epoch} | Loss: {avg_loss:.6f} | Val: {avg_val_loss:.6f} "
                f"| MSE: {mse_history[-1]:.6f} | Sym: {sym_history[-1]:.6f} | Neg: {neg_history[-1]:.6f}"
            )

        # write all metrics to disk every epoch so the run can be inspected mid-training
        import json
        metrics = {
            'epochs': epochs_range,
            'loss': loss_history,
            'val_loss': val_loss_history,
            'mse': mse_history,
            'sym': sym_history,
            'neg': neg_history,
            'compliance_epochs': compliance_epochs,
            'pct_negative': pct_negative_history,
            'sym_score': sym_score_history,
            'fid_epochs': fid_epochs,
            'fid': fid_history,
            'kid': kid_history,
        }
        with open(os.path.join(result_directory, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        # save sample images every 5 epochs for visual progress checks
        if epoch % 5 == 0:
            unet.eval()
            with torch.no_grad():
                zero_images = sample_pid_zeros(unet, scheduler, class_emb, 4, num_classes, device)
                one_images  = sample_pid_ones(unet, scheduler, class_emb, 4, num_classes, device)

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                def prep_for_plot(img_tensor):
                    grid = torchvision.utils.make_grid(img_tensor, nrow=2, normalize=True, value_range=(-1, 1))
                    return grid.permute(1, 2, 0).cpu().numpy()

                axes[0].imshow(prep_for_plot(zero_images), cmap='gray')
                axes[0].set_title(f"Class 0 FR-I (Epoch {epoch})")
                axes[0].axis('off')

                axes[1].imshow(prep_for_plot(one_images), cmap='gray')
                axes[1].set_title(f"Class 1 FR-II (Epoch {epoch})")
                axes[1].axis('off')

                plt.tight_layout()
                plt.savefig(f"{result_directory}/comparison_epoch_{epoch}.png")
                plt.close()

                # measure physics compliance on the generated samples
                all_generated = torch.cat([zero_images, one_images], dim=0)
                pct_neg = (all_generated < 0).float().mean().item() * 100
                sym_score = symmetry_loss(all_generated).item()
                compliance_epochs.append(epoch)
                pct_negative_history.append(pct_neg)
                sym_score_history.append(sym_score)
                print(f"  Compliance — % negative pixels: {pct_neg:.2f}% | symmetry score: {sym_score:.6f}")

                # compute distributional quality metrics against the validation set
                print(f"Computing FID/KID at epoch {epoch}...")
                fid_score, kid_score = compute_fid_kid(
                    unet, scheduler, class_emb, num_classes, valloader, device,
                    sample_pid_zeros, sample_pid_ones,
                )
                fid_history.append(fid_score)
                kid_history.append(kid_score)
                fid_epochs.append(epoch)
                print(f"  FID: {fid_score:.4f} | KID: {kid_score:.6f}")

                if isinstance(dataset, MiraBestFITS):
                    fits_dir = os.path.join(result_directory, 'generated_fits')
                    os.makedirs(fits_dir, exist_ok=True)

                    for class_idx, imgs in [(0, zero_images), (1, one_images)]:
                        for i, img in enumerate(imgs):
                            norm_array = img.squeeze(0).cpu().numpy()
                            jy_array = dataset.denormalise(norm_array)
                            fname = os.path.join(fits_dir, f"generated_class{class_idx}_{i:03d}_{epoch}.fits")
                            MiraBestFITS.write_fits(jy_array, fname)

                    print(f"FITS files saved to {fits_dir}")

            unet.train()

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
                    'mse_history': mse_history,
                    'sym_history': sym_history,
                    'neg_history': neg_history,
                    'epochs_range': epochs_range,
                    'fid_history': fid_history,
                    'kid_history': kid_history,
                    'fid_epochs': fid_epochs,
                    'compliance_epochs': compliance_epochs,
                    'pct_negative_history': pct_negative_history,
                    'sym_score_history': sym_score_history,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                },
                f'{CHECKPOINT_DIR}/pid'
            )

    num_samples = config['data']['batch_size']
    class_0_images = sample_pid_zeros(unet, scheduler, class_emb, num_samples, num_classes, device)
    class_1_images = sample_pid_ones(unet, scheduler, class_emb, num_samples, num_classes, device)

    torchvision.utils.save_image(class_0_images, f"{result_directory}/generated_images_class_0.png", nrow=2, normalize=True, value_range=(-1, 1))
    torchvision.utils.save_image(class_1_images, f"{result_directory}/generated_images_class_1.png", nrow=2, normalize=True, value_range=(-1, 1))

    if isinstance(dataset, MiraBestFITS):
        fits_dir = os.path.join(result_directory, 'generated_fits')
        os.makedirs(fits_dir, exist_ok=True)

        for class_idx, imgs in [(0, class_0_images), (1, class_1_images)]:
            for i, img in enumerate(imgs):
                norm_array = img.squeeze(0).cpu().numpy()
                jy_array = dataset.denormalise(norm_array)
                fname = os.path.join(fits_dir, f"generated_class{class_idx}_{i:03d}.fits")
                MiraBestFITS.write_fits(jy_array, fname)

        print(f"FITS files saved to {fits_dir}")

    save_pid_training_plots(
        epochs_range, loss_history, val_loss_history,
        mse_history, sym_history, neg_history,
        compliance_epochs, pct_negative_history, sym_score_history,
        result_directory,
    )
    save_generative_metrics_plot(fid_epochs, fid_history, kid_history, result_directory)

    torch.save(
        {'model_state_dict': unet.state_dict(), 'class_emb_state_dict': class_emb.state_dict(), 'config': config},
        os.path.join(result_directory, 'final_weights.pt')
    )
    print("Generated images saved.")

    return unet

def train_robust_classifier_guided_diffusion(config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=None):
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    unet, scheduler, class_emb, optimizer = build_diffusion_components(config, {}, device)

    alphas_cumprod = scheduler.alphas_cumprod.to(device)

    num_classes = config['data']['num_classes']
    num_epochs = config['training']['epochs']

    start_epoch = 0

    loss_history = []
    val_loss_history = []
    epochs_range = []
    fid_history = []
    kid_history = []
    fid_epochs = []

    if resume is not None:
        checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/robust_classifier_guided_diffusion', device)

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
        kid_history = checkpoint.get('kid_history', [])
        fid_epochs = checkpoint.get('fid_epochs', [])
        print(f"Resumed from checkpoint: {resume} (epoch {start_epoch})")

    if resume is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # load pre-trained robust classifier; freeze it so only the UNet is updated,
    # but gradients still flow back through x_0_pred into the UNet
    classifier = TimeDependentResNet(num_classes, pretrained=False)
    cls_ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/robust_classification', device)
    classifier.load_state_dict(cls_ckpt['model_state_dict'])
    classifier.to(device)
    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad_(False)

    lambda_cls = float(config['training'].get('lambda_cls', 0.1))

    # --- Training loop ---
    for epoch in range(start_epoch, num_epochs):
        unet.train()
        epoch_loss = 0
        epoch_cls_loss = 0
        batch_count = 0

        print(f'Epoch {epoch}')
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # dropout labels to train on unclassified images (classifier-free guidance)
            drop_mask = torch.rand(labels.shape, device=device) < config['training']['label_dropout']
            training_labels = labels.clone()
            training_labels[drop_mask] = num_classes

            t = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device)
            noise = torch.randn_like(images)
            noisy_images = scheduler.add_noise(images, noise, t)

            # get class embeddings and add sequence dimension
            class_embeddings = class_emb(training_labels).unsqueeze(1)  # (B, 1, D)

            # predict noise conditioned on class
            model_output = unet(noisy_images, t, encoder_hidden_states=class_embeddings).sample

            # convert noise prediction to image space
            x_0_pred = estimate_x0(noisy_images, model_output, alphas_cumprod, t)

            # classifier guidance: penalise the UNet when x_0_pred is misclassified.
            # only applied to non-dropped samples (null-class has no target label).
            # t=0 signals to the robust classifier that x_0_pred is an estimated clean image.
            cls_mask = training_labels != num_classes
            cls_loss = torch.tensor(0.0, device=device)
            if cls_mask.any():
                t_clean = torch.zeros(cls_mask.sum(), dtype=torch.long, device=device)
                cls_input = x_0_pred[cls_mask]
                # FITS images are log-SNR normalised; convert to linear [-1,1] so the
                # classifier (trained on standard MiraBest images) sees the same distribution.
                if isinstance(dataset, MiraBestFITS):
                    cls_input = fits_to_linear(cls_input, dataset)
                cls_logits = classifier(cls_input, t_clean)
                cls_loss = lambda_cls * F.cross_entropy(cls_logits, labels[cls_mask])

            loss = F.mse_loss(model_output, noise) + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        avg_cls_loss = epoch_cls_loss / batch_count
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
                t_val = torch.linspace(0, scheduler.num_train_timesteps - 1, batch_sz, dtype=torch.long, device=device)
                noise_val = torch.randn_like(val_images)
                noisy_val = scheduler.add_noise(val_images, noise_val, t_val)

                class_emb_val = class_emb(val_labels).unsqueeze(1)
                model_output = unet(noisy_val, t_val, encoder_hidden_states=class_emb_val).sample

                v_loss = F.mse_loss(model_output, noise_val)
                val_loss_accum += v_loss.item()

            # --- Finalize Metrics for the Epoch ---
            avg_val_loss = val_loss_accum / len(testloader)

            print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Cls Loss: {avg_cls_loss:.6f}")
            
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
                    'kid_history': kid_history,
                    'fid_epochs': fid_epochs,
                    'rng_state': torch.get_rng_state(),
                    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                },
                f'{CHECKPOINT_DIR}/robust_classifier_guided_diffusion'
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

def train_model(model, config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=None):
    print(f"Training {model} for {config['training']['epochs']} epochs")
    if model == 'classification':
        return train_classification(config, trainloader, valloader, device, result_directory, resume, checkpoint)
    elif model == 'robust_classification':
        return train_robust_classification(config, trainloader, valloader, device, result_directory, resume, checkpoint)
    elif model == 'diffusion':
        return train_diffusion(config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=dataset)
    elif model == 'pid':
        return train_pid(config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=dataset)
    elif model == 'robust_classifier_guided_diffusion':
        return train_robust_classifier_guided_diffusion(config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=dataset)
    elif model == 'classifier_guided_diffusion':
        return train_classifier_guided_diffusion(config, trainloader, valloader, testloader, device, result_directory, resume, checkpoint, dataset=dataset)
    else:
        raise ValueError(f'Model {model} not supported ["diffusion", "pid", "robust_classification", "classification", "robust_classifier_guided_diffusion", "classifier_guided_diffusion"]')
