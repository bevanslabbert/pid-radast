from src.utils.data import get_data_loaders
from src.models.time_dependent_resnet import TimeDependentResNet
from src.utils.augmentation import pgd_attack_early_stop, get_max_timestep, get_noisy_image
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet18
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import UNet2DConditionModel, DDPMScheduler
import matplotlib.pyplot as plt
import os
 
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

def train_classification(config, trainloader, valloader, device, result_directory, resume):

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

    if resume is not None and os.path.isfile(resume):
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
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
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f'checkpoints/classifier.pth')

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

def train_diffusion(config, trainloader,  device, result_directory, resume):
    # --- UNet that supports class conditioning ---
    unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        cross_attention_dim=128,   # needed for conditioning
    ).to(device)

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    # Embed class labels
    num_classes = 2
    num_epochs = 5
    class_emb = nn.Embedding(num_classes, 128).to(device)

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-5)

    # --- Training loop ---
    for epoch in range(num_epochs):
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

        print(f"Epoch {epoch+1}, loss={loss.item():.4f}")

    unet.eval()
    with torch.no_grad():
        label = torch.tensor([0] * 8, device=device)  # generate "class 3"
        class_embeddings = class_emb(label)

        scheduler.set_timesteps(50)
        noisy = torch.randn(8, 3, 32, 32, device=device)

        for t in scheduler.timesteps:
            noise_pred = unet(noisy, t, encoder_hidden_states=class_embeddings).sample
            noisy = scheduler.step(noise_pred, t, noisy).prev_sample

        # noisy now contains generated images

    return unet

def train_robust_classifier(config, trainloader, device, result_directory, resume):
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

    if resume is not None and os.path.isfile(resume):
        checkpoint = torch.load(resume, map_location=device)
        rob_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from checkpoint: {resume} (epoch {start_epoch})")

    # train model
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}")

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
        torch.save({
            'epoch': epoch,
            'model_state_dict': rob_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, f'checkpoints/robust_classifier.pth')

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

    return rob_model

def train_model(model, config, trainloader, valloader, device, result_directory, resume):
    print(f"🚀 Training {model} for {config['training']['epochs']} epochs")
    if model == 'classifier':
        return train_classification(config, trainloader, valloader, device, result_directory, resume)
    elif model == 'robust_classifier':
        return train_robust_classifier(config, trainloader, device, result_directory, resume)
    elif model == 'diffuser':
        return train_diffusion(config, trainloader, valloader, device, result_directory, resume)
