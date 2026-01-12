import torch
from src.utils.checkpoint import load_checkpoint
from torchvision.models import resnet50
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torchvision
from diffusers import UNet2DConditionModel, DDPMScheduler

# evaluate model performance
def test_model(model_type, model, config, testloader, device, result_directory):
    print(f'Testing model')
    if model_type == 'robust_classification' or model_type == 'classification':
        # TODO: Test this - added last
        model = resnet50(pretrained=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])

        checkpoint = load_checkpoint(f'checkpoints/{model_type}', device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        num_classes = config['data']['num_classes']

        # replace last layer to match number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        checkpoint = torch.load('checkpoints/classifier.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)
        model.eval()

        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            accuracy = 100 * correct / total
            print(f'Accuracy: {accuracy}%')

            # Compute confusion matrix
            cm = confusion_matrix(all_labels, all_preds)

            # Display confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues', xticks_rotation=45)
            plt.title("Confusion Matrix")
            plt.savefig(f'{result_directory}/confusion_matrix.png')

    elif model_type == "diffusion":
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        num_classes = config['data']['num_classes']
        class_emb = nn.Embedding(num_classes, 128).to(device)

        # --- UNet that supports class conditioning ---
        unet = UNet2DConditionModel(
            sample_size=32,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=(64, 64, 128, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
            cross_attention_dim=128,   # needed for conditioning
        ).to(device)

        unet.eval()
        with torch.no_grad():
            target_class = 1
            label = torch.tensor([target_class] * 8, device=device)  # generate target class
            class_embeddings = class_emb(label).unsqueeze(1)

            scheduler.set_timesteps(50)
            noisy = torch.randn(8, 3, 224, 224, device=device)

            for t in scheduler.timesteps:
                noise_pred = unet(noisy, t, encoder_hidden_states=class_embeddings).sample
                noisy = scheduler.step(noise_pred, t, noisy).prev_sample

        torchvision.utils.save_image(
            noisy, 
            f"{result_directory}/generated_class_{target_class}.png", 
            nrow=2, 
            normalize=True, 
            value_range=(-1, 1)
        )

        print(f"✅ Generated images for class {target_class} saved to PNG.")
