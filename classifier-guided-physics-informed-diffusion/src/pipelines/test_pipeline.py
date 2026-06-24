import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision.models import resnet50
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from src.utils.checkpoint import load_checkpoint
from src.models.time_dependent_resnet import TimeDependentResNet
from src.utils.augmentation import pgd_attack_early_stop, get_noisy_image

CHECKPOINT_DIR = 'checkpoints'

def _eval_accuracy(model, loader, device, t_fixed=None, alphas_cumprod=None):
    """Return (accuracy, all_preds, all_labels) for a single noise level."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if t_fixed is not None:
                t = torch.full((inputs.size(0),), t_fixed, dtype=torch.long, device=device)
                inputs = get_noisy_image(inputs, t, alphas_cumprod)
            else:
                t = torch.zeros(inputs.size(0), dtype=torch.long, device=device)

            outputs = model(inputs, t)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return 100.0 * correct / total, all_preds, all_labels

# evaluate model performance
def test_model(model_type, config, testloader, device, result_directory, model=None):
    print(f'Testing model')
    if model_type == 'robust_classification':
        num_classes = config['data']['num_classes']
        num_timesteps = config['training'].get('num_timesteps', 1000)
        pgd_cfg = config['training'].get('pgd', {})
        pgd_epsilon   = float(pgd_cfg.get('epsilon',    0.03))
        pgd_alpha     = float(pgd_cfg.get('alpha',      0.01))
        pgd_num_steps = int(pgd_cfg.get('num_steps',    20))

        rob_model = TimeDependentResNet(num_classes, pretrained=False)
        checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/robust_classification', device)
        rob_model.load_state_dict(checkpoint['model_state_dict'])
        rob_model.to(device)
        rob_model.eval()

        # noise schedule matching training
        betas = torch.linspace(0.0001, 0.02, num_timesteps).to(device)
        alphas_cumprod = torch.cumprod(1 - betas, dim=0)

        # --- 1. Clean accuracy (t=0) ---
        # This is the most important metric: x_0_pred passed to the classifier
        # during diffusion training always uses t=0.
        clean_acc, clean_preds, clean_labels = _eval_accuracy(rob_model, testloader, device)
        clean_report = classification_report(clean_labels, clean_preds,
                                             target_names=['FR-I', 'FR-II'], digits=3)
        print(f"\nClean accuracy (t=0): {clean_acc:.2f}%")
        print(clean_report)

        cm = confusion_matrix(clean_labels, clean_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FR-I', 'FR-II'])
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix — clean (t=0)  acc={clean_acc:.1f}%")
        plt.savefig(f'{result_directory}/confusion_matrix_clean.png', bbox_inches='tight')
        plt.close()

        # --- 2. Robustness curve: accuracy at increasing noise levels ---
        noise_levels = [0, 100, 200, 300, 500, 700, 900]
        noisy_accs = []
        for t_val in noise_levels:
            acc, _, _ = _eval_accuracy(rob_model, testloader, device,
                                       t_fixed=t_val, alphas_cumprod=alphas_cumprod)
            noisy_accs.append(acc)
            print(f"  t={t_val:4d}  acc={acc:.2f}%")

        plt.figure(figsize=(8, 5))
        plt.plot(noise_levels, noisy_accs, marker='o', linewidth=2, color='tab:blue')
        plt.axhline(50, color='gray', linestyle='--', linewidth=1, label='random chance')
        plt.xlabel('Noise level (timestep t)')
        plt.ylabel('Accuracy (%)')
        plt.title('Robust Classifier: Accuracy vs Noise Level')
        plt.ylim(0, 105)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f'{result_directory}/robustness_curve.png', bbox_inches='tight', dpi=150)
        plt.close()

        # --- 3. Adversarial accuracy at t=0 (PGD attack on clean images) ---
        # Tests how much the adversarial training actually helped.
        adv_correct = 0
        adv_total = 0
        adv_preds = []
        adv_labels_all = []

        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            t = torch.zeros(inputs.size(0), dtype=torch.long, device=device)

            x_adv = pgd_attack_early_stop(
                rob_model, inputs, t, labels,
                epsilon=pgd_epsilon, alpha=pgd_alpha,
                num_steps=pgd_num_steps, random_start=True,
                clamp=(-1.0, 1.0),
            )

            rob_model.eval()
            with torch.no_grad():
                logits = rob_model(x_adv, t)
                predicted = logits.argmax(dim=1)
                adv_correct += (predicted == labels).sum().item()
                adv_total += labels.size(0)
                adv_preds.extend(predicted.cpu().numpy())
                adv_labels_all.extend(labels.cpu().numpy())

        adv_acc = 100.0 * adv_correct / adv_total
        adv_report = classification_report(adv_labels_all, adv_preds,
                                           target_names=['FR-I', 'FR-II'], digits=3)
        print(f"\nAdversarial accuracy (PGD ε={pgd_epsilon}, steps={pgd_num_steps}): {adv_acc:.2f}%")
        print(adv_report)

        cm_adv = confusion_matrix(adv_labels_all, adv_preds)
        disp_adv = ConfusionMatrixDisplay(confusion_matrix=cm_adv, display_labels=['FR-I', 'FR-II'])
        disp_adv.plot(cmap='Oranges')
        plt.title(f"Confusion Matrix — adversarial (PGD)  acc={adv_acc:.1f}%")
        plt.savefig(f'{result_directory}/confusion_matrix_adversarial.png', bbox_inches='tight')
        plt.close()

        report_path = f'{result_directory}/classification_report.txt'
        with open(report_path, 'w') as f:
            f.write(f"Clean accuracy (t=0): {clean_acc:.2f}%\n\n")
            f.write("--- Clean Classification Report ---\n")
            f.write(clean_report)
            f.write(f"\nRobustness curve (accuracy vs noise level):\n")
            for t_val, acc in zip(noise_levels, noisy_accs):
                f.write(f"  t={t_val:4d}  acc={acc:.2f}%\n")
            f.write(f"\nAdversarial accuracy (PGD ε={pgd_epsilon}, steps={pgd_num_steps}): {adv_acc:.2f}%\n\n")
            f.write("--- Adversarial Classification Report ---\n")
            f.write(adv_report)

        print(f"\nSummary saved to {result_directory}/")

    elif model_type == 'classification':
        num_classes = config['data']['num_classes']
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        checkpoint = load_checkpoint(f'{CHECKPOINT_DIR}/classification', device)
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
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            accuracy = 100 * correct / total
            print(f'Accuracy: {accuracy}%')

            cm = confusion_matrix(all_labels, all_preds)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(cmap='Blues', xticks_rotation=45)
            plt.title("Confusion Matrix")
            plt.savefig(f'{result_directory}/confusion_matrix.png')

    elif model_type in ('diffusion', 'pid', 'classifier_guided_diffusion', 'robust_classifier_guided_diffusion'):
        from src.models.diffusion import build_diffusion_components
        from src.models.pid import sample_pid_zeros, sample_pid_ones

        num_classes = config['data']['num_classes']
        unet, scheduler, class_emb, _ = build_diffusion_components(config, {}, device)

        ckpt_dir = {
            'diffusion': 'diffusion',
            'pid': 'pid',
            'classifier_guided_diffusion': 'classifier_guided_diffusion',
            'robust_classifier_guided_diffusion': 'robust_classifier_guided_diffusion',
        }[model_type]

        ckpt = load_checkpoint(f'{CHECKPOINT_DIR}/{ckpt_dir}', device)
        unet.load_state_dict(ckpt['model_state_dict'])
        class_emb.load_state_dict(ckpt['class_emb_state_dict'])
        unet.to(device)
        unet.eval()

        with torch.no_grad():
            zero_images = sample_pid_zeros(unet, scheduler, class_emb, 4, num_classes, device)
            one_images  = sample_pid_ones(unet, scheduler, class_emb, 4, num_classes, device)

        torchvision.utils.save_image(zero_images, f"{result_directory}/test_generated_class_0.png", nrow=2, normalize=True, value_range=(-1, 1))
        torchvision.utils.save_image(one_images,  f"{result_directory}/test_generated_class_1.png", nrow=2, normalize=True, value_range=(-1, 1))
        print(f"Generated test images saved to {result_directory}/")
