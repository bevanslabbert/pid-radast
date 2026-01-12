import torch
import datetime
from torchvision.models import resnet50
import torchvision
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# evaluate model performance
def test_model(model_type, model, config, testloader, device, result_directory):
    print(f'Testing model')
    if model_type == 'robust_classification' or model_type == 'classification':
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
