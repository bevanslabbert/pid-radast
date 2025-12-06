import torch
from torchvision.models import resnet50
import torchvision
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def test_model(model_type, config, testloader, device, resume):
    if model_type == 'classifier':
        evaluate_model_performance(model_type, config, testloader, device, resume)

# evaluate model performance
def evaluate_model_performance(model_type, config, testloader, device):
    if model_type == 'classifier':
        # model definition
        model = resnet50(pretrained=True)

        num_classes = config['data']['num_classes']

        # replace last layer to match number of classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        checkpoint = torch.load('checkpoints/classifier.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

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
            plt.show()

