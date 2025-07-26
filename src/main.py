import sys
import os
from ray import tune
from torch import nn
import torch
from parameter_optimization.optimize import optimize_parameters, get_best_config
from models.classification_model import ClassificationModel

project_root = "/Users/bevanslabbert/Documents/GitHub/pid-radast"
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main():
    config = {
        'lr': tune.loguniform(1e-3, 1e-1),
        'optimizer_class': tune.choice([torch.optim.AdamW, torch.optim.Adam]),
        'model_class': ClassificationModel,
        'criterion_class': nn.CrossEntropyLoss,
        'dataset': 'MiraBest',
        'batch_size': tune.choice([8, 16])
    }

    best_result = optimize_parameters(config)

    # best_result = get_best_config(ClassificationModel, os.getcwd())

    print(best_result)


if __name__ == "__main__":
    main()
