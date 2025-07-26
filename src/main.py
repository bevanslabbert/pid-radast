import os
import sys
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
        'lr': tune.loguniform(1e-4, 1e-1),
        'optimizer': tune.choice([torch.optim.AdamW, torch.optim.Adam]),
        'model_class': ClassificationModel,
        'criterion_class': nn.CrossEntropyLoss,
        'dataset': 'MiraBest',
        'batch_size': tune.choice([2, 4, 8, 16, 32])
    }

    results = optimize_parameters(config)

    results = get_best_config(ClassificationModel)
    print(results)
    print(results.get_best_result().config)


if __name__ == "__main__":
    main()
