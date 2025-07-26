import torch
import torch.nn as nn
from torch.utils.data import random_split
from ray import tune
from ray.tune import Tuner
from data import get_data
import os
from models.classification_model import ClassificationModel

__all__ = ['optimize_parameters', 'get_best_config']


def optimize_parameters(config):
    print("Starting optimization")
    tuner = Tuner(
        train_,
        param_space=config,
        tune_config=tune.TuneConfig(
            metric='val_loss',
            mode='min',
            num_samples=10,
        ),
        run_config=tune.RunConfig(
            name=config["model_class"].__name__,
            storage_path=os.path.join(
                os.getcwd(), "tuning_results")
        ),
    )
    results = tuner.fit()
    return results


def train_(config):

    print("Training...")

    # extract inputs from config
    model_class = config["model_class"]
    model = model_class()

    criterion_class = config["criterion_class"]
    dataset = config["dataset"]

    criterion = criterion_class()

    # attempt to set calculation to parallel gpu training
    device, model = set_gpu_parallel_training(model)

    current_optimizer = config["optimizer"]
    current_lr = config["lr"]
    print(f'Using optimizer {current_optimizer}')
    print(f'Using lr {current_lr}')

    optimizer_class = current_optimizer

    # initialize optimizer function from optimizer class
    optimizer = optimizer_class(model.parameters(), lr=config["lr"])

    start_epoch = 0

    trainset, testset = get_data(dataset)

    # split dataset into training and validation sets
    test_abs = int(len(trainset) * 0.8)
    train_subset, val_subset = random_split(
        trainset,
        [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config['batch_size']), shuffle=True, num_workers=8
    )

    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config['batch_size']), shuffle=True, num_workers=8
    )

    model.train()  # set to training mode

    training_loss = 0.0  # loss for reporting

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times
        print(f'Epoch {epoch} training...')
        running_loss = 0.0
        epoch_steps = 0
        for batch in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1

        training_loss = running_loss / len(trainloader)

    model.eval()  # set to evaluation mode

    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in valloader:
            outputs = model(inputs)  # evaluate inputs
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        tune.report({
            "loss": training_loss,
            "val_loss": val_loss/len(valloader)
        })


def get_best_config(model_class):

    experimental_results_path = os.path.join(
        os.getcwd(), "tuning_results", model_class.__name__)

    if not os.path.exists(experimental_results_path):
        raise FileExistsError(f"Parameter optimization not run for {
                              model_class.__name__}")

    restored_tuner = tune.Tuner.restore(
        experimental_results_path, trainable=train_)

    result_grid = restored_tuner.get_results()

    print(result_grid)
    print(f"Best trial config: {result_grid.get_best_result()}")

    return result_grid.get_best_result()


def set_gpu_parallel_training(model):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    model.to(device)

    return device, model
