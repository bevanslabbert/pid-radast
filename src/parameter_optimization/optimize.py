import torch
def optimize_parameters(config, model, checkpoint, data_dir, optimizer, criterion, dataset):
    train(config, model, checkpoint, data_dir, optimizer, criterion, dataset)
    test(model)

def train(config, model, checkpoint, data_dir, optimizer, criterion, dataset):
    set_gpu_parallel_training() # attempt to set calculation to parallel gpu training

    optimizer = optimizer(model.parameters(), lr=config["lr"], momentum=0.9) # initialize optimizer function from optimizer class

    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            start_epoch = checkpoint_state["epoch"]
            net.load_state_dict(checkpoint_state["net_state_dict"])
            optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])

    else:
        start_epoch = 0

    trainset, testset = get_data(dataset)

    # split dataset into training and validation sets
    test_abs = int(len(trainloader))




def test():
    pass


# attempt to set calculation to GPU to enhance performace for image optimization
def set_gpu_parallel_training(model):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)
