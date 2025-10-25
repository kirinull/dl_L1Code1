import torch.nn as nn
from data_module import load_cifar10, create_dataloaders
from model_module import create_model, count_parameters
from optimizer_module import get_optimizer
from train_module import create_trainer
from utils import set_seed, get_device

def run_basic_training():
    set_seed(42)
    device = get_device()

    train_data, test_data = load_cifar10()
    train_loader, test_loader = create_dataloaders(train_data, test_data)

    model = create_model('base')
    params = count_parameters(model)
    print(f"Model parameters: {params:,}")

    optimizer = get_optimizer(model, 'SGD', learning_rate=0.01)
    criterion = nn.CrossEntropyLoss()

    trainer = create_trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        model_name="base_model"
    )

    accuracy = trainer.train(
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=30
    )

    return accuracy


if __name__ == "__main__":
    run_basic_training()