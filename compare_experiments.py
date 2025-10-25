import sys
import os
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
import torch
import torch.nn as nn
from data_module import load_cifar10, create_dataloaders
from model_module import create_model, count_parameters
from optimizer_module import get_optimizer, OPTIMIZER_CONFIGS
from train_module import create_trainer
from utils import set_seed, get_device


def run_comparison():
    set_seed(42)
    device = get_device()

    train_data, test_data = load_cifar10()
    train_loader, test_loader = create_dataloaders(train_data, test_data)

    experiments = [
        {'name': 'SGD', 'model_type': 'base', 'optimizer': 'SGD'},
        {'name': 'SGD_Momentum', 'model_type': 'base', 'optimizer': 'SGD_Momentum'},
        {'name': 'Adam', 'model_type': 'base', 'optimizer': 'Adam'},
        {'name': 'BN_Adam', 'model_type': 'enhanced', 'optimizer': 'Adam'},
    ]

    results = {}

    for exp_config in experiments:
        print(f"\nTraining: {exp_config['name']}")
        print("-" * 40)

        model = create_model(exp_config['model_type'])
        params = count_parameters(model)
        print(f"Model parameters: {params:,}")

        optimizer_config = OPTIMIZER_CONFIGS[exp_config['optimizer'].lower()]
        optimizer = get_optimizer(model, **optimizer_config)
        criterion = nn.CrossEntropyLoss()

        trainer = create_trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            model_name=exp_config['name']
        )

        accuracy = trainer.train(
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=30
        )

        results[exp_config['name']] = accuracy

    print("\nResults:")
    for name, accuracy in results.items():
        print(f"{name}: {accuracy:.2f}%")

    best_name = max(results, key=results.get)
    print(f"Best: {best_name} - {results[best_name]:.2f}%")


if __name__ == "__main__":
    run_comparison()
