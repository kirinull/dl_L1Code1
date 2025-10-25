import torch
import numpy as np


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """获取可用设备（优先GPU）"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def run_basic_training():
    set_seed(42)
    device = get_device()

    train_data, test_data = load_cifar10()
    train_loader, test_loader = create_dataloaders(train_data, test_data)

    model = create_model('base')
    # 将模型移动到GPU
    model = model.to(device)
    
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