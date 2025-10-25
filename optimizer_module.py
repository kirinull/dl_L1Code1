import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR, LinearLR, StepLR


def get_optimizer(model, optimizer_name, learning_rate, **kwargs):
    if optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate)
#模块2内容
    elif optimizer_name == 'SGD_Momentum':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif optimizer_name == 'SGD_Nesterov':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    elif optimizer_name == 'AdaGrad':
        return optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSProp':
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def get_scheduler(optimizer, scheduler_name, total_epochs, **kwargs):
    if scheduler_name is None or scheduler_name == 'none':
        return None

    if scheduler_name == 'linear':
        def lr_lambda(epoch):
            return 1.0 - epoch / total_epochs

        return LambdaLR(optimizer, lr_lambda)

    elif scheduler_name == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=total_epochs, **kwargs)

    elif scheduler_name == 'warmup_cosine':
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                            milestones=[warmup_epochs])

    elif scheduler_name == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")



OPTIMIZER_CONFIGS = {
    'sgd': {'optimizer_name': 'SGD', 'learning_rate': 0.01},
    #模块2内容
    'sgd_momentum': {'optimizer_name': 'SGD_Momentum', 'learning_rate': 0.01},
    'adam': {'optimizer_name': 'Adam', 'learning_rate': 0.001},
    'adamw': {'optimizer_name': 'AdamW', 'learning_rate': 0.001},
    'adagrad': {'optimizer_name': 'AdaGrad', 'learning_rate': 0.01},
    'rmsprop': {'optimizer_name': 'RMSProp', 'learning_rate': 0.001},
}