import torch.nn as nn


class BaseLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(BaseLeNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.model(x)


#----------------------模块2内容-------------------------------
class EnhancedLeNet(nn.Module):
    def __init__(self, num_classes=10, use_bn=True):
        super(EnhancedLeNet, self).__init__()

        layers = [
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.BatchNorm2d(32) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 32, 5, 1, 2),
            nn.BatchNorm2d(32) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 5, 1, 2),
            nn.BatchNorm2d(64) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.BatchNorm1d(64) if use_bn else nn.Identity(),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
#----------------------模块2内容-------------------------------

def create_model(model_type='base', **kwargs):
    if model_type == 'base':
        return BaseLeNet(**kwargs)
    # 模块2内容
    elif model_type == 'enhanced':
        return EnhancedLeNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)