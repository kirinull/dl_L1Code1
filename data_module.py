import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


# CIFAR-10类别名称
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def load_cifar10(data_path='./cifar-10-python'):
    train_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    test_data = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )

    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    return train_data, test_data


def create_dataloaders(train_data, test_data, batch_size=64):
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


def visualize_dataset(dataset, num_images=16, title="CIFAR-10 Images"):
    """
    可视化数据集中的图片
    """
    # 随机选择图片
    indices = np.random.choice(len(dataset), num_images, replace=False)

    # 创建子图
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    axes = axes.ravel()

    for i, idx in enumerate(indices):
        image, label = dataset[idx]

        # 将tensor转换为numpy并调整维度顺序 (C, H, W) -> (H, W, C)
        image = image.numpy().transpose((1, 2, 0))

        # 显示图片
        axes[i].imshow(image)
        axes[i].set_title(f'{classes[label]}')
        axes[i].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# 测试代码
if __name__ == "__main__":
    # 加载数据
    train_data, test_data = load_cifar10()

    # 可视化训练集
    print("训练集样本示例:")
    visualize_dataset(train_data, title="CIFAR-10 Training Images")

    # 可视化测试集
    print("测试集样本示例:")
    visualize_dataset(test_data, title="CIFAR-10 Test Images")