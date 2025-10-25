import matplotlib.pyplot as plt
import numpy as np
import torchvision


# CIFAR-10类别名称
classes = ('airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# 注意：如果之前下载过数据集，建议将download=True改为False
train_data = torchvision.datasets.CIFAR10(
    root='../L1Code1/cifar-10-python',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_data = torchvision.datasets.CIFAR10(
    root='../L1Code1/cifar-10-python',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 可视化CIFAR-10数据集中的图片
def show_cifar10_images(dataset, num_images=16):
    """
    显示CIFAR-10数据集中的图片
    """
    # 随机选择一些图片
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
        axes[i].set_title(f'{classes[label]} (Label: {label})')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# 显示训练集中的一些图片
print("显示训练集中的图片:")
show_cifar10_images(train_data)

# 显示测试集中的一些图片
print("显示测试集中的图片:")
show_cifar10_images(test_data)
