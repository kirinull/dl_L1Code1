import torch
from PIL import Image
import torchvision.transforms as transforms
from model_module import create_model

# CIFAR-10类别
classes = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']


def infer(image_path, exp_num, epoch=30):
    """简单推理函数"""
    # 加载模型
    model = create_model('base')
    model.load_state_dict(torch.load(f"exp{exp_num}/epoch_{epoch}.pth")['model'])
    model.eval()

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0)

    # 预测
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    print(f"预测: {classes[predicted.item()]}")
    return classes[predicted.item()]


# 使用
if __name__ == "__main__":
    infer("data/img_2.png", exp_num=2)