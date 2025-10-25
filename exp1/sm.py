import torch  # 命令行是逐行立即执行的
content = torch.load('best.pth')
print(content.keys())
print(content['model'])
