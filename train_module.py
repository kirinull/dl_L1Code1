import torch
import time
import os
import glob


class Trainer:
    def __init__(self, model, optimizer, criterion, device, model_name="model"):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model_name = model_name
        self.exp_folder = self._get_next_exp_folder()

        self.model = self.model.to(self.device)

    def _get_next_exp_folder(self):
        """获取下一个实验文件夹"""
        existing = glob.glob("exp*")
        exp_num = len(existing) + 1
        folder = f"exp{exp_num}"
        os.makedirs(folder, exist_ok=True)
        return folder

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        return 100. * correct / total

    def train(self, train_loader, test_loader, num_epochs=30):
        print(f"Training: {self.model_name}")
        print(f"Experiment: {self.exp_folder}")
        print(f"Epochs: {num_epochs}, Device: {self.device}")

        start_time = time.time()
        best_accuracy = 0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            accuracy = self.evaluate(test_loader)

            # 保存每轮模型
            filepath = f"{self.exp_folder}/epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'accuracy': accuracy,
            }, filepath)

            # 更新最佳模型
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(self.model.state_dict(), f"{self.exp_folder}/best.pth")

            print(f'Epoch {epoch + 1:2d}/{num_epochs} | '
                  f'Loss: {train_loss:.4f} | '
                  f'Accuracy: {accuracy:6.2f}% | '
                  f'Best: {best_accuracy:6.2f}%')

        training_time = time.time() - start_time

        print(f"Training completed: {accuracy:.2f}% final, {best_accuracy:.2f}% best")
        print(f"Training time: {training_time:.2f}s")
        print(f"All models saved to: {self.exp_folder}")

        return accuracy


def create_trainer(model, optimizer, criterion, device, **kwargs):
    return Trainer(model, optimizer, criterion, device, **kwargs)


def load_model(model, optimizer, exp_num, epoch):
    """从实验文件夹加载模型"""
    path = f"exp{exp_num}/epoch_{epoch + 1}.pth"
    checkpoint = torch.load(path)

    model.load_state_dict(checkpoint['model'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])

    print(f"模型已加载: {path}")
    return model, optimizer