import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10("dataset", True, torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("dataset", False, torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("train数据集的长度为：{}".format(train_data_size))
print("test数据集的长度为：{}".format(test_data_size))

# 定义训练使用的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 利用DataLoader加载数据集
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# 搭建神经网络
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

# 创建网络模型
classifier = Classifier()
classifier.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 1、记录训练的次数
total_train_step = 0
# 2、记录测试的次数
total_test_step = 0
# 3、训练的轮数
epoch = 30

# 使用tensorboard展示数据
writer = SummaryWriter("logs")

start_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤开始
    classifier.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = classifier(imgs)
        loss = loss_fn(output, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练时间：{}".format(end_time - start_time))
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    classifier.eval()
    # 计算test集loss
    total_test_loss = 0
    # 计算正确率
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = classifier(imgs)
            loss = loss_fn(output, targets)
            total_test_loss += loss.item()
            # argmax(1)表示在一行中找最大，0表示在列中
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1

    # 将每一轮的模型都保存下来
    # torch.save(classifier.state_dict(), "./models/gpu_train_model{}.pth".format(i + 1))
    torch.save(classifier, "./models/gpu_train_model{}.pth".format(i + 1))
    print("模型已保存")

writer.close()
