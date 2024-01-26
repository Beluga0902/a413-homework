# https://github.com/Beluga0902/a413-homework.git
import torch
import torch.nn as nn
# 定义简单的卷积神经网络模型
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc1 = nn.Linear(16 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # 前向传播过程
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)  # 展平特征图
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
# 创建模型实例
num_classes = 10  # 分类数目
model = SimpleCNN(num_classes)
# 打印模型结构
print(model)
