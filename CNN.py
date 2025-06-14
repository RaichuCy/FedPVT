####################################################################################################
# 功能：CNN3模型
# 作者：洪建华
# 版本：创建——20250123_1555
#       编写ResNet模型代码——20250123_1603
####################################################################################################
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        # 第一层卷积：1个输入通道（灰度图像），32个3x3的卷积核
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二层卷积：32个输入通道，64个3x3的卷积核
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        # 第三层卷积：64个输入通道，128个3x3的卷积核
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 前向传播
        x = self.pool(torch.relu(self.conv1(x)))  # 第一层卷积 + 激活 + 池化
        x = self.pool(torch.relu(self.conv2(x)))  # 第二层卷积 + 激活 + 池化
        x = self.pool(torch.relu(self.conv3(x)))  # 第三层卷积 + 激活 + 池化
        x = x.view(-1, 128 * 3 * 3)  # 扁平化
        x = torch.relu(self.fc1(x))  # 全连接层1 + 激活
        x = self.fc2(x)  # 输出层
        return x

def CNN3():
    return CNN()


####################################################################################################
####################################################################################################