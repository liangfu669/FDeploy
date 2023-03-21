import torch.nn.functional as F
from torch import nn
from torchvision import models
import torch


class MYnet(nn.Module):
    def __init__(self):
        super(MYnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1,
                               padding=1)  # the size does not change after convolutional layer
        self.pool = nn.MaxPool2d(2, 2)  # 2*2 pooling, the size is halved after pooling
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(16 * 56 * 56, 256)  # two pooling, so dim is 224/2/2=56
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)  # two category, so set 2

    def forward(self, x):
        # two conv
        x = self.pool(F.relu(self.conv1(x)))  # conv-->activate-->pooling
        x = self.pool(F.relu(self.conv2(x)))
        # three linear
        x = x.view(-1, 16 * 56 * 56)  # put data to 1 dim
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # the last fc no need to relu
        # x = F.log_softmax(x, dim=1) # NLLLoss() need, cross entropy no need
        return x  # the output of network is a positive and negative value, not a probability value


def load_net(flag):
    if flag == 'mynet':
        net = MYnet()

    elif flag == 'resnet':
        resnet50 = models.resnet50(pretrained=False)
        resnet50.load_state_dict(torch.load('resnet50-19c8e357.pth'))
        resnet50.fc = nn.Linear(2048, 2)  # 修改最后一层网络将输出调整为两维
        net = resnet50
    else:
        return None
    return net


