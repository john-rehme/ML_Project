import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module): # TODO: revise architecture (add dropout, max pooling, etc)
    def __init__(self, start_dim, end_dim, bias=True):
        super(Net, self).__init__()
        self.conv2d1 = nn.Conv2d(start_dim, 16, 5, bias=bias)
        self.conv2d2 = nn.Conv2d(16, 16, 9, stride=3, bias=bias)
        self.conv2d3 = nn.Conv2d(16, 1, 9, bias=bias)

        self.linear1 = nn.Linear(1692, 256, bias=bias)
        self.linear2 = nn.Linear(256, end_dim, bias=bias)

    def forward(self, x):
        # x shape [BATCH_SIZE, 1, 208, 176] (WITHOUT DATA PREPROCESSING)
        y = nn.ReLU(self.conv2d1(x))
        y = nn.ReLU(self.conv2d2(y))
        y = nn.ReLU(self.conv2d3(y))
        y = y.squeeze(1).flatten(start_dim=1) # shape [2048, x]
        y = nn.ReLU(self.linear1(y))
        y = nn.ReLU(self.linear2(y))
        # y shape [BATCH_SIZE, NUM_CLASS]
        return y
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bias=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, start_dim, end_dim, dropout_rate=0.2, bias=True):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(start_dim, 16, 3, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, 2, bias)
        self.layer2 = self._make_layer(16, 32, 2, bias)
        self.layer3 = self._make_layer(32, 64, 2, bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(1280, 256, bias=bias)
        self.fc2 = nn.Linear(256, end_dim, bias=bias)

    def _make_layer(self, in_channels, out_channels, stride, bias):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride, bias))
        layers.append(BasicBlock(out_channels, out_channels, 1, bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x