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
    
class Test_Net(nn.Module):
    def __init__(self, start_dim, end_dim, dropout_rate=0.2, bias=True):
        super(Test_Net, self).__init__()
        self.conv1 = nn.Conv2d(start_dim, 16, 3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1, bias=bias)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1, bias=bias)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(25344, 256, bias=bias)
        self.fc2 = nn.Linear(256, end_dim, bias=bias)

    def forward(self, x):
        
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        #x = x.squeeze(1).flatten(start_dim=1)
        x = self.dropout(x)
        x = x.squeeze(1).flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x