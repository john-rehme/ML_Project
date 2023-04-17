import torch
import torch.nn as nn
    
class Net(nn.Module): # TODO: revise architecture (add dropout, max pooling, etc)
    def __init__(self, start_dim, end_dim, bias=True, dropout = 0):
        super(Net, self).__init__()
        self.conv2d1 = nn.Conv2d(start_dim, 16, 3, bias=bias)
        self.maxpool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout2d(p=dropout)
        self.conv2d2 = nn.Conv2d(16, 16, 3, bias=bias)
        self.maxpool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout2d(p=dropout)
        self.conv2d3 = nn.Conv2d(16, 1, 3, bias=bias)
        
        self.linear1 = nn.Linear(1280, 256, bias=bias)
        self.dropout3 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(256, end_dim, bias=bias)

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.conv2d1(x))
        y = self.maxpool1(y)
        y = self.dropout1(y)
        y = self.relu(self.conv2d2(y))
        y = self.maxpool2(y)
        y = self.dropout2(y)
        y = self.relu(self.conv2d3(y))
        y = y.squeeze(1).flatten(start_dim=1)
        y = self.relu(self.linear1(y))
        y = self.dropout3(y)
        y = self.relu(self.linear2(y))
        return y
