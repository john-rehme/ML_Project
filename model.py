import torch
import torch.nn as nn

class Net(nn.Module): # TODO: revise architecture (add dropout, max pooling, etc)
    def __init__(self, start_dim, end_dim, bias=True):
        super(Net, self).__init__()
        self.conv2d1 = nn.Conv2d(start_dim, 16, 5, bias=bias)
        self.conv2d2 = nn.Conv2d(16, 16, 9, stride=3, bias=bias)
        self.conv2d3 = nn.Conv2d(16, 1, 9, bias=bias)

        self.linear1 = nn.Linear(1692, 256, bias=bias)
        self.linear2 = nn.Linear(256, end_dim, bias=bias)

    def forward(self, x):
        # x shape [BATCH_SIZE, 1, 208, 176] (WITHOUT CROPPING)
        y = torch.tanh(self.conv2d1(x))
        y = torch.tanh(self.conv2d2(y))
        y = torch.tanh(self.conv2d3(y))
        y = y.squeeze(1).flatten(start_dim=1) # shape [2048, x]
        y = torch.tanh(self.linear1(y))
        y = torch.tanh(self.linear2(y))
        # y shape [BATCH_SIZE, NUM_CLASS]
        return y