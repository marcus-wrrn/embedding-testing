import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, N_input=786, N_bottleneck=200, N_output=786):
        super(AutoEncoder, self).__init__()
        N2 = N_input // 2
        self.fc1 = nn.Linear(N_input, N2)       # input = 1x784, output = 1x392
        self.fc2 = nn.Linear(N2, N_bottleneck)  # output = 1xN
        self.fc3 = nn.Linear(N_bottleneck, N2)  # output = 1x392
        self.fc4 = nn.Linear(N2, N_output)      # output = 1x784
        self.input_shape = (1, 28 * 28)

    def forward(self, X):
        return self.decode(self.encode(X))

    def encode(self, X):
        # encoder
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))

        return X

    def decode(self, X):
        # decoder
        X = torch.relu(self.fc3(X))
        X = torch.sigmoid(self.fc4(X))
        return X
