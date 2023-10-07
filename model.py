import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, N_input=786, layer_size=4, N_bottleneck=200, N_output=786):
        super(AutoEncoder, self).__init__()
        if layer_size - 1 % 2 == 0:
            print("Error, layers must be")
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


class MLP(nn.Module):
    def __init__(self, layers=[768, 768//2, 200, 768 // 2, 768]):
        super().__init__(self)
        self.hidden_size = len(layers) - 2

        if self.hidden_size <= 0:
            print("Model hidden size must be a positive number")
        
        self.layers = []
        for i in range(1, len(layers)):
            start = layers[i - 1]
            end = layers[i]
            self.layers.append(nn.Linear(start, end))
        self.input_shape = (1, 768)

    def forward(self, X):
        for i in range(len(self.layers) - 1):
            X = torch.relu(self.layers[i](X))
        X = torch.sigmoid(self.layers[-1](X))
        return X
    
    
