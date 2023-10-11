import torch
from torch import nn

class Auto4LayerEncoder(nn.Module):
    def __init__(self, N_input=768, layer_size=4, N_bottleneck=200, N_output=768):
        super(Auto4LayerEncoder, self).__init__()
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
    """
    Multi Layer Perceptron with definable layers
    """
    def __init__(self, layers=[768, 768//2, 200, 768 // 2, 768]):
        super(MLP, self).__init__()
        self.hidden_size = len(layers) - 2

        if self.hidden_size <= 0:
            print("Model hidden size must be a positive number")
        
        self.layers = nn.ModuleList()  # Use nn.ModuleList instead of a Python list
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.input_shape = (1, layers[0])
        self.output_shape = (1, layers[-1])

    def forward(self, X):
        for i in range(len(self.layers) - 1):
            X = torch.relu(self.layers[i](X))
        X = torch.sigmoid(self.layers[-1](X))
        return X
    
    def save_file(self, save_path: str):
        checkpoint = {
            "layers": self.layers,
            "state_dict": self.state_dict()
        }
        torch.save(checkpoint, save_path)

def load_mlp_model(load_path: str):
    checkpoint = torch.load(load_path)
    model = MLP(layers=checkpoint["layers"])  # rebuild model from saved layers
    model.load_state_dict(checkpoint["state_dict"])  # load weights
    return model