import torch
import torch.nn as nn

# default hidden layers configs
DEFAULT_NN_CONFIGS = {
    # COVID-19 configs
    "A": [19],
    # Shelter dataset configs
    "C": [18, 18],
    "D": [36, 18],
}

class NeuralNetworkContainer(nn.Module):
    def __init__(self, n_features, output_size, config=None):
        super(NeuralNetworkContainer, self).__init__()
        self.n_features = n_features
        self.output_size = output_size

        if config is None:
            config = DEFAULT_NN_CONFIGS['A']

        self.features = self.__build_layers(config)

    def __build_layers(self, config):
        layers = []
        for i, h in enumerate(config):
            if i == 0:
                layers += [nn.Linear(self.n_features, h), nn.ReLU(), nn.BatchNorm1d(h)]  
            else:
                layers += [nn.Linear(prev_h, h), nn.ReLU(), nn.BatchNorm1d(h)]
            prev_h = h

        layers += [nn.Linear(prev_h, self.output_size), nn.Softmax(dim=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)
