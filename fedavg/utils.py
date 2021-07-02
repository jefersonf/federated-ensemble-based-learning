import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset
from .models import NeuralNetworkContainer, DEFAULT_NN_CONFIGS

def build_network(config_type, n_features, output_size):
    """
    Define wich model fedavg will use
    args (argparse): Main options that users give
    """
    if config_type.upper() in DEFAULT_NN_CONFIGS:
        config = DEFAULT_NN_CONFIGS[config_type.upper()]
    else:
        raise error('A model config. type should be set!')

    model = NeuralNetworkContainer(n_features, output_size, config)
    print(model)
    return model


def define_nn_params(args, num_output):
    """
    Define train params for fedavg
    args (argparse): Main options that users give
    """
    params = {
        "epochs": args.epochs,
        "lr": args.lr,
        "criterion": nn.BCEWithLogitsLoss()
        if num_output == 1
        else F.cross_entropy,
        "optmizer": optim.Adam ,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.test_batch_size,
        "output_size": num_output,
    }

    return params


class TorchDataset(Dataset):
    
    def __init__(self, X_data, y_data=None):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        if not self.y_data is None:
            return self.X_data[index], self.y_data[index]
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)