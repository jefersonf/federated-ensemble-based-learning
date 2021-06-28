import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn import metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from collections import defaultdict

from fedavg.models import NeuralNetworkContainer
from fedavg.utils import TorchDataset

from default_configs_baseline import (
    DEFAULT_COLUMNS,
    DEFAULT_NN_CONFIGS
)


import utils

def build_parser():
    """
    Define options to run baseline metrics
    """

    parser = argparse.ArgumentParser(description="Baseline metrics to each individual ml model")
    parser.add_argument("--model_type", type=str,default="A", 
        help="Define wich machine learning model will be used for evaluate task")
    parser.add_argument("--data-path", dest="data_path", help="Path to training data.")
    parser.add_argument("--target", default="label",
        help="Target column name to predict (default is label).")   
    parser.add_argument("--test-split", dest="test_split", type=float, default=0.2)
    parser.add_argument("--log-dir", dest="log_dir",default="./baseline")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, 
        help="Sets verbose mode.")
    parser.add_argument("--tag", help="Tag to add in report files", default="")
    parser.add_argument("--seed", type=int, default=1, help="Random seed value.")
    parser.add_argument("--train_batch_size", type=int, default=32, help="batch size for train NN model", )
    parser.add_argument("--evaluate_batch_size", type=int, default=64, help="batch size for test/val NN model")
    parser.add_argument("--epochs",type=int, default=5, help="num of epochs to train NN model")

    return parser.parse_args()



def evaluate_cnn(X, y, args, model):
    """Prepare data to test
    """
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed
    )
    
    x_train_torch, y_train_torch = (
        torch.FloatTensor(X_train.values.astype(dtype='float64')),
        torch.FloatTensor(y_train.values),
    )

    train_data = DataLoader(
        dataset=TorchDataset(x_train_torch, y_train_torch),
        batch_size=args.train_batch_size,
        shuffle=True,
    )


    log_path = utils.create_logdir(
        os.path.join(args.log_dir,'fedavg', str(args.model_type.upper()))
    )
    log_dict = defaultdict(list)

    trained_model = train_model(model, train_data, args)
    
    precision, recall, roc_auc, confusion_matrix = compute_metrics(trained_model, X_test, y_test, args, label=str(DEFAULT_NN_CONFIGS[args.model_type.upper()]))


def plot_roc(fpr, tpr, label, args):
    import matplotlib.pyplot as plt 
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=f'{label}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(bbox_to_anchor=(0.08, 1.15), loc="upper left")
    plt.savefig(f"./baseline/fedavg/{args.model_type.upper()}/roc_curve_{args.tag}.png", dpi=600)

def predict_proba(model, features):
    x_train = torch.FloatTensor(features.values.astype(dtype='float64'))
    model.eval()
    with torch.no_grad():
        probs = model(x_train)
    return probs

def predict(model, features):
    x_train = torch.FloatTensor(features.values.astype(dtype='float64'))
    model.eval()
    with torch.no_grad():
        out = model(x_train)
    pred = out.data.max(1, keepdim=True)[1]
    return pred.squeeze()

def compute_metrics(model, features, target, args, label):

    pred_proba = predict_proba(model, features).numpy()
    pred = predict(model, features).numpy()

    fpr, tpr, thresholds = metrics.roc_curve(target, pred_proba[:,1])
    target = target.to_numpy()

    roc_auc = metrics.roc_auc_score(target, pred_proba[:,1])
    confusion_matrix = metrics.confusion_matrix(target, pred).ravel()
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(target, pred)
    fpr, tpr, _ = metrics.roc_curve(target, pred_proba[:,1])

    plot_roc(fpr, tpr, label=f"{label}, AUC={roc_auc:.2f}", args=args)

    return precision, recall, roc_auc, confusion_matrix


def train_model(model, train_data, args):
    """Train NN model
    """
    criterion = F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-5)
    model.train()

    for epoch in range(1, args.epochs + 1):

        total_loss = 0
        corrects = 0

        for data, target in train_data:
            optimizer.zero_grad()
            out = model(data)
            loss= criterion(out, target.long())
            total_loss += loss.item()

            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            pred = out.data.max(1, keepdim=True)[1]
            corrects += pred.eq(target.data.view_as(pred)).sum().item()

            accuracy = corrects / len(train_data.dataset)
            avg_loss = total_loss / len(train_data.dataset)
            batch_loss = total_loss / len(train_data)


    return model

if __name__ == "__main__":
    
    args = build_parser()
    np.random.seed(args.seed)
    utils.create_logdir(args.log_dir)
    utils.create_logdir(os.path.join(args.log_dir, 'fedavg'))
    df = pd.read_csv(args.data_path)

    X, y = df.drop(args.target, axis=1), df[args.target]
    
    model = NeuralNetworkContainer(X.columns.shape[0], y.unique().shape[0],DEFAULT_NN_CONFIGS[args.model_type.upper()])
    evaluate_cnn(X, y,args, model )