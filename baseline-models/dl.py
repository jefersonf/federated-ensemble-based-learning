import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from collections import defaultdict

sys.path.append("../")
from fedavg.models import NeuralNetworkContainer
from fedavg.utils import TorchDataset
from utils import *


def plot_roc(fpr, tpr, label, args):
    import matplotlib.pyplot as plt 
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=f'{label}')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend(bbox_to_anchor=(0.08, 1.15), loc="upper left")
    plt.savefig(f"./baseline/fedavg/{args.model_type.upper()}/roc_curve.png", dpi=600)


def save_extra_reports(log_dict, archtype, n_classes, log_dir):
    if n_classes > 2:
        for i in range(len(log_dict['FalsePositiveRate_Class0'])):
            for j in range(n_classes):
                if isinstance(log_dict[f'FalsePositiveRate_Class{j}'][i], np.ndarray):
                    save_reports({
                        f'FPR_Class{j}': log_dict[f'FalsePositiveRate_Class{j}'][i], 
                        f'TPR_Class{j}': log_dict[f'TruePositiveRate_Class{j}'][i]},
                        f"{log_dir}/extras/{archtype}_last_fpr_tpr_for_class_{j}.csv")

        for i in range(len(log_dict['MicroFalsePositiveRate'])):
            if isinstance(log_dict['MicroFalsePositiveRate'][i], np.ndarray):
                save_reports({
                    'FPR_Micro': log_dict['MicroFalsePositiveRate'][i], 
                    'TPR_Micro': log_dict['MicroTruePositiveRate'][i]},
                    f"{log_dir}/extras/{archtype}_last_micro_fpr_tpr.csv")
    else:
        for i in range(len(log_dict['FalsePositiveRate'])):
            if isinstance(log_dict['FalsePositiveRate'][i], np.ndarray):
                save_reports({
                    'FPR': log_dict['FalsePositiveRate'][i], 
                    'TPR': log_dict['TruePositiveRate'][i]},
                    f"{log_dir}/extras/{archtype}_last_fpr_tpr.csv")
                    

def clean_reports(log_dict, n_classes):
    if n_classes > 2:
        for i in range(n_classes):
            if f"FalsePositiveRate_Class{i}" in log_dict:
                del log_dict[f"FalsePositiveRate_Class{i}"]
                del log_dict[f"TruePositiveRate_Class{i}"]
                del log_dict[f"AUC_Class{i}"]
        if "MicroFalsePositiveRate" in log_dict:
            del log_dict["MicroFalsePositiveRate"]
            del log_dict["MicroTruePositiveRate"]
            del log_dict["MicroAUC"]
    else:
        if "FalsePositiveRate" in log_dict:
            del log_dict["FalsePositiveRate"]
            del log_dict["TruePositiveRate"]
            del log_dict["AUC"]

def evaluate_cnn(X, y, cfg, args):
    """Prepare data to test."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed)

    x_train_torch, y_train_torch = (
        torch.FloatTensor(X_train.values.astype(dtype='float64')),
        torch.FloatTensor(y_train.values),
    )

    train_data = DataLoader(
        dataset=TorchDataset(x_train_torch, y_train_torch),
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    
    log_dict = defaultdict(list)
    log_training = defaultdict(list)

    models = cfg['DLModels']
    for archtype in models:
        print(models[archtype])
        model = NeuralNetworkContainer(
            X.columns.shape[0], 
            y.unique().shape[0], models[archtype])
        
        trained_model = train_model(model, train_data, log_training, args)
        compute_metrics(trained_model, X_test, y_test, log_dict, args, label=archtype)

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

def compute_metrics(model, features, target, log_dict, args, label):

    pred_proba = predict_proba(model, features).numpy()
    pred = predict(model, features).numpy()
    target = target.to_numpy()

    n_classes = np.unique(target).shape[0]
    if n_classes == 2:
        fpr, tpr, _ = metrics.roc_curve(target, pred_proba[::, 1])
        auc = metrics.roc_auc_score(target, pred_proba[::, 1], multi_class="ovr")
        log_dict["FalsePositiveRate"].append(fpr)
        log_dict["TruePositiveRate"].append(tpr)
        log_dict["AUC"].append(auc)
    else:
        target_bin = label_binarize(target, classes=np.arange(n_classes))
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(target_bin[:, i], pred_proba[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            log_dict[f"FalsePositiveRate_Class{i}"].append(fpr[i])
            log_dict[f"TruePositiveRate_Class{i}"].append(tpr[i])
            log_dict[f"AUC_Class{i}"].append(roc_auc[i])
            
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = metrics.roc_curve(target_bin.ravel(), pred_proba.ravel())
        roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
        log_dict[f"MicroFalsePositiveRate"].append(fpr["micro"])
        log_dict[f"MicroTruePositiveRate"].append(tpr["micro"])
        log_dict[f"MicroAUC"].append(roc_auc["micro"])

    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(target, pred)

    for i in range(n_classes):
        log_dict[f"Precision_Class{i}"].append(precision[i])
        log_dict[f"Recall_Class{i}"].append(recall[i])
        log_dict[f"F1Score_Class{i}"].append(fscore[i])
    
    save_extra_reports(log_dict, label, n_classes, args.log_dir)
    clean_reports(log_dict, n_classes)
    save_reports(log_dict, os.path.join(args.log_dir, "all_metrics.csv"))

def train_model(model, train_data, log_training, args):
    """Train NN model."""
    criterion = F.cross_entropy
    optimizer = optim.Adam(model.parameters(), lr = 0.001, weight_decay = 1e-5)
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        corrects = 0
        log_training["Epoch"].append(epoch)
        for data, target in train_data:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target.long())
            total_loss += loss.item()

            loss.backward()
            total_loss += loss.item()
            optimizer.step()

            pred = out.data.max(1, keepdim=True)[1]
            corrects += pred.eq(target.data.view_as(pred)).sum().item()

        accuracy = corrects / len(train_data.dataset)
        avg_loss = total_loss / len(train_data.dataset)
        batch_loss = total_loss / len(train_data)

        log_training["Accuray"].append(accuracy)
        log_training["AvgLoss"].append(avg_loss)
        log_training["BatchLoss"].append(batch_loss)

    save_reports(log_training, os.path.join(args.log_dir, "training_results.csv"))
    return model

def build_parser():
    """Define options to run baseline metrics."""
    parser = argparse.ArgumentParser(description="Baseline metrics to each individual ml model")
    parser.add_argument("--data-path", dest="data_path", 
        help="Path to training data.")
    parser.add_argument("--target", default="",
        help="Target column name to predict (default is label).")   
    parser.add_argument("--test-split", dest="test_split", type=float, default=0.2)
    parser.add_argument("--log-dir", dest="log_dir", default="./baselines/dl")
    parser.add_argument("-c", "--config", dest="config_path", default="./configs.yaml",
        help="Path to training configuration file")
    parser.add_argument("--train_batch_size", type=int, default=32,
        help="batch size for train NN model", )
    parser.add_argument("--evaluate_batch_size", type=int, default=64, 
        help="batch size for test/val NN model")
    parser.add_argument("--epochs",type=int, default=5,
        help="num of epochs to train NN model")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, 
        help="Sets verbose mode.")
    parser.add_argument("--seed", type=int, default=1, 
        help="Random seed value.")

    return parser.parse_args()


if __name__ == "__main__":
    
    args = build_parser()
    np.random.seed(args.seed)

    cfg = load_config(args.config_path)
    create_logdir(args.log_dir)
    create_logdir(args.log_dir + "/extras")

    if args.target == "":
        dataset_name = args.data_path.split("/")[-1][:-4]
        if dataset_name == "shelter":
            args.target = "OutcomeType"
            args.dataset_name = "shelter"
        elif dataset_name == "diabetes":
            args.target = "Outcome"
            args.dataset_name = "diabetes"
        print(f"Setting target feature to \"{args.target}\" for {dataset_name} dataset")

    df = pd.read_csv(args.data_path)
    X, y = df.drop(args.target, axis=1), df[args.target]
    
    evaluate_cnn(X, y, cfg, args)