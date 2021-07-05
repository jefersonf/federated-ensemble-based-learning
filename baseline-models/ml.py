import argparse
import itertools
from math import log
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from collections import defaultdict

sys.path.append("../")
from utils import * 

def plot_roc(fpr, tpr, label, tag, model_type):
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot([0, 1], [0, 1], "k--")
    ax.plot(fpr, tpr, label=f"{label}")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(bbox_to_anchor=(0.08, 1.15), loc="upper left")
    plt.savefig(f"./fedel/{model_type.upper()}/roc_curve_{tag}_{label}.png", dpi=600)

def compute_metrics(model, features, target, log_dict):
    pred = model.predict(features)
    n_classes = np.unique(target).shape[0]
    if n_classes == 2:
        pred_proba = model.predict_proba(features)[::, 1]
        auc = metrics.roc_auc_score(target, pred_proba, multi_class="ovr")
        fpr, tpr, _ = metrics.roc_curve(target, pred_proba)
        log_dict["FalsePositiveRate"].append(fpr)
        log_dict["TruePositiveRate"].append(tpr)
        log_dict["AUC"].append(auc)
    else:
        pred_proba = model.predict_proba(features)
        target_bin = label_binarize(target.to_numpy(), classes=np.arange(n_classes))
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

    # confusion_matrix = metrics.confusion_matrix(target, pred).ravel()
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        target, pred, average="weighted"
    )

    # last_round_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    # last_round_roc.to_csv(f'./fedel/shelter/last_fpr_tpr.csv', index=False)

    log_dict["Precision"].append(precision)
    log_dict["Recall"].append(recall)
    log_dict["F1Score"].append(fscore)

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

def make_all_combinations(params):
    """ Form a combinations that includes all params positions"""
    if params is None:
        return []
    all_combinations_list = [[i for i in range(len(x))] for x in params.values()]
    all_combination_idxs = list(itertools.product(*all_combinations_list))
    all_combinations = []
    param_names = list(params.keys())
    for param_set in all_combination_idxs:
        param_dict = {}
        for j in range(len(param_set)):
            param_dict[param_names[j]] = params[param_names[j]][param_set[j]]
        all_combinations.append(param_dict)
    return all_combinations

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

def evaluate_ml(X, y, cfg, args):
    """Evaluate each ML model over all train/test data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed
    )
    
    n_classes = np.unique(y_test).shape[0]

    log_dict = defaultdict(list)
    log_gs_dict = defaultdict(list)
    
    models =  cfg['MLModels']
    for archtype in models:
        try: 
            gs = GridSearchCV(
                eval(archtype)(), 
                models[archtype], 
                verbose=1, cv=cfg['KFold'], n_jobs=-1,
                scoring="roc_auc" if y_test.unique().shape[0] == 2 else "accuracy")
            gs_model = gs.fit(X_train, y_train)
            log_gs_dict["ModelType"].append(archtype)
            compute_metrics(gs_model.best_estimator_, X_test, y_test, log_gs_dict)  
            print(f"{archtype} best params by Grid Search:", gs_model.best_params)
        except:
            print("No parameters set.")

        all_combinations = make_all_combinations(models[archtype])
        if len(all_combinations) != 0:
            for params in all_combinations:
                model = eval(archtype)(**params)
                print("Running..", model)
                log_dict["ModelType"].append(archtype)
                log_dict["Params"].append(params)
                model.fit(X_train, y_train)
                compute_metrics(model, X_test, y_test, log_dict)
                save_extra_reports(log_dict, archtype, n_classes, args.log_dir)
                clean_reports(log_dict, n_classes)
                save_reports(log_dict, os.path.join(args.log_dir, "all_configurations.csv"))

        save_extra_reports(log_gs_dict, archtype + "_GridSearch", n_classes, args.log_dir)

    clean_reports(log_gs_dict, n_classes)
    save_reports(log_gs_dict, os.path.join(args.log_dir, "grid_search_results.csv"))

def build_parser():
    """Define options to run baseline metrics. """
    parser = argparse.ArgumentParser(
        description="Baseline metrics to each individual ml model"
    )
    parser.add_argument("--data-path", dest="data_path", help="Path to training data.")
    parser.add_argument("--target", default="",
        help="Target column name to predict (default is label).")
    parser.add_argument("--test-split", dest="test_split", type=float, default=0.2)
    parser.add_argument("--log-dir", dest="log_dir", default="./baselines/ml")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, 
        help="Sets verbose mode.")
    parser.add_argument("-c", "--config", dest="config_path", default="./configs.yaml",
        help="Path to training configuration file")
    parser.add_argument("--tag", help="Tag to add in report files", default="")
    parser.add_argument("--seed", type=int, default=1, help="Random seed value.")

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

    evaluate_ml(X, y, cfg, args)