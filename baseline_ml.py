import argparse
import itertools
import os

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from collections import defaultdict

import utils

from default_configs_baseline import (
    DEFAULT_ML_MODELS,
    DEFAULT_ML_PARAMETERS,
    DEFAULT_COLUMNS,
)


def build_parser():
    """
    Define options to run baseline metrics
    """

    parser = argparse.ArgumentParser(
        description="Baseline metrics to each individual ml model"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="nb",
        help="Define wich machine learning model will be used for evaluate task",
    )
    parser.add_argument("--data-path", dest="data_path", help="Path to training data.")
    parser.add_argument(
        "--target",
        default="label",
        help="Target column name to predict (default is label).",
    )
    parser.add_argument("--test-split", dest="test_split", type=float, default=0.2)
    parser.add_argument("--log-dir", dest="log_dir", default="./baseline")
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Sets verbose mode."
    )
    parser.add_argument("--tag", help="Tag to add in report files", default="")
    parser.add_argument("--seed", type=int, default=1, help="Random seed value.")
    parser.add_argument(
        "--evaluate", action="store_true", default=False, help="Sets evaluate model"
    )

    return parser.parse_args()


def plot_roc(fpr, tpr, title, label, tag, model_type):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot([0, 1], [0, 1], "k--")
    ax.plot(fpr, tpr, label=f"{label}")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(bbox_to_anchor=(0.08, 1.15), loc="upper left")

    plt.savefig(f"./baseline/ensemble/{model_type.upper()}/roc_curve_{tag}_{label}.png", dpi=600)


def compute_metrics(ml_model, features, target, args, best_params=None):
    pred = ml_model.predict(features)
    pred_proba = ml_model.predict_proba(features)[::, 1]
    auc = metrics.roc_auc_score(target, pred_proba)
    fpr, tpr, _ = metrics.roc_curve(target, pred_proba)
    plot_roc(
        fpr,
        tpr,
        f"{args.model_type} curve",
        f"AUC = {round(auc,3)},\nParams = {best_params}",
        args.tag,
        args.model_type,
    )
    confusion_matrix = metrics.confusion_matrix(target, pred).ravel()
    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        target, pred, average="weighted"
    )
    last_round_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    last_round_roc.to_csv(f'./baseline/ensemble/{args.model_type.upper()}/last_fpr_tpr_{args.tag}.csv',index=False)
    return precision, recall, fscore, auc


def grid_search_best_auc(X, y, args):
    """Search for the best params for every single model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed
    )

    ml_model = DEFAULT_ML_MODELS[args.model_type.upper()]()
    params = DEFAULT_ML_PARAMETERS[args.model_type.upper()]

    gs = GridSearchCV(
        ml_model,
        params,
        verbose=1,
        cv=5,
        n_jobs=-1,
        scoring="roc_auc" if y_test.unique().shape[0] == 2 else "accuracy",
    )
    gs_fit = gs.fit(X_train, y_train)

    return gs_fit, gs_fit.best_params_


def evaluate_ml(ml_model, params, X, y, args):
    """
    Evaluate a single each ML model over all train/test data
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_split, random_state=args.seed
    )

    log_path = utils.create_logdir(
        os.path.join(args.log_dir,'ensemble', str(args.model_type.upper()))
    )
    log_dict = defaultdict(list)

    if args.evaluate:
        results_persistence(
            ml_model, X_test, y_test, log_dict, log_path, args, params=params
        )
    else:
        params = DEFAULT_ML_PARAMETERS[args.model_type.upper()]
        if params is None:

            model = DEFAULT_ML_MODELS[args.model_type.upper()]()
            model.fit(X_train, y_train)

            results_persistence(model, X_test, y_test, log_dict, log_path, args)

        else:
            all_combines = make_all_combines(params)

            for params_keys in all_combines:
                returned_params = get_params(params_keys, params)

                model = DEFAULT_ML_MODELS[args.model_type.upper()](**returned_params)

                model.fit(X_train, y_train)
                results_persistence(
                    model,
                    X_test,
                    y_test,
                    log_dict,
                    log_path,
                    args,
                    params=returned_params,
                )


def results_persistence(model, X_test, y_test, log_dict, log_path, args, params=None):
    """ Function that join all results to save"""
    precision, recall, fscore, auc = compute_metrics(
        model, X_test, y_test, args, best_params=params
    )

    log_dict["ModelType"].append(args.model_type.upper())
    log_dict["Precision"].append(precision)
    log_dict["Recall"].append(recall)
    log_dict["F1Score"].append(fscore)
    log_dict["AUC"].append(auc)

    tagname = f"_{args.tag}" if len(args.tag) != 0 else ""
    utils.save_logs(
        log_dict,
        os.path.join(
            log_path, f"model_{args.model_type.upper()}_metrics_{tagname}.csv"
        ),
    )


def make_all_combines(params):
    """ Form a combinations that includes all params positions"""
    all_combinations_list = [[i for i in range(len(x))] for x in params.values()]
    all_combines = list(itertools.product(*all_combinations_list))
    return all_combines


def get_params(params_keys, params):
    """ get the params based on keys"""
    returned_params = {}
    for keys, params_key in zip(params_keys, params.keys()):
        returned_params[str(params_key)] = params[str(params_key)][keys]

    return returned_params


if __name__ == "__main__":

    args = build_parser()
    np.random.seed(args.seed)
    utils.create_logdir(args.log_dir)
    utils.create_logdir(os.path.join(args.log_dir, 'ensemble'))
    df = pd.read_csv(args.data_path)

    X, y = df.drop(args.target, axis=1), df[args.target]

    gs_params = None
    gs_fitted = None

    if args.evaluate:
        gs_fitted, gs_params = grid_search_best_auc(X, y, args)

    evaluate_ml(gs_fitted, gs_params,  X, y, args)
