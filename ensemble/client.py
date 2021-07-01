import copy
import numpy as np

from itertools import cycle
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc


class Client:
    """
    Args:
        model: SKLearn model.
        client_id (str): (Optional) Client identifier.
        default_inference_model (str): Model to be used as inference model.
            Could be 'local' or 'shared' for ensemble model (default is local).
        verbose (bool): If False, it does not show any log.
    """
    def __init__(self, model, client_id=None, archtype=None, default_inference_model='local', verbose=True):

        self.local_model = copy.deepcopy(model)
        self.shared_model = None
        self.client_id = client_id
        self.archtype = archtype
        self.datapoints = 1
        self.default_inference_model = default_inference_model.lower()
        self.verbose = verbose

    def update_local_scores(self, features, target):
        if self.shared_model:
            relative_acc = []
            for k in self.shared_model.models.keys():
                prediction = self.shared_model[k].predict(features)
                relative_acc.append(accuracy_score(target, prediction))
            accum_acc = sum(relative_acc)
            self.shared_model.scores = [a/accum_acc for a in relative_acc]

    def fit(self, features, target):
        self.datapoints = max(self.datapoints, features.shape[0])
        self.local_model.fit(features, target)
        if self.shared_model:
            self.shared_model[self.client_id] = copy.deepcopy(self.local_model)

    def predict(self, features):
        self.local_model.predict(features)

    def auc_metric(self, target, proba):
        n_classes = target.unique().shape[0]
        if n_classes > 2:
            target_bin = label_binarize(target.to_numpy(), classes=np.arange(n_classes))
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(target_bin[:, i], proba[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(target_bin.ravel(), proba.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
           
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            return roc_auc["micro"], roc_auc["macro"]

        # binary case
        bin_auc = roc_auc_score(target.to_numpy(), proba[:,1])
        return bin_auc, bin_auc # FIXME just for compatibility


    def evaluate(self, features, target, metric=None):
        if self.shared_model and self.default_inference_model == 'shared':
            if metric == "AUC":
                return self.auc_metric(target, self.shared_model._predict_proba(features))
            prediction = self.shared_model.predict(features)
        else:
            if metric == "AUC":
                return self.auc_metric(target, self.local_model.predict_proba(features))
            prediction = self.local_model.predict(features)

        return accuracy_score(target, prediction)

    def set_ensemble_model(self, ensemble_model):
        self.shared_model = ensemble_model
