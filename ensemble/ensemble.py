import copy
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import linear_model, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import label_binarize


ML_MODELS = [
    ("NaiveBayes", GaussianNB()),
    ("LogisticRegression", linear_model.LogisticRegression(max_iter=1000, C=0.001)),
    ("DecisionTreeClassifier", DecisionTreeClassifier(criterion='entropy', max_depth=10, splitter='best')),
    ("SVC", svm.SVC(probability=True)),
    ("KNN", KNeighborsClassifier(metric='manhattan',n_neighbors=10)),
    ("SGD",SGDClassifier(alpha=10, loss='log', max_iter=3000)),
    ("LDA", LDA(solver='svd',tol=0.1))
]


class EnsembleModel:
    """
    Args:
        models (dict): Dictionary of models.
        scores (list): List of model's scores.
        ensemble_id (str): Ensemble identifier.
        enable_grouping (bool): Enables the grouping of models of the same arch.
        logger: logging handler
    """
    def __init__(self, models, scores=None, ensemble_id=None, enable_grouping=False):
        self.models = copy.deepcopy(models)
        total_datapoints = sum([models[i].datapoints for i in models.keys()])
        self.scores = scores or [models[i].datapoints/total_datapoints for i in models.keys()]
        assert len(self.scores) == len(self.models), 'Number of scores must match the number of models.'
        self.ensemble_id = ensemble_id or "ensemble0"
        self.archtype_grouping = enable_grouping

    def fit(self, features, target):
        for client_id in self.models.keys():
            self.models[client_id].fit(features, target)

    def get_scores(self):
        return self.scores

    def evaluate(self, features, target):
        accuracy = metrics.accuracy_score(target, self.predict(features))
        return accuracy

    def plot_roc(self, fpr, tpr, title, label, args):
        import matplotlib.pyplot as plt 
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label=f'{label}')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC curve - {title}')
        plt.legend(loc='best')

        plt.savefig(f"{args.logdir}/fedel/roc_curve.png", dpi=300)

    def plot_mult_roc(self, fpr, tpr, roc_auc, n_classes, title, label, args):
        import matplotlib.pyplot as plt
        from itertools import cycle
        from scipy import interp
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
        roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=2)

        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'indianred', 'green'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC curve - {title}')
        plt.legend(loc="lower right")

        plt.savefig(f"{args.logdir}/fedel/roc_curve.png", dpi=300)

        return roc_auc


    def compute_metrics(self, features, target, args, round_id, label):
        n_classes = target.unique().shape[0]
        pred = self.predict(features)
        pred_proba = self._predict_proba(features)
        target = target.to_numpy()

        if n_classes > 2: # multiclass case
            target_bin = label_binarize(target, classes=np.arange(n_classes))
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = metrics.roc_curve(target_bin[:, i], pred_proba[:, i])
                last_round_roc = pd.DataFrame({"fpr": fpr[i], "tpr": tpr[i]})
                last_round_roc.to_csv(f"{args.logdir}/fedel/last_fpr{i}_tpr{i}.csv", index=False)
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])
                
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = metrics.roc_curve(target_bin.ravel(), pred_proba.ravel())
            roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

            all_roc_auc = self.plot_mult_roc(fpr, tpr, roc_auc, n_classes, title=f"{args.n_clients} clients (alpha={args.dirichlet_alpha})", label=f"{label}", args=args)
            micro_auc, macro_auc = all_roc_auc["micro"], all_roc_auc["macro"]

        else: # binary case
            fpr, tpr, _ = metrics.roc_curve(target, pred_proba[:,1])
            if round_id + 1 == args.rounds:
                last_round_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
                last_round_roc.to_csv(f"{args.logdir}/fedel/last_fpr_tpr.csv", index=False)

            auc = metrics.roc_auc_score(target, pred_proba[:,1])
            self.plot_roc(fpr, tpr, title=f"{args.n_clients} clients (alpha={args.dirichlet_alpha})", label=f"{label}, AUC={auc:.2f}", args=args)
            micro_auc, macro_auc = auc, auc # FIXME

        confusion_matrix = metrics.confusion_matrix(target, pred).ravel()
        precision, recall, _, _ = metrics.precision_recall_fscore_support(target, pred, average='weighted')

        return precision, recall, micro_auc, macro_auc, confusion_matrix

    def predict(self, features):
        return np.argmax(self._predict_proba(features), axis=1)
        
    def _predict(self, features):
        """Collect results from model.predict calls."""
        return np.asarray([m.local_model.predict(features) for m in self.models.values()]).T
            
    def _predict_proba(self, features):
        """Predict class probabilities for X in 'soft' voting."""
        probas, weights = self._collect_probas(features)
        return np.average(probas, weights=weights, axis=0)
    
    def _collect_probas(self, features):
        """Collect results from model.predict calls."""
        pred_probas = []
        weights = self.scores
        # FIXME
        if self.archtype_grouping:
            # print("archtype grouping...")
            archtype_group = {}
            for i, m in self.models.items():
                if not m.archtype in archtype_group:
                    archtype_group[m.archtype] = {
                    'proba': [], 
                    'coefs': [],
                    'combined_weights': 0}
                proba = m.local_model.predict_proba(features)
                archtype_group[m.archtype]['proba'].append(proba)
                if m.archtype.startswith("LogisticRegression"):
                    archtype_group[m.archtype]['coefs'].append(m.local_model.coef_[0].tolist())
                else:
                    archtype_group[m.archtype]['coefs'].append(0)
                archtype_group[m.archtype]['combined_weights'] += self.scores[i]

            weights = []
            combined_coefs = None
            for g in archtype_group:
                print(f"MODEL GROUP: {g}")
                pred_probas.append(np.average(archtype_group[g]['proba'], axis=0))
                if g.startswith("LogisticRegression"):
                    lr_coefs = np.asarray(archtype_group[g]['coefs'])
                    combined_coefs = np.mean(lr_coefs, axis=0)
                weights.append(archtype_group[g]['combined_weights'])

            pred_probas = np.asarray(pred_probas)
        else:
            pred_probas = np.asarray([m.local_model.predict_proba(features) for m in self.models.values()])
    
        return pred_probas, weights

    def __getitem__(self, client_id):
        return self.models[client_id].local_model

    def __setitem__(self, client_id, model):
        if client_id in self.models.keys():
            self.models[client_id].local_model = copy.deepcopy(model)
    
    def __len__(self):
        return len(self.models)