import random
import copy

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize

class Server:
    """
    Args:
        Some: nothing for now

    """  
    def __init__(self):
        self.shared_model_archtype = None
        self.shared_global_weights = None
        self._averaged_datapoint_weights = None

    def fedavg(self, models, datapoint_freqs):
        assert len(models) == len(datapoint_freqs)
        avg_model = copy.deepcopy(models[0])
        for p in avg_model.keys():
            if 'weights' in p or 'bias' in p:
                avg_model[p] *= datapoint_freqs[0]
                for i in range(1, len(models)):
                    avg_model[p] += (models[i][p] * datapoint_freqs[i])

        return avg_model

    def set_model_archtype(self, archtype):
        self.shared_model_archtype = archtype

    def update_global_model(self, clients, datapoints):
        """
        Update the global model based on each client train and datapoints division
        Args:
            clients
            datatpoints
        """

        clients_weights = [clients[i].get_local_weights() for i in clients.keys()]
        total_datapoints = sum(len(datapoints[i]) for i in clients)
        self._averaged_datapoint_weights = [len(datapoints[i]) / total_datapoints for i in clients]
        
        # Averaged model weights
        self.shared_global_weights = self.fedavg(clients_weights, self._averaged_datapoint_weights)

        for i in clients:
            clients[i].set_local_model_weights(copy.deepcopy(self.shared_global_weights))

    def evaluate(self, clients, features, target):
        """
        Evalute a randomly selected model after update all clients
        """
        averaged_model = copy.deepcopy(clients[random.randint(0, len(clients) - 1)])
        averaged_model.local_model.load_state_dict(self.shared_global_weights)
        evaluation = averaged_model.evaluate_local_model(features, target)
        return evaluation

    def plot_roc(self, fpr, tpr, title, label, args):
        import matplotlib.pyplot as plt 
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label=f'{label}')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'ROC curve - {title}')
        plt.legend(loc='best')

        plt.savefig(f"{args.logdir}/fedavg/roc_curve.png", dpi=300)

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

        plt.savefig(f"{args.logdir}/fedavg/roc_curve.png", dpi=300)

        return roc_auc


    def compute_metrics(self, clients, features, target, args, label):
        n_classes = target.unique().shape[0]
        averaged_model = copy.deepcopy(clients[random.randint(0, len(clients) - 1)])
        averaged_model.local_model.load_state_dict(self.shared_global_weights)
        pred = averaged_model.predict(features, target).numpy()
        pred_proba = averaged_model.predict_proba(features, target).numpy()
        target = target.to_numpy()

        if n_classes > 2:
            target_bin = label_binarize(target, classes=np.arange(n_classes))
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = metrics.roc_curve(target_bin[:, i], pred_proba[:, i])
                last_round_roc = pd.DataFrame({"fpr": fpr[i], "tpr": tpr[i]})
                last_round_roc.to_csv(f"{args.logdir}/fedavg/last_fpr{i}_tpr{i}.csv", index=False)
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])
                
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = metrics.roc_curve(target_bin.ravel(), pred_proba.ravel())
            roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

            all_roc_auc = self.plot_mult_roc(fpr, tpr, roc_auc, n_classes, title=f"{args.n_clients} clients (alpha={args.dirichlet_alpha})", label=f"{label}", args=args)
            micro_auc, macro_auc = all_roc_auc["micro"], all_roc_auc["macro"]
            
        else:
            fpr, tpr, _ = metrics.roc_curve(target, pred_proba[:,1])
            auc = metrics.roc_auc_score(target, pred_proba[:,1])
            last_round_roc = pd.DataFrame({"fpr": fpr, "tpr": tpr})
            last_round_roc.to_csv(f"{args.logdir}/fedavg/last_fpr_tpr.csv", index=False)
            self.plot_roc(fpr, tpr, title=f"{args.n_clients} clients (alpha={args.dirichlet_alpha})", label=f"{label}, AUC={auc:.2f}", args=args)

            micro_auc, macro_auc = auc, auc
        
        confusion_matrix = metrics.confusion_matrix(target, pred).ravel()
        precision, recall, _, _ = metrics.precision_recall_fscore_support(target, pred)

        return precision, recall, micro_auc, macro_auc, confusion_matrix