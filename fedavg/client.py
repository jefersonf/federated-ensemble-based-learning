import copy
import torch
import numpy as np

from torch.utils.data import DataLoader
from .utils import TorchDataset

from itertools import cycle
from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve, auc


class Client:
    """
    Attributes:

        model: Torch model.
        client_id (int): (Optional) Client identifier.
        default_inference_model (str): Model to be used as inference model.
        params (dict): Train params
        verbose (bool): If False, it does not show any log.
    """

    def __init__(self, model, params, client_id=None, logger=None):
        self.local_model = copy.deepcopy(model)
        self.client_id = client_id
        self.params = params
        self.logger = logger

    def set_local_model_weights(self, w_glob):
        """
        Change local model weights
        """
        self.local_model.load_state_dict(w_glob)


    def get_local_weights(self):
        """
        Return the local model weights
        """
        return copy.deepcopy(self.local_model.state_dict())


    def predict(self, features, target):
        x_train = torch.FloatTensor(features.values.astype(dtype='float64'))
        with torch.no_grad():
            out = self.local_model(x_train)
        pred = out.data.max(1, keepdim=True)[1]
        return pred.squeeze()

    def predict_proba(self, features, target):
        x_train = torch.FloatTensor(features.values.astype(dtype='float64'))
        with torch.no_grad():
            probs = self.local_model(x_train)
        return probs

    def train_local_model(self, features, target):
        """
        Will train models given the specifc architecture

        features (dataframe): train faeatures
        target (dataframe): output train label
        """

        x_train, y_train = (
            torch.FloatTensor(features.values.astype(dtype='float64')),
            torch.FloatTensor(target.values),
        )

        train_data = DataLoader(
            dataset=TorchDataset(x_train, y_train),
            batch_size=self.params["train_batch_size"],
            shuffle=True,
            drop_last=True,                                                                                                                                                                                                                   
        )

        if self.params["output_size"] == 1:
            assert (self.params["output_size"] + 1) >= len(
                set(list(target))
            ), "Model and target size does not match"
            self.train_torch_binary(train_data)
        else:
            assert self.params["output_size"] >= len(
                set(list(target))
            ), "Model and target size does not match"
            self.train_torch_multi(train_data)
    
    def binary_acc(self, y_pred, y_test):
        """
        calc binary accuracy
        """
        y_pred_tag = torch.round(torch.sigmoid(y_pred))
        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        # acc = torch.round(acc * 100)

        return acc
    
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
            fpr["micro"], tpr["micro"], _ = roc_curve(target_bin.ravel(), proba.numpy().ravel())
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

    def evaluate_local_model(self, features, target, metric=None):
        """
        Evaluate local NN model in test/validation set

        features (dataframe): train faeatures
        target (dataframe): output train label
        """
        if metric == "AUC":
            return self.auc_metric(target, self.predict_proba(features, target))

        x_evaluate, y_evaluate = (
            torch.FloatTensor(features.values.astype(dtype='float64')),
            torch.FloatTensor(target.values),
        )

        evaluate_data = DataLoader(
            dataset=TorchDataset(x_evaluate, y_evaluate),
            batch_size=self.params["evaluate_batch_size"],
            shuffle=True
        )

        self.local_model.eval()
        corrects = 0
        for X_batch, y_batch in evaluate_data:
            with torch.no_grad():
                out = self.local_model(X_batch)
                pred = out.data.max(1, keepdim=True)[1]
                corrects += pred.eq(y_batch.view_as(pred)).sum().item()

        accuracy = corrects / len(evaluate_data.dataset)
        return accuracy

    def train_torch_multi(self, train_data):
        """
        Train process for torch model
        """
        criterion = self.params["criterion"]
        optimizer = self.params["optmizer"](self.local_model.parameters(), lr=self.params["lr"], weight_decay=1e-5)

        self.local_model.train()
        for epoch in range(1, self.params["epochs"] + 1):
            total_loss = 0
            corrects = 0
            for data, target in train_data:
                optimizer.zero_grad()
                out = self.local_model(data)
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

            self.logger.info(
                f"Client{self.client_id}: epoch {epoch+0:02}, avg. loss: {avg_loss:.3f}, accuracy: {accuracy:.3f}"
            )


    def train_torch_binary(self, train_data):
        """
        Train process for torch model
        """
        criterion = self.params["criterion"]
        optimizer = self.params["optmizer"](self.local_model.parameters(), lr=self.params["lr"])

        self.local_model.train()
        for epoch in range(1, self.params["epochs"] + 1):
            epoch_loss = 0
            corrects = 0
            for X_batch, y_batch in train_data:
                optimizer.zero_grad()
                y_pred = self.local_model(X_batch)
                # pred = y_pred.data.max(1, keepdim=True)[1]
                pred = torch.sigmoid(y_pred)
                loss = criterion(y_pred, y_batch.unsqueeze(1))
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                corrects += pred.eq(y_batch.data.view_as(pred))

            avg_loss = epoch_loss / len(train_data)
            test_acc = corrects / len(train_data.dataset)
            self.logger.info(
                f"Epoch {epoch+0:03}, Batch Loss: {avg_loss:.3f}, Test Accuracy: {test_acc:.3f}"
            )
