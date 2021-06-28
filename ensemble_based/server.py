from .ensemble import EnsembleModel

import copy
import numpy as np


class Server:
    """
    Args:
        ensemble_model: An instance of EnsembleModel. 

    """
    def __init__(self, ensemble_model=None):
        self.shared_ensemble = ensemble_model

    def _update_scores(self, score_list):
        self.shared_ensemble.scores = list(np.mean(score_list, axis=0))

    def update_ensemble(self, clients, datapoints):

        total_datapoints = sum(len(datapoints[i]) for i in clients)
        averaged_datapoint_weights = [len(datapoints[i]) / total_datapoints for i in clients]

        if not self.shared_ensemble:
            return
        n = len(self.shared_ensemble)
        score_list = np.zeros((n, n))
        for i, client_id in enumerate(clients.keys()):
            self.shared_ensemble[client_id] = copy.deepcopy(clients[client_id].local_model)
            client_weights = copy.deepcopy(clients[client_id].shared_model.scores)
            client_over_datapoints = [score * averaged_datapoint_weights[i] for score in client_weights]
            score_list[i] = [score/sum(client_over_datapoints) for score in client_over_datapoints]


        self._update_scores(score_list)

    def get_shared_model(self):
        return self.shared_ensemble

    def set_shared_model(self, ensemble_model):
        self.shared_ensemble = ensemble_model