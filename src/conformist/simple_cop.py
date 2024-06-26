import numpy as np
from .base_cop import BaseCoP
from .prediction_dataset import PredictionDataset


class SimpleCoP(BaseCoP):
    def __init__(self, prediction_dataset: PredictionDataset, alpha=0.1):
        super().__init__(prediction_dataset, alpha)

    def calibrate(self):
        super().calibrate()

        # Conformal prediction time
        # 1: get conformal scores.
        cal_scores = 1 - self.cal_smx[np.arange(self.n_cal), self.cal_labels]

        # 2: get adjusted quantile
        q_level = np.ceil((self.n_cal + 1) * (1 - self.alpha)) / self.n_cal
        self.qhat = np.quantile(cal_scores, q_level, interpolation='higher')
        self.softmax_threshold = 1 - self.qhat
