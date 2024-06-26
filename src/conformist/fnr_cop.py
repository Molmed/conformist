import numpy as np
from scipy.optimize import brentq

from .base_cop import BaseCoP
from .prediction_dataset import PredictionDataset


class FNRCoP(BaseCoP):
    def __init__(self, prediction_dataset: PredictionDataset, alpha=0.1):
        super().__init__(prediction_dataset, alpha)

    def false_negative_rate(self, prediction_set, gt_labels):
        return 1 - ((prediction_set * gt_labels).sum(axis=1) /
                    gt_labels.sum(axis=1)).mean()

    def calibrate(self):
        super().calibrate()

        def lamhat_threshold(lam):
            return self.false_negative_rate(
                    self.cal_smx >= lam, self.cal_labels) - \
                        ((self.n_cal + 1) / self.n_cal * self.alpha - 1
                         / (self.n_cal + 1))

        self.lamhat = brentq(lamhat_threshold, 0, 1)
        self.softmax_threshold = self.lamhat

        # For graphing
        # lambdas to test: from 0.0 to 1.0 in 0.01 increments
        self.lambdas = np.linspace(0, 1, 101)
        self.lambdas = {lam: lamhat_threshold(lam) for lam in self.lambdas}
        self.softmax_thresholds = self.lambdas
