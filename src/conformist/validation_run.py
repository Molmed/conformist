import numpy as np
import pandas as pd
from .output_dir import OutputDir


class ValidationRun(OutputDir):
    EPSILON = 1e-10

    def __init__(self,
                 smx_idx,
                 sample_names,
                 prediction_sets,
                 labels_idx,
                 model_predictions,
                 softmax_threshold,
                 class_names,
                 softmax_thresholds=None):
        # The indices in the softmax array that correspond to the
        # samples in this run
        self.smx_idx = smx_idx

        # The names of the samples in this run
        self.sample_names = sample_names

        # The predictions for this run
        self.prediction_sets = prediction_sets

        # The labels for all samples as indices
        self.labels_idx = labels_idx

        # The qhat/lamhat value for this run, if provided
        self.softmax_threshold = softmax_threshold

        # The model predictions for this run
        self.model_predictions = model_predictions

        # The class names
        self.class_names = class_names

        # All attempted softmax thresholds
        self.softmax_thresholds = softmax_thresholds

    def mean_set_size(self):
        return sum(sum(prediction_set) for
                   prediction_set in self.prediction_sets) / \
                    len(self.prediction_sets)

    def pct_empty_sets(self):
        return sum(sum(prediction_set) == 0 for
                   prediction_set in self.prediction_sets) / \
                    len(self.prediction_sets)

    def pct_singleton_sets(self):
        return sum(sum(prediction_set) == 1 for
                   prediction_set in self.prediction_sets) / \
                    len(self.prediction_sets)

    def pct_singleton_or_duo_sets(self):
        return sum(sum(prediction_set) == 1 or sum(prediction_set) == 2 for
                   prediction_set in self.prediction_sets) / \
                    len(self.prediction_sets)

    def pct_trio_plus_sets(self):
        return sum(sum(prediction_set) >= 3 for
                   prediction_set in self.prediction_sets) / \
                    len(self.prediction_sets)

    def false_negative_rate(self):
        return 1 - ((self.prediction_sets * self.labels_idx).sum(axis=1) /
                    (self.labels_idx.sum(axis=1) + self.EPSILON)).mean()

    def model_false_negative_rate(self):
        return 1 - ((self.model_predictions * self.labels_idx).sum(axis=1) /
                    (self.labels_idx.sum(axis=1) + self.EPSILON)).mean()

    def true_positive_rate(self):
        return ((self.prediction_sets * self.labels_idx).sum(axis=1) /
                (self.labels_idx.sum(axis=1) + self.EPSILON)).mean()

    def model_true_positive_rate(self):
        return ((self.model_predictions * self.labels_idx).sum(axis=1) /
                (self.labels_idx.sum(axis=1) + self.EPSILON)).mean()

    def bin_set_sizes_by_class(self, class_names):
        set_sizes = {}
        for i in range(len(self.prediction_sets)):
            labels = self.labels_idx[i]
            # Get corresponding values from class_names
            pset_class_names = [class_names[i] for i, label in enumerate(labels) if label == 1]
            pset = self.prediction_sets[i]
            for class_name in pset_class_names:
                sizes = set_sizes.get(class_name, [])
                sizes.append(sum(pset))
                set_sizes[class_name] = sizes
        return set_sizes

    def mean_set_sizes_by_class(self, class_names):
        set_sizes = self.bin_set_sizes_by_class(class_names)
        averages = {}
        for key in set_sizes:
            sizes = set_sizes[key]
            averages[key] = sum(sizes) / len(sizes)
        return averages

    def mean_fnrs_by_class(self, class_names):
        fnrs = {}
        for i in range(len(self.prediction_sets)):
            labels = self.labels_idx[i]
            # Get corresponding values from class_names
            pset_class_names = [class_names[i] for i, label in enumerate(labels) if label == 1]
            pset = np.array([int(value) for value in self.prediction_sets[i]])
            if (pset[labels == 1] == 1).size > 0:
                fnr = 1 - np.mean((pset[labels == 1] == 1))
            else:
                fnr = np.nan  # or any other value you want to assign when the array is empty
            for class_name in pset_class_names:
                class_fnrs = fnrs.get(class_name, [])
                class_fnrs.append(fnr)
                fnrs[class_name] = class_fnrs
        averages = {}
        for key in fnrs:
            sizes = fnrs[key]
            averages[key] = sum(sizes) / len(sizes)
        return averages

    def run_reports(self, base_output_dir):
        np.seterr(all='raise')
        self.create_output_dir(base_output_dir)

        df = pd.DataFrame({
            'false_negative_rate': self.false_negative_rate(),
            'model_false_negative_rate': self.model_false_negative_rate(),
            'mean_set_size': self.mean_set_size(),
            'pct_empty_sets': self.pct_empty_sets(),
            'pct_singleton_sets': self.pct_singleton_sets(),
            'pct_singleton_or_duo_sets': self.pct_singleton_or_duo_sets(),
            'pct_trio_plus_sets': self.pct_trio_plus_sets()
        }, index=[0])

        df.T.to_csv(f'{self.output_dir}/summary.csv', header=False)

        df = pd.DataFrame(self.mean_set_sizes_by_class(self.class_names), index=[0])
        df.T.to_csv(f'{self.output_dir}/mean_set_sizes_by_class.csv', header=False)

        df = pd.DataFrame(self.mean_fnrs_by_class(self.class_names), index=[0])
        df.T.to_csv(f'{self.output_dir}/mean_fnrs_by_class.csv', header=False)

        print(f'Reports saved to {self.output_dir}')
