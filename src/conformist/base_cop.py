import math
import numpy as np
import matplotlib.pyplot as plt
from upsetplot import plot
import pandas as pd

from .validation_run import ValidationRun
from .validation_trial import ValidationTrial
from .prediction_dataset import PredictionDataset
from .output_dir import OutputDir
from .performance_report import PerformanceReport


DEFAULT_VALIDATION_PROPORTION = 0.25


class BaseCoP(OutputDir):
    def __init__(self, prediction_dataset: PredictionDataset, alpha=0.1):
        self.prediction_dataset = prediction_dataset
        self.alpha = alpha
        self.class_names = prediction_dataset.class_names()
        self._df = prediction_dataset.df
        self.n_total = self._df.shape[0]

        # Initialize
        self.n_cal, self.cal_smx, self.cal_labels, self.cal_labels_idx = \
            None, None, None, None

        # Depending on the type of predictor, this will be e.g. qhat or lamhat
        self.softmax_threshold = None

        # This is all the thresholds attempted by the CoP
        self.softmax_thresholds = None

        # Parse predictions
        self.smx = self.prediction_dataset.smx
        self.model_predictions = self.prediction_dataset.predictions_bool
        self.model_predictions_txt = self.prediction_dataset.predictions_str
        self.labels_txt = self.prediction_dataset.labels_str

    def calibrate(self):
        if self.n_cal is None or self.cal_smx is None or self.cal_labels is None:
            self._set_calibration_index()

    def get_prediction_sets(self):
        return self.val_smx >= self.softmax_threshold

    def _set_calibration_index(self, shuffle=False):
        if self.n_cal is None:
            self.n_cal = self.n_total

        idx = np.array([1] *
                       self.n_cal + [0] *
                       (self.smx.shape[0] - self.n_cal)) > 0
        if shuffle:
            np.random.shuffle(idx)

        # Calibration data
        self.cal_smx = self.smx[idx, :]
        self.cal_labels = self.prediction_dataset.labels_idx[idx]

        return idx

    def split_data(self,
                   validation_proportion,
                   shuffle=False):

        # Determine number of calibration and validation data points
        if validation_proportion == 0:
            self.n_cal = self.n_total
            self.n_val = 0
        else:
            calibration_proportion = 1 - validation_proportion
            self.n_cal = math.ceil(self.n_total * calibration_proportion)
            self.n_val = self.n_total - self.n_cal

        idx = self._set_calibration_index(shuffle)

        # Validation data
        if self.n_val > 0:
            self.val_smx = self.smx[~idx, :]
            self.val_labels = self.prediction_dataset.labels_idx[~idx]
            self.val_sample_names = np.array(self._df[
                PredictionDataset.ID_COL].values)[~idx]
            self.val_labels_txt = self.labels_txt[~idx]
            self.val_model_predictions = self.model_predictions[~idx]
            self.val_idx = ~idx

    def shuffle_data(self, validation_proportion):
        self.split_data(validation_proportion, shuffle=True)

    def do_validation_run(self, validation_proportion):
        self.shuffle_data(validation_proportion)
        self.calibrate()
        return ValidationRun(
                self.val_idx,
                self.val_sample_names,
                self.get_prediction_sets(),
                self.val_labels,
                self.val_model_predictions,
                self.softmax_threshold,
                self.class_names,
                self.softmax_thresholds)

    def do_validation_trial(self,
                            n_runs=1,
                            val_proportion=DEFAULT_VALIDATION_PROPORTION):
        self.runs = []
        for i in range(n_runs):
            result = self.do_validation_run(val_proportion)
            self.runs.append(result)
        return ValidationTrial(self.runs, self.class_names)

    def predict(self,
                pds,
                output_dir,
                validate=False,
                upset_plot_color="black"):

        self.create_output_dir(output_dir)

        self.smx = pds.smx
        self.val_smx = pds.smx
        self.val_labels = pds.labels_idx
        self.val_sample_names = pds.df[PredictionDataset.ID_COL].values
        self.val_labels_txt = pds.labels_str
        self.val_model_predictions = pds.predictions_bool
        self.val_idx = np.array([1] * pds.smx.shape[0])

        prediction_sets = self.get_prediction_sets()
        prediction_sets_text = self.prediction_sets_to_text(prediction_sets)
        formatted_predictions = pds.prediction_sets_df(prediction_sets_text)

        # -- WRITE PREDICTIONS TO CSV
        formatted_predictions.to_csv(f'{self.output_dir}/prediction_sets.csv')

        # -- GENERATE REPORTS --
        # Upset plot
        self.upset_plot(
            prediction_sets,
            self.output_dir,
            upset_plot_color)

        if validate:
            vr = ValidationRun(
                    self.val_idx,
                    self.val_sample_names,
                    prediction_sets,
                    self.val_labels,
                    self.val_model_predictions,
                    self.softmax_threshold,
                    self.class_names)
            vr.run_reports(self.output_dir)
        else:
            # Generate just basic statistics
            mean_set_size = PerformanceReport.mean_set_size(
                    prediction_sets)
            pr = PerformanceReport(self.output_dir)
            pr.visualize_mean_set_sizes_by_class(mean_set_size)

            stats = {
                'mean_set_size': mean_set_size,
                'pct_empty_sets': PerformanceReport.pct_empty_sets(
                    prediction_sets),
                'pct_singleton_sets': PerformanceReport.pct_singleton_sets(
                    prediction_sets),
                'pct_singleton_or_duo_sets':
                    PerformanceReport.pct_singleton_or_duo_sets(
                        prediction_sets),
                'pct_trio_plus_sets': PerformanceReport.pct_trio_plus_sets(
                    prediction_sets)
            }
            df = pd.DataFrame(stats, index=[0])
            df.T.to_csv(f'{self.output_dir}/summary.csv', header=False)

        return formatted_predictions

    def prediction_set_to_text(self, prediction_set, display_classes=None):
        class_names = self.class_names

        # Replace any class_names in display_classes with the display name
        if display_classes:
            class_names = [display_classes.get(cn, cn) for cn in class_names]

        ps = [class_names[i] for i in range(len(prediction_set)) if
              prediction_set[i]]

        return ','.join(ps)

    def prediction_sets_to_text(self, prediction_sets, display_classes=None):
        return [self.prediction_set_to_text(prediction_set, display_classes)
                for prediction_set in prediction_sets]

    def upset_plot(self, predictions_sets, output_dir, color="black"):
        plt.figure()

        class_names = self.class_names

        # Make prediction_sets into a df with class_names as columns
        upset_data = pd.DataFrame(predictions_sets, columns=class_names)

        upset_data.fillna(0, inplace=True)

        # Remove columns with no data
        upset_data = upset_data.loc[:, upset_data.sum(axis=0) > 0]

        # Set a multi-index
        upset_data.set_index(upset_data.columns.tolist(), inplace=True)

        plot(upset_data, sort_by="cardinality", facecolor=color, show_counts="%d", show_percentages="{:.0%}")

        path = f'{output_dir}/upset_plot.png'
        print("Saving upset plot to", path)
        plt.savefig(path)
