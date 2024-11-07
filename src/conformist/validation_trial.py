import statistics
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from .output_dir import OutputDir
from .performance_report import PerformanceReport


class ValidationTrial(OutputDir):
    def __init__(self, runs, class_names=[]):
        self.runs = runs
        self.class_names = class_names

    def mean_set_size(self):
        means = []
        for run in self.runs:
            means.append(run.mean_set_size())
        return statistics.mean(means)

    def pct_empty_sets(self):
        empty = []
        for run in self.runs:
            empty.append(run.pct_empty_sets())
        return statistics.mean(empty)

    def pct_singleton_sets(self):
        singleton = []
        for run in self.runs:
            singleton.append(run.pct_singleton_sets())
        return statistics.mean(singleton)

    def pct_singleton_or_duo_sets(self):
        singleton_or_duo = []
        for run in self.runs:
            singleton_or_duo.append(run.pct_singleton_or_duo_sets())
        return statistics.mean(singleton_or_duo)

    def pct_trio_plus_sets(self):
        trio_plus = []
        for run in self.runs:
            trio_plus.append(run.pct_trio_plus_sets())
        return statistics.mean(trio_plus)

    def mean_false_negative_rate(self):
        fns = []
        for run in self.runs:
            fns.append(run.false_negative_rate())
        return statistics.mean(fns)

    def mean_model_false_negative_rate(self):
        fns = []
        for run in self.runs:
            fns.append(run.model_false_negative_rate())
        return statistics.mean(fns)

    def mean_true_positive_rate(self):
        tps = []
        for run in self.runs:
            tps.append(run.true_positive_rate())
        return statistics.mean(tps)

    def mean_model_true_positive_rate(self):
        tps = []
        for run in self.runs:
            tps.append(run.model_true_positive_rate())
        return statistics.mean(tps)

    def mean_softmax_threshold(self):
        return sum(run.softmax_threshold for run in self.runs) / len(self.runs)

    def mean_softmax_thresholds(self):
        return {key: statistics.mean([run.softmax_thresholds[key] for
                                      run in self.runs]) for key in self.runs[0].softmax_thresholds}

    def mean_set_sizes_by_class(self, class_names):
        set_size_dicts = []
        for run in self.runs:
            set_size_dicts.append(run.mean_set_sizes_by_class(class_names))

        means = {}
        for class_name in class_names:
            d_means = []
            for d in set_size_dicts:
                if class_name in d:
                    d_means.append(d[class_name])
            if len(d_means) > 0:
                means[class_name] = statistics.mean(d_means)
        return means

    def mean_fnrs_by_class(self, class_names):
        fnr_dicts = []
        for run in self.runs:
            fnr_dicts.append(run.mean_fnrs_by_class(run.prediction_sets,
                                                    class_names))

        means = {}
        for class_name in class_names:
            d_means = []
            for d in fnr_dicts:
                if class_name in d:
                    d_means.append(d[class_name])
            if len(d_means) > 0:
                means[class_name] = statistics.mean(d_means)
        return means

    def run_reports(self, base_output_dir):
        self.create_output_dir(base_output_dir)
        self.visualize_empirical_fnr()
        self.visualize_class_performance()
        print(f'Reports saved to {self.output_dir}')

    def visualize_empirical_fnr(self):
        plt.figure()

        # Generate a pastel palette
        color_palette = sns.color_palette("deep")

        ax = plt.gca()

        fnrs = []

        for run in self.runs:
            fnrs.append(run.false_negative_rate())

        # Get overall FNR
        mean_fnrs = round(self.mean_false_negative_rate(), 4)

        # Draw a histogram on the current subplot
        sns.histplot(fnrs, kde=False, ax=ax, color=color_palette[2])

        # Draw a vertical line at the mean
        ax.axvline(x=mean_fnrs, color='r', linestyle='--')

        # Set the labels of the x-axis and y-axis
        ax.set_xlabel('Mean run FNR')
        ax.set_ylabel('n runs')

        # Add a label to the inner right-hand corner of the current subplot
        ax.text(0.99, 0.95, f"Mean FNR across {len(self.runs)} runs: {mean_fnrs}", ha='right', va='top', transform=ax.transAxes)

        # Save the figure
        plt.savefig(f'{self.output_dir}/empirical_fnr.png')

    def visualize_class_performance(self):
        fnrs_by_class = self.mean_fnrs_by_class(self.class_names)
        mean_set_sizes = self.mean_set_sizes_by_class(self.class_names)
        pr = PerformanceReport(self.output_dir)
        pr.report_class_statistics(mean_set_sizes, fnrs_by_class)
