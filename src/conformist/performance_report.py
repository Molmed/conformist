import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .output_dir import OutputDir


class PerformanceReport(OutputDir):
    def __init__(self, base_output_dir):
        self.base_output_dir = base_output_dir

    def mean_set_size(prediction_sets):
        return sum(sum(prediction_set) for
                   prediction_set in prediction_sets) / \
                   len(prediction_sets)

    def pct_empty_sets(prediction_sets):
        return sum(sum(prediction_set) == 0 for
                   prediction_set in prediction_sets) / \
                    len(prediction_sets)

    def pct_singleton_sets(prediction_sets):
        return sum(sum(prediction_set) == 1 for
                   prediction_set in prediction_sets) / \
                    len(prediction_sets)

    def pct_singleton_or_duo_sets(prediction_sets):
        return sum(sum(prediction_set) == 1 or sum(prediction_set) == 2 for
                   prediction_set in prediction_sets) / \
                    len(prediction_sets)

    def pct_trio_plus_sets(prediction_sets):
        return sum(sum(prediction_set) >= 3 for
                   prediction_set in prediction_sets) / \
                    len(prediction_sets)

    def report_class_statistics(self,
                                mean_set_sizes_by_class,
                                mean_fnrs_by_class):

        # Setup
        self.create_output_dir(self.base_output_dir)
        plt.figure()

        # Sort the dictionary by its values
        mean_fnrs = dict(sorted(mean_fnrs_by_class.items(),
                                key=lambda item: item[1]))

        # Visualize this dict as a bar chart
        sns.set_style('whitegrid')
        palette = sns.color_palette("deep")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(mean_fnrs)), mean_fnrs.values(), color=palette[0])
        ax.set_xticks(range(len(mean_fnrs)))
        ax.set_xticklabels(mean_fnrs.keys(), rotation='vertical')
        ax.set_ylabel('Mean FNR')
        ax.set_xlabel('True class')
        plt.tight_layout()

        # Export as fig and text
        plt.savefig(f'{self.output_dir}/mean_fnrs_by_class.png')

        # Convert dictionary to dataframe and transpose
        df = pd.DataFrame(mean_fnrs, index=[0]).T

        # Save as csv
        df.to_csv(f'{self.output_dir}/mean_fnrs_by_class.csv',
                  index=True, header=False)

        # Reset plt
        plt.figure()

        # Sort the dictionary by its values
        mean_set_sizes = dict(sorted(mean_set_sizes_by_class.items(),
                                     key=lambda item: item[1]))

        # Convert dictionary to dataframe and transpose
        df = pd.DataFrame(mean_set_sizes, index=[0]).T

        # Save as csv
        df.to_csv(f'{self.output_dir}/mean_set_sizes_class.csv',
                  index=True, header=False)

        # Visualize this dict as a bar chart
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(mean_set_sizes)), mean_set_sizes.values(), color=palette[1])
        ax.set_xticks(range(len(mean_set_sizes)))
        ax.set_xticklabels(mean_set_sizes.keys(), rotation='vertical')
        ax.set_ylabel('Mean set size')
        ax.set_xlabel('True class')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/mean_set_sizes_by_class.png')
