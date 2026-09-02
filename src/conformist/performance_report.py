import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .output_dir import OutputDir


class PerformanceReport(OutputDir):
    FIGURE_FONTSIZE = 12
    FIGURE_WIDTH = 12
    FIGURE_HEIGHT = 8
    plt.rcParams.update({'font.size': FIGURE_FONTSIZE})
    plt.rcParams["pdf.fonttype"] = 42  # If saving as PDF too
    plt.rcParams["font.family"] = "Univers, DejaVu Sans"  # Choose a font that supports embedding



    def __init__(self, base_output_dir):
        self.create_output_dir(base_output_dir)

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

    def _pct_sets_of_min_size(prediction_sets, min_size):
        return sum(sum(prediction_set) >= min_size for
                   prediction_set in prediction_sets) / \
                    len(prediction_sets)

    def pct_duo_plus_sets(prediction_sets):
        return PerformanceReport._pct_sets_of_min_size(prediction_sets, 2)

    def pct_trio_plus_sets(prediction_sets):
        return PerformanceReport._pct_sets_of_min_size(prediction_sets, 3)

    def _class_report(self,
                      items_by_class,
                      output_file_prefix,
                      ylabel,
                      color,
                      title=None):
        # Reset plt
        fig = plt.figure(figsize=(self.FIGURE_WIDTH, self.FIGURE_HEIGHT))
        plt.tight_layout()
        plt.rcParams.update({'font.size': 8})

        # Remove the grid
        plt.grid(False)

        # Sort the dictionary by its values
        mean_sizes = dict(sorted(items_by_class.items(),
                                 key=lambda item: item[1]))

        # Convert dictionary to dataframe and transpose
        df = pd.DataFrame(mean_sizes, index=[0]).T

        # Save as csv
        df.to_csv(f'{self.output_dir}/{output_file_prefix}.csv',
                  index=True, header=False)

        # Visualize this dict as a bar chart
        # sns.set_style('whitegrid')
        fig, ax = plt.subplots()
        # plt.tight_layout()
        bars = ax.barh(
            range(len(mean_sizes)),
            mean_sizes.values(),
            color=color)
        ax.set_yticks(range(len(mean_sizes)))
        ax.set_yticklabels(mean_sizes.keys(), fontsize=8)
        ax.tick_params(axis='both')
        ax.set_xlabel(ylabel)
        ax.set_ylabel('True class')

        # Print the number above each bar
        for bar in bars:
            width = bar.get_width()
            ax.annotate(
                f'{width:.2f}',
                xy=(width, bar.get_y() + bar.get_height() / 2.0),
                xytext=(6, 0),
                textcoords='offset points',
                ha='left',
                va='center',
                fontsize=8,
            )
        ax.margins(x=0.15)

        # fig.set_size_inches(4, 3)
        plt.tight_layout(w_pad=0)
        plt.savefig(f'{self.output_dir}/{output_file_prefix}.svg', format='svg')

    def visualize_mean_set_sizes_by_class(self,
                                          mean_set_sizes_by_class):
        palette = sns.color_palette("deep")
        self._class_report(mean_set_sizes_by_class,
                           'mean_set_sizes_by_class',
                           'Mean set size',
                           palette[1])

    def visualize_mean_fnrs_by_class(self,
                                     mean_fnrs_by_class):
        palette = sns.color_palette("deep")
        self._class_report(mean_fnrs_by_class,
                           'mean_fnrs_by_class',
                           'Mean FNR',
                           palette[0])

    def visualize_mean_model_fnrs_by_class(self,
                                           mean_fnrs_by_class):
        palette = sns.color_palette("deep")
        self._class_report(mean_fnrs_by_class,
                           'mean_model_fnrs_by_class',
                           'Mean model FNR',
                           palette[2])

    def report_class_statistics(self,
                                mean_set_sizes_by_class,
                                mean_fnrs_by_class,
                                mean_model_fnrs_by_class=None):
        self.visualize_mean_fnrs_by_class(mean_fnrs_by_class)
        self.visualize_mean_set_sizes_by_class(mean_set_sizes_by_class)
        if mean_model_fnrs_by_class:
            self.visualize_mean_model_fnrs_by_class(mean_model_fnrs_by_class)
