import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from .output_dir import OutputDir
from .prediction_dataset import PredictionDataset


class AlphaSelector(OutputDir):
    def __init__(self,
                 prediction_dataset: PredictionDataset,
                 cop_class,
                 base_output_dir,
                 min_alpha=0.05,
                 max_alpha=0.5,
                 increment_alpha=0.05,
                 n_runs_per_alpha=1000,
                 ):
        self.create_output_dir(base_output_dir)
        self.cop_class = cop_class
        self.prediction_dataset = prediction_dataset
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.increment_alpha = increment_alpha
        self.n_runs_per_alpha = n_runs_per_alpha

        # Initialize lists
        self.alphas = []
        self.trials = {}
        self.mean_set_sizes = []
        self.pcts_empty_sets = []
        self.pcts_singleton_sets = []
        self.pcts_singleton_or_duo_sets = []
        self.pcts_trio_plus_sets = []
        self.mean_false_negative_rates = []
        self.mean_softmax_threshold = []
        self.lamhats = {}
        self.lambdas = {}

    def run(self):
        for a in np.arange(self.min_alpha,
                           self.max_alpha + self.increment_alpha,
                           self.increment_alpha):
            alpha = round(a, 2)
            cop = self.cop_class(self.prediction_dataset, alpha=alpha)
            trial = cop.do_validation_trial(n_runs=self.n_runs_per_alpha)

            # Store values for graphing
            self.alphas.append(alpha)
            self.trials[alpha] = trial
            self.mean_set_sizes.append(trial.mean_set_size())
            self.pcts_empty_sets.append(trial.pct_empty_sets())
            self.pcts_singleton_sets.append(trial.pct_singleton_sets())
            self.pcts_singleton_or_duo_sets.append(
                trial.pct_singleton_or_duo_sets())
            self.pcts_trio_plus_sets.append(trial.pct_trio_plus_sets())
            self.mean_false_negative_rates.append(
                trial.mean_false_negative_rate())
            self.mean_softmax_threshold.append(trial.mean_softmax_threshold())
            self.lamhats[alpha] = trial.mean_softmax_threshold()
            self.lambdas[alpha] = trial.mean_softmax_thresholds()

    def run_reports(self):
        self.visualize()
        self.visualize_lambdas()
        self.save_summary()
        print(f'Reports saved to {self.output_dir}')

    def visualize(self):
        # MEAN SET SIZES GRAPH
        plt.figure()

        data = pd.DataFrame({
            'Alpha': self.alphas,
            'Mean Set Size': self.mean_set_sizes
        })

        sns.lineplot(data=data, x='Alpha', y='Mean Set Size')
        plt.savefig(f'{self.output_dir}/alpha_to_mean_set_size.png')

        # PERCENT EMPTY/SINGLETON SETS GRAPH
        plt.figure()

        # Labels
        x_label = 'Alpha'
        y_label = 'Proportion of prediction sets with size n'
        legend_title = 'Set size'

        # Create a DataFrame for the pct_empty_sets and pct_singleton_sets
        data = pd.DataFrame({
            x_label: self.alphas,
            'n = 0': self.pcts_empty_sets,
            'n ∈ {1, 2}': self.pcts_singleton_or_duo_sets,
            'n ≥ 3': self.pcts_trio_plus_sets
        })

        # Melt the DataFrame to have the set types as a separate column
        data_melted = data.melt(id_vars=x_label,
                                var_name=legend_title,
                                value_name=y_label)

        # Create the bar chart
        sns.barplot(data=data_melted,
                    x=x_label,
                    y=y_label,
                    hue=legend_title,
                    alpha=0.5)

        # Get the current x-tick labels
        labels = [item.get_text() for item in plt.gca().get_xticklabels()]

        # Draw a horizontal line across the top of the highest orange bar
        max_singleton_or_duo_sets = data['n ∈ {1, 2}'].max()
        plt.axhline(y=max_singleton_or_duo_sets,
                    color='#cccccc',
                    linestyle='--')

        # Get the index of the label with the highest value
        idx = data['n ∈ {1, 2}'].idxmax()

        # Make this label bold
        labels[idx] = f'$\\bf{{{labels[idx]}}}$'
        # Set the new x-tick labels
        plt.gca().set_xticklabels(labels)

        # Add a legend
        plt.legend(loc='upper right', title=legend_title)

        # Save the plot to a file
        plt.savefig(f'{self.output_dir}/alpha_to_set_sizes.png')

    def visualize_lambdas(self):
        plt.figure()

        # Only use reasonable alphas
        alphas = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

        SYM_LAMHAT = r'$\hatλ$'
        n_colors = len(alphas)
        # Select n_colors colors from the 'tab10' colormap
        colors = sns.color_palette('tab10', n_colors)

        i = 0
        for a in alphas:
            # Plot lamdas as lineplot with each alpha in different color
            sns.lineplot(x=list(self.lambdas[a].keys()),
                         y=list(self.lambdas[a].values()),
                         color=colors[i],
                         label=f'α={a}')

            # Draw a circle at the lambda[a] value on x and 0 on y
            plt.scatter(self.lamhats[a], 0, s=100, facecolors='none', edgecolors='black')

            padding = 0.05  # adjust this value to get the desired amount of space
            plt.text(self.lamhats[a], 0 + padding,
                     f'{self.lamhats[a]:.2f}',
                     ha='center', va='bottom',
                     fontsize=8, color='black',
                     weight='bold')
            i += 1

        plt.legend(title='Error rate')

        # Draw a dashed gray line at 0
        plt.axhline(0, color='gray', linestyle='--')

        plt.xlabel('λ')
        plt.ylabel('Constraint function f(λ)')
        plt.title(f'FNR-controlling threshold {SYM_LAMHAT}')

        # Save the plot to a file
        plt.savefig(f'{self.output_dir}/lambdas.png')

    def save_summary(self):
        # Export the statistics to CSV
        # Create a dictionary where the keys are the column names and the values are the lists
        data = {
            'alpha': self.alphas,
            'Mean set size': self.mean_set_sizes,
            '% sets n=0': self.pcts_empty_sets,
            '% sets n={1}': self.pcts_singleton_sets,
            '% sets n={1|2}': self.pcts_singleton_or_duo_sets,
            '% sets n>=3': self.pcts_trio_plus_sets,
            'Mean FNR': self.mean_false_negative_rates,
            'Mean softmax threshold': self.mean_softmax_threshold
        }

        # Create a DataFrame from the dictionary
        df = pd.DataFrame(data)

        # Export the DataFrame to a CSV file
        df.to_csv(f'{self.output_dir}/alpha_selection.csv', index=False)
