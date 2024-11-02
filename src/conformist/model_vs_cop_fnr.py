import matplotlib.pyplot as plt
import seaborn as sns
from . import OutputDir, PredictionDataset


class ModelVsCopFNR(OutputDir):
    def __init__(self,
                 prediction_dataset: PredictionDataset,
                 cop_class,
                 base_output_dir,
                 n_runs_per_alpha=10000,
                 alphas=[0.1, 0.2, 0.3],
                 colors=['blue', 'purple', 'green']
                 ):
        self.create_output_dir(base_output_dir)
        self.cop_class = cop_class
        self.prediction_dataset = prediction_dataset
        self.n_runs_per_alpha = n_runs_per_alpha
        self.alphas = alphas
        self.colors = colors

    def run(self):
        def do_round(x):
            return round(x, 3)

        self.trial_results = {}

        for alpha in self.alphas:
            cop = self.cop_class(self.prediction_dataset, alpha=alpha)
            trial = cop.do_validation_trial(n_runs=self.n_runs_per_alpha)

            mean_cp_fnr = do_round(trial.mean_false_negative_rate())
            mean_model_fnr = do_round(trial.mean_model_false_negative_rate())

            mean_cp_tpr = do_round(trial.mean_true_positive_rate())
            mean_model_tpr = do_round(trial.mean_model_true_positive_rate())

            cp_fnrs = []
            cp_tprs = []

            model_fnrs = []
            model_tprs = []

            for run in trial.runs:
                cp_fnrs.append(run.false_negative_rate())
                cp_tprs.append(run.true_positive_rate())
                model_fnrs.append(run.model_false_negative_rate())
                model_tprs.append(run.model_true_positive_rate())

            self.trial_results[alpha] = {'mean_cp_fnr': mean_cp_fnr,
                                         'mean_model_fnr': mean_model_fnr,
                                         'mean_cp_tpr': mean_cp_tpr,
                                         'mean_model_tpr': mean_model_tpr,
                                         'cp_fnrs': cp_fnrs,
                                         'cp_tprs': cp_tprs,
                                         'model_fnrs': model_fnrs,
                                         'model_tprs': model_tprs}

    def run_reports(self):
        plt.figure()

        model_rates = []
        model_mean = 0
        i = 0

        for alpha in self.alphas:
            color = self.colors[i]
            tr = self.trial_results[alpha]
            cp_rates = tr['cp_fnrs']
            model_rates = tr['model_fnrs']
            cp_mean = tr['mean_cp_fnr']
            model_mean = tr['mean_model_fnr']

            # Create a KDE plot of cp_fnrs and model_fnrs
            sns.kdeplot(cp_rates, label=f'CP, α={alpha}', color=color)

            # Draw lines for mean_cp_fnr and mean_model_fnr
            plt.axvline(cp_mean, color=color, linestyle='dashed', linewidth=2)

            # Add text labels for mean_cp_fnr and mean_model_fnr
            plt.text(cp_mean - 0.01, plt.ylim()[1]*0.95, f'x\u0304={cp_mean}', color=color, ha='right', weight='bold')
            i += 1

        sns.kdeplot(model_rates, label='Model', color='orange')
        plt.axvline(model_mean, color='orange', linestyle='dashed', linewidth=2)
        plt.text(model_mean + 0.01, plt.ylim()[1]*0.85, f'x\u0304={model_mean}', color='orange', ha='left', weight='bold')

        # Add labels and legend
        plt.xlabel('False Negative Rates')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')

        # Save the plot to a file
        plt.savefig(f'{self.output_dir}/model_vs_CoP_FNR.png', bbox_inches='tight')
        print(f'Reports saved to {self.output_dir}')
