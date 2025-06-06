import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc
from . import OutputDir, PredictionDataset


class ROC(OutputDir):
    FIGURE_FONTSIZE = 12
    FIGURE_WIDTH = 12
    FIGURE_HEIGHT = 8
    plt.rcParams.update({'font.size': FIGURE_FONTSIZE})

    def __init__(self,
                 prediction_dataset: PredictionDataset,
                 cop_class,
                 base_output_dir,
                 n_runs_per_alpha=1000,
                 n_alphas=50,
                 min_alpha=0.05,
                 max_alpha=0.95,
                 colors=['purple', 'green']
                 ):
        self.create_output_dir(base_output_dir)
        self.cop_class = cop_class
        self.prediction_dataset = prediction_dataset
        self.n_runs_per_alpha = n_runs_per_alpha
        self.n_alphas = n_alphas
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.colors = colors

        self.cp_tpr_list = []
        self.cp_fpr_list = []
        self.model_tpr_list = []
        self.model_fpr_list = []

    def uncalibrated_model_rates(self, alphas):
        # Compute TPR and FPR for the uncalibrated model
        model_tpr = []
        model_fpr = []

        for alpha in alphas:
            prediction_sets = self.prediction_dataset.smx >= alpha

            # TPR calculation
            tps = ((prediction_sets * self.prediction_dataset.labels_idx).sum(axis=1))
            fns = ((~prediction_sets * self.prediction_dataset.labels_idx).sum(axis=1))

            # Get sums of tps and fns
            tps = tps.sum()
            fns = fns.sum()
            tpr = tps / (tps + fns)

            model_tpr.append(tpr)

            # FPS calculation
            fps = ((prediction_sets * (1 - self.prediction_dataset.labels_idx)).sum(axis=1))
            tns = np.array([
                len(np.setdiff1d(labels, predictions))
                for labels, predictions in zip(self.prediction_dataset.labels_idx, prediction_sets)
            ])

            # Get sums of tns and fps
            tns = tns.sum()
            fps = fps.sum()
            fpr = fps / (fps + tns)

            model_fpr.append(fpr)

        return model_tpr, model_fpr

    def run(self):
        # Define a range of significance levels (alpha values)
        alpha_levels = np.linspace(self.min_alpha,
                                   self.max_alpha,
                                   self.n_alphas)

        self.model_tpr_list, self.model_fpr_list = \
            self.uncalibrated_model_rates(alpha_levels)

        # Compute TPR and FPR for different alpha thresholds
        for alpha in alpha_levels:
            print(f'alpha={alpha}')
            cop = self.cop_class(self.prediction_dataset, alpha=alpha)
            trial = cop.do_validation_trial(n_runs=self.n_runs_per_alpha)

            mean_cp_tpr = trial.mean_true_positive_rate()
            mean_cp_fpr = trial.mean_FPR()

            self.cp_tpr_list.append(mean_cp_tpr)
            self.cp_fpr_list.append(mean_cp_fpr)

        # Ensure x values are sorted in ascending order
        sorted_indices = np.argsort(self.cp_fpr_list)
        self.cp_fpr_list = np.array(self.cp_fpr_list)[sorted_indices]
        self.cp_tpr_list = np.array(self.cp_tpr_list)[sorted_indices]

    def run_reports(self):
        plt.figure(figsize=(self.FIGURE_WIDTH, self.FIGURE_HEIGHT))
        plt.tight_layout()

        # Compute AUC
        roc_auc_cp = auc(self.cp_fpr_list, self.cp_tpr_list)
        roc_auc_model = auc(self.model_fpr_list, self.model_tpr_list)

        # Plot ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(self.cp_fpr_list, self.cp_tpr_list, color="blue", lw=2, label=f"CP ROC curve (AUC = {roc_auc_cp:.2f})")
        plt.plot(self.model_fpr_list, self.model_tpr_list, color="red", lw=2, label=f"Model ROC curve (AUC = {roc_auc_model:.2f})")
        plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Diagonal line for reference
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("ROC Curve for Conformal Predictor and Model")
        plt.legend(loc="lower right")
        plt.grid(True)

        # Save the plot to a file
        plt.savefig(f'{self.output_dir}/ROC.png')
        print(f'Reports saved to {self.output_dir}')
