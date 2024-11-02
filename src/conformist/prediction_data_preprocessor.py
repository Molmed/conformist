import pandas as pd
from . import OutputDir, PredictionDataset


class PredictionDataPreprocessor(OutputDir):
    def __init__(self,
                 model_name,
                 predictions_csv,
                 dataset_name,
                 id_col,
                 predicted_class_col,
                 known_classes_df=None,
                 csv_delimiter=','):

        self.model_name = model_name
        self.df_raw = pd.read_csv(predictions_csv, delimiter=csv_delimiter)
        self.predicted_class_col = predicted_class_col
        self.set_name(dataset_name)

        # Start building the processed df
        self.df = pd.DataFrame()
        self.df[PredictionDataset.ID_COL] = self.df_raw[id_col]
        self.df[PredictionDataset.DATASET_NAME_COL] = dataset_name

        self.df[PredictionDataset.PREDICTED_CLASS_COL] = \
            self.df_raw[predicted_class_col]

        if known_classes_df is not None:
            subframe = known_classes_df[[PredictionDataset.KNOWN_CLASS_COL,
                                         PredictionDataset.ID_COL]]
            self.df = pd.merge(self.df,
                               subframe,
                               on=PredictionDataset.ID_COL)

    def set_name(self, dataset_name):
        self.dataset_name = dataset_name
        self.output_file = f'{self.model_name}_predictions_{dataset_name}.csv'

    def save(self, base_output_dir):
        self.create_output_dir(base_output_dir, self.dataset_name)
        path = f'{self.output_dir}/{self.output_file}'
        self.df.to_csv(path, index=False)
        print(f"Saved to {path}")

    def export(self, dst):
        super().export(self.output_file, dst)

    def append_dataset(self, other, merged_name=None):
        if merged_name:
            self.set_name(merged_name)
        self.df = pd.concat([self.df, other.df])
