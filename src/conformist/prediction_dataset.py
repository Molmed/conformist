import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from upsetplot import plot

from .output_dir import OutputDir

plt.rcParams.update({'font.size': 10})


class PredictionDataset(OutputDir):
    DATASET_NAME_COL = 'dataset'
    ID_COL = 'id'
    KNOWN_CLASS_COL = 'known_class'
    PREDICTED_CLASS_COL = 'predicted_class'
    MELTED_KNOWN_CLASS_COL = 'melted_known_class'

    def __init__(self,
                 df=None,
                 predictions_csv=None,
                 dataset_col_name=None,
                 dataset_name=None,
                 display_classes=None):
        self.output_dir = None

        if df is None and predictions_csv is None:
            raise ValueError('Either df or predictions_csv must be provided')

        if df is None:
            df = pd.read_csv(predictions_csv)

        self.df = df

        # Fill na's with 0s for prediction columns only
        self.df[self.class_names()] = self.df[self.class_names()].fillna(0)
        if dataset_col_name:
            self._create_dataset_column_from_col(dataset_col_name)
        elif dataset_name:
            self._create_dataset_column_from_name(dataset_name)
        self._parse_predictions()

        self.display_classes = display_classes

    def _create_dataset_column(self):
        # make dataset the first column
        self.df = self.df[[self.DATASET_NAME_COL] +
                          [col for col in self.df.columns
                           if col != self.DATASET_NAME_COL]]

        # De-fragment
        self.df = self.df.copy()

    def _create_dataset_column_from_col(self, col_name):
        self.df[self.DATASET_NAME_COL] = self.df[col_name]
        self._create_dataset_column()

    def _create_dataset_column_from_name(self, name):
        self.df[self.DATASET_NAME_COL] = name
        self._create_dataset_column()

    def _parse_predictions(self):
        self.smx = self.df[self.class_names()].values

        # Replace 1.0 with 0.99999
        self.smx[self.smx == 1.0] = 1 - (1e-10)

        self.predictions_str = self.df[
            PredictionDataset.PREDICTED_CLASS_COL].values

        # Replace nans with empty strings
        self.predictions_str = np.where(pd.isnull(self.predictions_str),
                                        '', self.predictions_str)

        predicted_classes = [classes.split(',') for classes
                             in self.predictions_str]

        self.predictions_bool = np.array(
            [[class_name in labels for class_name in self.class_names()]
             for labels in predicted_classes])

        self.labels_str = None
        self.labels_idx = None
        if self.KNOWN_CLASS_COL not in self.df.columns:
            return

        self.labels_str = self.df[
            PredictionDataset.KNOWN_CLASS_COL].values

        class_names = self.class_names()
        known_classes = [classes.split(',') for classes
                         in self.df[self.KNOWN_CLASS_COL].values]

        indices_list = [[class_names.index(class_name)
                         for class_name in class_list if class_name in
                         class_names] for class_list in known_classes]

        # convert the list of indices to a binary list
        self.labels_idx = np.array([[1 if i in indices else 0
                                    for i in range(len(self.class_names()))]
                                    for indices in indices_list])

    def append_dataset(self, other):
        self.df = pd.concat([self.df, other.df])

    def export(self, path):
        self.df.to_csv(path, index=False)

    # Create a new df creating a new record for every known class
    def melt(self, primary_class_only=False):
        # Take KNOWN_CLASS_COL and split the values by comma into a new df with a column for each class
        known_classes_df = self.df[self.KNOWN_CLASS_COL].str.split(',', expand=True)

        # Label each column as known_class_1, known_class_2, etc.
        known_classes_df.columns = [f'{self.KNOWN_CLASS_COL}_{i+1}' for
                                    i in range(known_classes_df.shape[1])]

        if primary_class_only:
            # Keep only first column
            known_classes_df = known_classes_df.iloc[:, 0:1]

        df = pd.concat([self.df, known_classes_df], axis=1)

        # Use melt to reshape the DataFrame
        new_df = df.melt(id_vars=self.ID_COL,
                         value_vars=known_classes_df.columns,
                         value_name=self.MELTED_KNOWN_CLASS_COL)

        # Drop the 'variable' column and any rows with null 'class' values
        new_df = new_df.drop(columns='variable').dropna(subset=[self.MELTED_KNOWN_CLASS_COL])

        # Re-join it with the original DataFrame
        new_df = new_df.merge(self.df, left_on=self.ID_COL, right_on=self.ID_COL)

        return new_df

    def class_counts(self, translate=False):
        counting_df = self.melt()[self.MELTED_KNOWN_CLASS_COL]
        counts = counting_df.value_counts()
        if translate and self.display_classes:
            return counts.rename(index=self.display_classes)
        # Remove index name
        counts.index.name = None
        return counts

    def class_counts_by_dataset(self, primary_class_only=False):
        counting_df = self.melt(primary_class_only=primary_class_only)
        counts = counting_df.groupby([self.DATASET_NAME_COL, self.MELTED_KNOWN_CLASS_COL]).size()
        return counts

    def translate_class_name(self, class_name):
        if self.display_classes and class_name in self.display_classes:
            return self.display_classes[class_name]
        return class_name

    def class_names(self, translate=False):
        cols_to_exclude = [self.DATASET_NAME_COL, self.ID_COL,
                           self.KNOWN_CLASS_COL, self.PREDICTED_CLASS_COL,
                           self.MELTED_KNOWN_CLASS_COL]

        cols = [col for col in self.df.columns
                if col not in cols_to_exclude]

        if translate and self.display_classes:
            return [self.translate_class_name(col) for col in cols]

        # Return everything that is not in the exclusion list
        return cols

    def run_reports(self,
                    base_output_dir,
                    upset_plot_color='black',
                    min_softmax_threshold=0.5,
                    primary_class_only_in_class_counts=False):
        self.create_output_dir(base_output_dir)
        self.visualize_class_counts()
        self.visualize_class_counts_by_dataset(
            primary_class_only=primary_class_only_in_class_counts)
        self.visualize_prediction_heatmap()
        self.visualize_prediction_stripplot('prediction_stripplot_all')
        self.visualize_prediction_stripplot(
            'prediction_stripplot_min_threshold',
            min_softmax_threshold=min_softmax_threshold)
        self.visualize_model_sets(0.5, upset_plot_color)
        print(f'Reports saved to {self.output_dir}')

    def _class_colors(self):
        # Define a colormap
        colormap = plt.cm.get_cmap('tab20')

        # Create a dictionary to map each class to a color
        classes = self.class_names()
        class_to_color = {
            cls: colormap(i) for i, cls in enumerate(classes)}

        return class_to_color

    def visualize_class_counts(self):
        plt.figure()

        # create a bar chart
        ccs = self.class_counts()

        # Translate if necessary
        if self.display_classes:
            ccs = ccs.rename(index=self.display_classes)

        # Print count above each bar
        for i, v in enumerate(ccs):
            plt.text(i, v, str(v), ha='center', va='bottom')

        ccs.plot.bar()

        # Dump class counts to CSV
        ccs.to_csv(f'{self.output_dir}/class_counts.csv')

        # show the plot
        plt.savefig(f'{self.output_dir}/class_counts.png', bbox_inches='tight')

    def visualize_class_counts_by_dataset(self,
                                          primary_class_only=False):
        plt.figure()

        # create a bar chart
        ccs = self.class_counts_by_dataset(
            primary_class_only=primary_class_only)

        # Create a dictionary to map each class to a color
        class_to_color = self._class_colors()

        # Count how many datasets and create a grid of plots
        num_datasets = len(ccs.index.get_level_values(0).unique())
        fig, axs = plt.subplots(num_datasets, 1, figsize=(10, 2 * num_datasets))

        if num_datasets == 1:
            axs = [axs]

        # Group by the first level of the index (dataset) and count the number of unique classes
        # grouped_ccs = ccs.groupby(level=0).apply(lambda x: x.index.get_level_values(1).nunique())

        # No, not number of classes, but sum of their values
        grouped_ccs = ccs.groupby(level=0).sum()

        # Order datasets by number of unique classes
        ordered_datasets = grouped_ccs.sort_values(ascending=False).index

        # For each dataset, create a bar chart
        for i, dataset in enumerate(ordered_datasets):
            dataset_series = ccs.loc[dataset]

            # Ensure dataset_series is a Series
            if not isinstance(dataset_series, pd.Series):
                raise ValueError(f"Expected ccs.loc[{dataset}] to be a Series")

            sorted_series = dataset_series.sort_values(ascending=False)

            # Get colors for the bars
            bar_colors = [class_to_color[cls] for cls in sorted_series.index]

            # Plot bar chart with fixed width
            bars = axs[i].bar(sorted_series.index,
                              sorted_series.values,
                              width=0.5,
                              color=bar_colors)
            axs[i].set_title(dataset)

            # Padding
            axs[i].margins(y=0.3)

            # Print count above each bar
            for j, v in enumerate(sorted_series):
                if np.isfinite(v):
                    axs[i].text(j, v, str(v), ha='center', va='bottom')

            # Set x-axis labels to class names and rotate them vertically
            # axs[i].set_xticks(range(len(sorted_series.index)))
            # axs[i].set_xticklabels(sorted_series.index, rotation=90)
            # Remove x-axis labels
            axs[i].set_xticks([])
            # Remove xticks
            axs[i].tick_params(axis='x', which='both', bottom=False, top=False)

            # Add legend
            # if i == 0:
            #     axs[i].legend(bars, sorted_series.index, title="Classes")

        title = 'Primary known class' if primary_class_only else 'Known class'

        # Add a custom legend
        legend_handles = [Patch(color=class_to_color[cls], label=cls) for cls in self.class_names()]
        fig.legend(legend_handles,
                   self.class_names(),
                   title=title,
                   loc='lower center',
                   ncol=len(legend_handles)/2,
                   bbox_to_anchor=(0.5, -0.05),  # Adjust position: (x, y)
                   frameon=False  # Remove the frame around the legend
                   )
        fig.subplots_adjust(bottom=0.1)

        # Adjust layout to prevent overlap and add margin under each panel
        plt.tight_layout()
        plt.subplots_adjust(top=1)  # Adjust hspace to add margin

        # show the plot
        plt.savefig(f'{self.output_dir}/class_counts_by_dataset.png',
                    bbox_inches='tight')

    def visualize_prediction_heatmap(self):
        plt.figure(figsize=(10, 8))

        group_by_col = self.MELTED_KNOWN_CLASS_COL
        df = self.melt()

        grouped_df = df.groupby(group_by_col)
        pred_col_names = self.class_names()

        mean_smx = []

        for name, group in grouped_df:
            name = self.translate_class_name(name)
            mean_smx_row = [name]

            for col in pred_col_names:
                mean_smx_row.append(group[col].mean())

            mean_smx.append(mean_smx_row)

        col_names = ['true_class_name'] + self.class_names(translate=True)

        mean_smx_df = pd.DataFrame(mean_smx, columns=col_names)
        mean_smx_df.set_index('true_class_name', inplace=True)

        # Sort the rows and columns
        mean_smx_df.sort_index(axis=0, inplace=True)  # Sort rows
        mean_smx_df.sort_index(axis=1, inplace=True)  # Sort columns

        # Remove any columns where all the rows are 0
        mean_smx_df = mean_smx_df.loc[:, (mean_smx_df != 0).any(axis=0)]

        hm = sns.heatmap(mean_smx_df,
                 cmap="coolwarm",
                 annot=True,
                 fmt='.2f')

        labelpad = 20
        plt.setp(hm.get_yticklabels(), rotation=0)

        hm.set_xlabel('MEAN PROBABILITY SCORE',
                                 weight='bold', labelpad=labelpad)
        hm.set_ylabel('TRUE CLASS',
                                 weight='bold', labelpad=labelpad)

        # Save the plot to a file
        plt.savefig(f'{self.output_dir}/prediction_heatmap.png', bbox_inches='tight')

    def visualize_prediction_stripplot(self,
                                       output_filename_prefix,
                                       min_softmax_threshold=None):
        plt.figure()

        df = self.melt()
        cols = [col for col in df.columns if col in self.class_names()]

        # Create a new df new_df
        # Loop through rows in df. For each row, create a new row in new_df for each class in cols
        # For each row in new_df, add the softmax score for the corresponding class

        new_df = pd.DataFrame(columns=['True class', 'Predicted class', 'Softmax score'])

        rows = []
        for index, row in df.iterrows():
            for col in cols:
                new_row = {
                    'True class': row[self.MELTED_KNOWN_CLASS_COL],
                    'Predicted class': col,
                    'Softmax score': row[col]
                }

                if min_softmax_threshold is None or row[col] >= min_softmax_threshold:
                    rows.append(new_row)

        new_df = pd.concat([new_df, pd.DataFrame(rows)], ignore_index=True)

        # Increase the height of each row by adjusting the figure size
        num_classes = new_df['True class'].nunique()
        plt.figure(figsize=(10, num_classes * 1))  # Adjust the height multiplier as needed

        sns.stripplot(data=new_df,
                      x='Softmax score',
                      y='True class',
                        hue='Predicted class',
                        jitter=0.5,
                        alpha=0.75,
                        dodge=True,
                        palette=self._class_colors(),
                        size=4)

        # Create custom legend handles
        class_to_color = self._class_colors()
        legend_handles = [Patch(color=class_to_color[cls], label=cls) for cls in new_df['Predicted class'].unique()]

        # Position the legend to the right of the plot with bars instead of dots
        plt.legend(handles=legend_handles, title="Predicted Classes", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        # Save the plot to a file
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(f'{self.output_dir}/{output_filename_prefix}.png', bbox_inches='tight')

    def visualize_model_sets(self, min_softmax_threshold=0.5, color="black"):
        plt.figure()

        df = self.melt()
        cols = [col for col in df.columns if col in self.class_names()]

        new_df = pd.DataFrame(columns=cols)

        rows = []
        for index, row in df.iterrows():
            new_row = {}
            for col in cols:
                new_row[col] = (row[col] >= min_softmax_threshold)
            rows.append(new_row)

        upset_data = pd.concat([new_df, pd.DataFrame(rows)], ignore_index=True)
        # Set a multi-index
        upset_data.set_index(upset_data.columns.tolist(), inplace=True)

        # upset = UpSet(upset_data)
        # upset.add_catplot("violin")
        plot(upset_data,
             sort_by="cardinality",
             facecolor=color,
             show_counts="%d",
             show_percentages="{:.0%}",
             orientation='horizontal')
        plt.savefig(f'{self.output_dir}/upset.png', bbox_inches='tight')

    def prediction_sets_df(self, prediction_sets, export_to_dir=None):
        # Make a copy of the DataFrame
        df = self.df.copy()

        # Add the prediction sets to the DataFrame
        df['prediction_sets'] = prediction_sets

        has_known_class = self.KNOWN_CLASS_COL in df.columns

        if has_known_class:
            # Get the known class names of the prediction set members
            def process_kc_row(row):
                classes = []
                for col in str(row[self.KNOWN_CLASS_COL]).split(','):
                    if col in row:
                        classes.append(str(row[col]))
                return ','.join(classes)

            df['known_class_softmax_scores'] = df.apply(
                lambda row: process_kc_row(row), axis=1)

        # Get the softmax scores of the prediction set members
        def process_row(row):
            scores = []
            if row['prediction_sets'] is None:
                return ''
            for col in str(row['prediction_sets']).split(','):
                if col in row:
                    scores.append(str(row[col]))
            return ','.join(scores)

        df['prediction_set_softmax_scores'] = df.apply(
            lambda row: process_row(row), axis=1)

        cols_to_keep = [self.DATASET_NAME_COL, self.ID_COL,
                        self.PREDICTED_CLASS_COL,
                        'prediction_sets']

        if has_known_class:
            cols_to_keep.append(self.KNOWN_CLASS_COL)

        cols_to_keep.append('prediction_set_softmax_scores')

        if has_known_class:
            cols_to_keep.append('known_class_softmax_scores')

        df = df[cols_to_keep]

        return df

