import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.font_manager import FontProperties
import matplotlib.gridspec as gridspec
from upsetplot import plot

from .output_dir import OutputDir


class PredictionDataset(OutputDir):
    DATASET_NAME_COL = 'dataset'
    ID_COL = 'id'
    KNOWN_CLASS_COL = 'known_class'
    PREDICTED_CLASS_COL = 'predicted_class'
    MELTED_KNOWN_CLASS_COL = 'melted_known_class'

    FIGURE_FONTSIZE = 12
    FIGURE_WIDTH = 12
    plt.rcParams.update({'font.size': FIGURE_FONTSIZE})

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
            # For items with multiple known classes,
            # change the first of the known_classes_df to 'multiclass'
            # But first check if there are more than one known class columns
            if known_classes_df.shape[1] > 1:
                known_classes_df[f"{self.KNOWN_CLASS_COL}_1"] = known_classes_df.apply(
                    lambda row: 'multiclass' if row[1] is not None else row[0],
                    axis=1)

                # Drop the other known class columns
                known_classes_df = known_classes_df[[f"{self.KNOWN_CLASS_COL}_1"]]

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

    def create_reports_dir(self, base_output_dir):
        if self.output_dir is None:
            self.create_output_dir(base_output_dir)

    def _class_colors(self, custom_color_palette=None):
        if custom_color_palette:
            return custom_color_palette

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

    def _sort_class_names_by_palette(self, class_names, custom_color_palette):
        if isinstance(custom_color_palette, dict):
            class_order = list(custom_color_palette.keys())
            class_names = sorted(class_names, key=lambda x: class_order.index(x))
        return class_names

    def visualize_class_counts_by_dataset(self,
                                          primary_class_only=False,
                                          custom_color_palette=None,
                                          legend_top_padding=0.125):
        plt.figure()

        # create a bar chart
        ccs = self.class_counts_by_dataset(
            primary_class_only=primary_class_only)

        # Get unique class names from ccs
        class_names = ccs.index.get_level_values(1).unique().sort_values()
        class_names = self._sort_class_names_by_palette(class_names,
                                                        custom_color_palette)

        # Create a dictionary to map each class to a color
        class_to_color = self._class_colors(
            custom_color_palette=custom_color_palette)

        # Count how many datasets and create a grid of plots
        num_datasets = len(ccs.index.get_level_values(0).unique())
        fig, axs = plt.subplots(num_datasets,
                                1,
                                figsize=(self.FIGURE_WIDTH, 2.5 * num_datasets))

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
        legend_handles = [Patch(color=class_to_color[cls], label=cls) for cls in class_names]
        legend = fig.legend(legend_handles,
                            class_names,
                            title=title,
                            loc='lower center',
                            frameon=False,
                            ncol=len(legend_handles)/4,
                            bbox_to_anchor=(0.5, 0-legend_top_padding),  # Adjust position: (x, y)
                            handletextpad=1,  # Increase padding between legend handle and text
                            columnspacing=8  # Increase spacing between columns
                            )
        font_properties = FontProperties(weight='bold')
        legend.get_title().set_font_properties(font_properties)

        fig.subplots_adjust(bottom=0.2)

        # Adjust layout to prevent overlap and add margin under each panel
        plt.tight_layout()
        plt.subplots_adjust(top=1)  # Adjust hspace to add margin

        # show the plot
        plt.savefig(f'{self.output_dir}/class_counts_by_dataset.png',
                    bbox_inches='tight')

    def visualize_prediction_heatmap(self, min_softmax_threshold=0.5):
        fig = plt.figure(figsize=(self.FIGURE_WIDTH, 10))

        # Create two subplots, we will create two separate heatmaps
        # Define a GridSpec with unequal column widths
        # Column 1 is nx as wide as column 2
        width_ratio = 8
        gs = gridspec.GridSpec(1, 3, width_ratios=[width_ratio, 1, 1])
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])

        SET_SIZE_COL_NAME = "mean set size"
        MEAN_FP_COL_NAME = "mean false positive softmax"

        group_by_col = self.MELTED_KNOWN_CLASS_COL
        df = self.melt()
        grouped_df = df.groupby(group_by_col)
        pred_col_names = self.class_names()
        set_size_col_names = [SET_SIZE_COL_NAME]
        mean_fp_col_names = [MEAN_FP_COL_NAME]

        psets = self._get_prediction_sets_by_softmax_threshold(
            min_softmax_threshold)
        # Index psets by df.id
        psets.set_index(df[self.ID_COL], inplace=True)
        # Set new column set_size that contains the number of True values in other cols
        psets['set_size'] = psets.sum(axis=1)
        # Add melted known class to psets
        psets = psets.merge(df[[self.ID_COL, group_by_col]],
                            left_index=True,
                            right_on=self.ID_COL)

        # Create new df with columns class and mean_set_size
        psets_df = psets.groupby(
            self.MELTED_KNOWN_CLASS_COL)['set_size'].mean()

        mean_smx = []
        mean_set_size = []
        mean_fp_smx = []

        for name, group in grouped_df:
            mean_smx_row = [name]
            mean_set_size_row = [name]
            mean_fp_smx_row = [name]

            for col in pred_col_names:
                if col != name:
                    mean_fp_smx_row.append(group[col].mean())
                mean_smx_row.append(group[col].mean())
            mean_set_size_row.append(psets_df[name])

            mean_smx.append(mean_smx_row)
            mean_set_size.append(mean_set_size_row)
            mean_fp_smx.append(mean_fp_smx_row)

        # Iterate over rows in mean_fp_smx
        new_mean_fp_smx = []
        for row in mean_fp_smx:
            # Pop the first element to get the class name
            class_name = row.pop(0)
            # Remove zeroes from row
            row = [x for x in row if x != 0]
            # Get mean of the row
            mean = np.mean(row)
            new_mean_fp_smx.append([class_name, mean])
        mean_fp_smx = new_mean_fp_smx

        col_names_1 = ['true_class_name'] + pred_col_names
        col_names_2 = ['true_class_name'] + set_size_col_names
        col_names_3 = ['true_class_name'] + mean_fp_col_names

        mean_smx_df = pd.DataFrame(mean_smx, columns=col_names_1)
        mean_smx_df.set_index('true_class_name', inplace=True)

        mean_set_size_df = pd.DataFrame(mean_set_size, columns=col_names_2)
        mean_set_size_df.set_index('true_class_name', inplace=True)

        mean_fp_smx_df = pd.DataFrame(mean_fp_smx, columns=col_names_3)
        mean_fp_smx_df.set_index('true_class_name', inplace=True)

        # Sort the rows and columns
        mean_smx_df.sort_index(axis=0, inplace=True)  # Sort rows
        mean_smx_df.sort_index(axis=1, inplace=True)  # Sort columns

        mean_set_size_df.sort_index(axis=0, inplace=True)  # Sort rows
        mean_set_size_df.sort_index(axis=1, inplace=True)  # Sort columns

        mean_fp_smx_df.sort_index(axis=0, inplace=True)  # Sort rows
        mean_fp_smx_df.sort_index(axis=1, inplace=True)  # Sort columns

        # Remove any columns where all the rows are 0
        mean_smx_df = mean_smx_df.loc[:, (mean_smx_df != 0).any(axis=0)]

        hm = sns.heatmap(mean_smx_df,
                 ax=ax1,
                 cmap="coolwarm",
                 annot=True,
                 fmt='.2f',
                 cbar=False)

        labelpad = 20
        plt.setp(hm.get_yticklabels(), rotation=0)

        hm.set_xlabel('MEAN SOFTMAX SCORE',
                                 weight='bold', labelpad=labelpad)
        hm.set_ylabel('TRUE CLASS',
                                 weight='bold', labelpad=labelpad)

        # Create second heatmap for mean set size
        hm2 = sns.heatmap(mean_fp_smx_df,
                    ax=ax2,
                    cmap=sns.light_palette("forestgreen", as_cmap=True),
                    annot=True,
                    fmt='.2f',
                    cbar=False)

        hm2.set_ylabel("")

        # Remove y ticks
        hm2.set_yticks([])

        # Add label
        hm2.set_xlabel("mean softmax FP",
                       weight='bold',
                       rotation=90)

        # Remove x ticks
        hm2.set_xticks([])

        # Create third heatmap for mean set size
        hm3 = sns.heatmap(mean_set_size_df,
                    ax=ax3,
                    cmap=sns.light_palette("purple", as_cmap=True),
                    annot=True,
                    fmt='.2f',
                    cbar=False)

        # Rotate x labels
        plt.setp(hm3.get_xticklabels(), rotation=90)

        # Remove y label
        hm3.set_ylabel('')

        # Position y label to the right of heatmap
        hm3.yaxis.set_label_position("right")

        # Remove y ticks
        hm3.set_yticks([])

        # Remove x label
        hm3.set_xlabel("mean set size\nsoftmax > 0.5",
                       weight='bold',
                       rotation=90)

        # Remove x ticks
        hm3.set_xticks([])

        plt.tight_layout(w_pad=0.1)  # Control padding

        # Save the plot to a file
        plt.savefig(f'{self.output_dir}/prediction_heatmap.png', bbox_inches='tight')

    def softmax_summary(self):
        df = self.melt()
        softmax_cols = [col for col in df.columns if col in self.class_names()]

        true_col_name = 'mean true positive softmax'
        false_col_name = 'mean false positive softmax'

        summary_df = pd.DataFrame(
            columns=[true_col_name, false_col_name])

        # For each col in softmax_cols, calculate the mean softmax score for the true class
        # and the mean softmax score for the false classes
        for col in softmax_cols:
            true_pos = df[df[self.MELTED_KNOWN_CLASS_COL] == col]
            false_pos = df[df[self.MELTED_KNOWN_CLASS_COL] != col]

            # Get mean of all scores in column col in true_pos
            mean_true_pos = true_pos[col].mean()
            mean_false_pos = false_pos[col].mean()

            summary_df.loc[col] = [mean_true_pos, mean_false_pos]

        # Sort the DataFrame by mean true positive softmax
        summary_df = summary_df.sort_values(by=true_col_name,
                                            ascending=False)

        # Name index "Predicted class"
        summary_df.index.name = 'Predicted class'

        # Dump to csv
        summary_df.to_csv(f'{self.output_dir}/softmax_summary.csv')

        # Round all values to 4 decimal places
        summary_df = summary_df.round(4)

        # Pad the decimal places with zeros
        summary_df = summary_df.applymap(lambda x: f'{x:.4f}')

        # Visualize
        plt.figure()
        plt.axis('off')  # Hide axes

        # Create table
        table = plt.table(cellText=summary_df.values,
                          colLabels=summary_df.columns,
                          rowLabels=summary_df.index,
                          loc='left',
                          cellLoc='center',
                          )

        plt.title('Predicted class', weight='bold')

        accent_color = "#d4cbb3"
        # Make font color dark gray
        font_color = '#222222'

        # Change border colors
        for key, cell in table.get_celld().items():
            cell.set_edgecolor(accent_color)  # Set border color
            cell.set_linewidth(1)      # Set border width
            cell.get_text().set_color(font_color)
            # cell.PAD = 0.2

        # Make every other row gray
        for i in range(0, len(summary_df), 2):
            table.get_celld()[(i + 1, 0)].set_facecolor(accent_color)
            table.get_celld()[(i + 1, 1)].set_facecolor(accent_color)

        table.auto_set_font_size(False)
        table.set_fontsize(self.FIGURE_FONTSIZE)
        table.scale(2, 2)  # Scale table size

        plt.savefig(f'{self.output_dir}/softmax_summary.png', bbox_inches='tight')

    def visualize_prediction_stripplot(self,
                                       custom_color_palette=None):
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

                rows.append(new_row)

        new_df = pd.concat([new_df, pd.DataFrame(rows)], ignore_index=True)

        # Increase the height of each row by adjusting the figure size
        num_classes = new_df['True class'].nunique()
        plt.figure(figsize=(self.FIGURE_WIDTH, num_classes * 1))  # Adjust the height multiplier as needed

        ax = plt.gca()
        # Add light gray background to every other row
        for i in range(0, num_classes, 2):
            ax.axhspan(i - 0.5, i + 0.5, facecolor='#eeeeee', alpha=0.5)

        class_names = self._sort_class_names_by_palette(
            new_df['True class'].unique(),
            custom_color_palette)

        sns.stripplot(data=new_df,
                      x='Softmax score',
                      y='True class',
                        hue='Predicted class',
                        jitter=0.5,
                        alpha=0.75,
                        dodge=True,
                        palette=self._class_colors(
                            custom_color_palette=custom_color_palette),
                        size=4,
                        ax=ax,
                        order=class_names)

        # Create custom legend handles
        class_to_color = self._class_colors(
            custom_color_palette=custom_color_palette)

        class_names = self._sort_class_names_by_palette(
            new_df['Predicted class'].unique(),
            custom_color_palette)
        legend_handles = [Patch(color=class_to_color[cls], label=cls) for
                          cls in class_names]

        legend = plt.legend(legend_handles,
                            class_names,
                            title="Predicted class",
                            loc='lower center',
                            frameon=False,
                            ncol=4,
                            bbox_to_anchor=(0.5, -0.2),  # Adjust position: (x, y)
                            handletextpad=1,  # Increase padding between legend handle and text
                            columnspacing=8  # Increase spacing between columns
                            )
        font_properties = FontProperties(weight='bold')
        legend.get_title().set_font_properties(font_properties)

        # Save the plot to a file
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        plt.savefig(f'{self.output_dir}/prediction_stripplot.png',
                    bbox_inches='tight')

    def _get_prediction_sets_by_softmax_threshold(self, min_softmax_threshold=0.5):
        df = self.melt()
        cols = [col for col in df.columns if col in self.class_names()]

        new_df = pd.DataFrame(columns=cols)

        rows = []
        for index, row in df.iterrows():
            new_row = {}
            for col in cols:
                new_row[col] = (row[col] >= min_softmax_threshold)
            rows.append(new_row)

        return pd.concat([new_df, pd.DataFrame(rows)], ignore_index=True)


    def visualize_model_sets(self, min_softmax_threshold=0.5, color="black"):
        plt.figure()
        plt.figure(figsize=(self.FIGURE_WIDTH, 8))
        upset_data = self._get_prediction_sets_by_softmax_threshold(
            min_softmax_threshold)
        # Set a multi-index
        upset_data.set_index(upset_data.columns.tolist(), inplace=True)

        plot(upset_data,
             sort_by="cardinality",
             facecolor=color,
             show_counts="%d",
             show_percentages="{:.0%}",
             orientation='horizontal',
             min_subset_size=3)
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

