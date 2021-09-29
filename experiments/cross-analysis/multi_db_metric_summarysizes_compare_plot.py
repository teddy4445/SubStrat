# library imports
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt

# project imports
from ds.table import Table


class CompareSummaryAlgorithmsPlotter:
    """
    Produce an analysis plot comparing the performance of the algorithms on several
    datasets, metrics, and summary sizes
    """

    # CONSTS #
    DATA_FOLDERS_PREFIX = "multi_db_multi_metric_initial_results_"
    DATA_FILE_POSTFIX = "_scores.csv"
    RESULTS_FOLDER_NAME = "meta_analysis_results"
    COLORS = ["black", "green", "blue", "red", "orange", "yellow"]
    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def generate():
        """
        Single entry point to the class, running the entire logic.
        1. gather the data from the files into a object that it easy to work with
        2. generate the plots and save them
        """
        # prepare object to work with
        data = {}

        # load data
        main_path = os.path.dirname(os.path.dirname(__file__))
        for data_folder_path in glob(os.path.join(main_path, "{}*".format(CompareSummaryAlgorithmsPlotter.DATA_FOLDERS_PREFIX))):
            for data_file_path in glob(os.path.join(data_folder_path, "*{}".format(CompareSummaryAlgorithmsPlotter.DATA_FILE_POSTFIX))):
                # read data from file
                df = pd.read_csv(data_file_path)
                # get algo name
                algo_name = os.path.basename(data_folder_path)[len(CompareSummaryAlgorithmsPlotter.DATA_FOLDERS_PREFIX):]
                # get summary size
                summary_size = os.path.basename(data_file_path).split("_")[0]
                # just make sure we have dict for this summary size
                if summary_size not in data:
                    data[summary_size] = {}
                data[summary_size][algo_name] = df

        # for easy work - get the datasets names and metrics
        first_algo_dict = data[list(data.keys())[0]]
        first_df = first_algo_dict[list(first_algo_dict.keys())[0]]
        summary_sizes = list(data.keys())
        algos = list(data[list(data.keys())[0]].keys())
        metrics = list(first_df)[1:]
        datasets = ["db_{}".format(1+index) for index in range(len(list(first_df.iloc[:, 0])))]

        # add folder for the meta analysis results
        try:
            os.mkdir(os.path.join(main_path, CompareSummaryAlgorithmsPlotter.RESULTS_FOLDER_NAME))
        except Exception as error:
            pass

        # calc sizes for the figure prints
        width = 0.8 / len(algos)

        # generate plot
        fig, axes = plt.subplots(len(metrics), len(data), figsize=(6 * len(metrics), 6 * len(data)))  # subplots as the number of summary sizes
        for summary_index, (summary_size, algo_dicts) in enumerate(data.items()):
            # init summary table to save
            summary_table = Table(columns=algos,
                                  rows_ids=metrics)
            # make one metric graph for all the algos and DBs
            for metric_index, metric_name in enumerate(metrics):
                for algo_index, algo in enumerate(algos):
                    # plot bar
                    axes[summary_index, metric_index].bar([val + (algo_index - len(algos)/2) * width for val in range(len(datasets))],
                                                          list(algo_dicts[algo][metric_name]),
                                                          color=CompareSummaryAlgorithmsPlotter.COLORS[algo_index],
                                                          width=width,
                                                          alpha=0.75,
                                                          label=algo.replace("_", " ").title())
                    # the mean line
                    mean_db_val = np.mean(list(algo_dicts[algo][metric_name]))
                    std_db_val = np.std(list(algo_dicts[algo][metric_name]))
                    axes[summary_index, metric_index].plot([- len(algos)/2 * width, len(datasets) - 1 + len(algos)/2 * width],
                                                           [mean_db_val, mean_db_val],
                                                           "--",
                                                           linewidth=2,
                                                           color=CompareSummaryAlgorithmsPlotter.COLORS[algo_index])
                    summary_table.add(column=algo,
                                      row_id=metric_name,
                                      data_point="{} +- {}".format(mean_db_val, std_db_val))
                axes[summary_index, metric_index].set_xlabel("Datasets")
                axes[summary_index, metric_index].set_ylabel("Error '{}' [1]".format(metric_name.replace("_", " ")))
                axes[summary_index, metric_index].set_xticks(range(len(datasets)))
                axes[summary_index, metric_index].set_xticklabels(datasets, rotation=45)
                axes[summary_index, metric_index].legend(loc="upper left")
            # save summary table to file
            summary_table.to_csv(save_path=os.path.join(main_path, CompareSummaryAlgorithmsPlotter.RESULTS_FOLDER_NAME, "score_compare_{}.csv".format(summary_size)))

        ROW_LABEL_PAD = 5
        for ax, row in zip(axes[:, 0], summary_sizes):
            ax.annotate("Summary size\n{}".format(row),
                        xy=(0, 0.5),
                        xytext=(-ax.yaxis.labelpad - ROW_LABEL_PAD, 0),
                        xycoords=ax.yaxis.label,
                        textcoords='offset points',
                        size='large',
                        ha='right',
                        va='center')
        # save plot
        plt.savefig(os.path.join(main_path, CompareSummaryAlgorithmsPlotter.RESULTS_FOLDER_NAME, "score_compare.png"))
        plt.close()


if __name__ == '__main__':
    CompareSummaryAlgorithmsPlotter.generate()
