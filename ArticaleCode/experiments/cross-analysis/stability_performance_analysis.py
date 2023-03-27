# library imports
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt


class StabilityPerformanceAnalysis:
    """
    This class analyze the generates the data for performance and stability connection in different summarization algorithms
    """

    # CONSTS #
    RESULTS_FOLDER_NAME = "meta_analysis_results"
    RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULTS_FOLDER_NAME)
    MARKERS = ["o", "^", "P", "s", "*", "+", "X", "D", "d"]
    COLORS = ["black", "blue", "red", "green", "yellow", "purple", "orange", "gray"]

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def single_compare(performance_file_path: str,
                       stability_file_path: str):
        """
        Visualizing the connection between performance and stability for a single algorithm and summary size
        :param performance_file_path: the performance data file path
        :param stability_file_path: the stability data file path
        :return: none - save a plot to a file
        """
        # add folder for the meta analysis results
        try:
            os.mkdir(StabilityPerformanceAnalysis.RESULTS_PATH)
        except Exception as error:
            pass
        # read data
        performance_df = pd.read_csv(performance_file_path)
        stability_df = pd.read_csv(stability_file_path)
        # convert to something easy to work with
        metrics = list(performance_df)[1:]
        datasets = ["db_{}".format(1+index) for index in range(len(list(performance_df.iloc[:, 0])))]
        for metric_index, metric in enumerate(metrics):
            for dataset_index, dataset in enumerate(datasets):
                plt.scatter(list(performance_df[metric])[dataset_index],
                            list(stability_df[metric])[dataset_index],
                            s=20,
                            marker=StabilityPerformanceAnalysis.MARKERS[metric_index],
                            color=StabilityPerformanceAnalysis.COLORS[dataset_index])
        plt.xlabel("Performance")
        plt.ylabel("Stability")
        plt.savefig(os.path.join(StabilityPerformanceAnalysis.RESULTS_PATH,
                                 "{}-{}.png".format(os.path.basename(os.path.dirname(performance_file_path).replace("multi_db_multi_metric_initial_results_", "")),
                                                    os.path.basename(performance_file_path).split("_")[0])))
        plt.close()


if __name__ == '__main__':
    StabilityPerformanceAnalysis.single_compare(
        performance_file_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           "multi_db_multi_metric_initial_results_genetic",
                                           "10X3_scores.csv"),
        stability_file_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                         "meta_analysis_results",
                                         "10X3_genetic_stability.csv"))
