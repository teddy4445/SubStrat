# library imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class StabilityPerformanceAnalysis:
    """
    This class analyze the connection between performance and stability in different summarization algorithms
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
    def prepare():
        """
        Technical IO preparations
        """
        # add folder for the meta analysis results
        try:
            os.mkdir(StabilityPerformanceAnalysis.RESULTS_PATH)
        except Exception as error:
            pass

    @staticmethod
    def stability_test_file_to_working_format(metrics_files_paths: dict):
        """
        Generating from the "StabilityExperiment" file results structure, the structure we want for this analysis
        """
        # load data
        dfs = {metric_name: pd.read_csv(path) for metric_name, path in metrics_files_paths.items()}

        algo_df = {}

        # save results back to the needed files
        [algo_data.to_csv(os.path.join(StabilityPerformanceAnalysis.RESULTS_PATH, "stability", "{}.csv".format(algo_name)))
         for algo_name, algo_data in algo_df.items()]

    @staticmethod
    def single_compare(performance_file_path: str,
                       stability_file_path: str):
        """
        Visualizing the connection between performance and stability for a single algorithm and summary size
        :param performance_file_path: the performance data file path
        :param stability_file_path: the stability data file path
        :return: none - save a plot to a file
        """
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
                            s=10,
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
