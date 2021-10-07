# library import
import os
import math
import numpy as np
import pandas as pd
from glob import glob
from time import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# project import
from ds.table import Table
from experiments.stability_experiment import prepare_dataset
from methods.summary_wellness_scores import SummaryWellnessScores
from summary_algorithms.las_vegas_summary_algorithm import LasVegasSummary
from ds.dataset_properties_measurements import DatasetPropertiesMeasurements
from summary_algorithms.genetic_algorithm_summary_algorithm import GeneticSummary


class DatasetSummaryPerformanceConnection:
    """
    This class manage the experiment where we aim to find a connection between the dataset, metric and summary
    """

    DATASETS = {os.path.basename(path).replace(".csv", ""): prepare_dataset(pd.read_csv(path))
                for path in glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*.csv"))}

    METRICS = {
        "entropy": SummaryWellnessScores.mean_entropy,
        "coefficient_of_anomaly": SummaryWellnessScores.coefficient_of_anomaly,
        "coefficient_of_variation": SummaryWellnessScores.coefficient_of_variation,
    }

    # STABILITY TEST FACTORS
    NOISE_FUNC = SummaryWellnessScores.add_dataset_subset_pick_noise
    NOISE_FACTOR = 0.05
    REPEAT_NOISE = 3
    REPEAT_START_CONDITION = 3
    REPEAT_SUMMARY = 5

    # ALGORITHM HYPER-PARAMETERS
    MAX_ITER = 25
    SUMMARY_ROW_SIZE = 20
    SUMMARY_COL_SIZE = 5

    # IO VALUES #
    RESULT_FOLDER_NAME = "dataset_summary_performance_connection"
    RESULT_PATH = os.path.join(os.path.dirname(__file__), RESULT_FOLDER_NAME)

    # END - IO VALUES #

    def __init__(self):
        self.summaries = []

    def run(self):
        """
        Single entry point, writing the results to csvs
        """
        # make sure we have the folder for the answers
        try:
            os.mkdir(DatasetSummaryPerformanceConnection.RESULT_PATH)
        except Exception as error:
            pass

        columns = ["ds_{}".format(val) for val in DatasetPropertiesMeasurements.get_columns()]
        columns.append("metric")
        columns.extend(["summary_{}".format(val) for val in DatasetPropertiesMeasurements.get_columns()])
        columns.append("performance")
        columns.append("stability")
        columns.append("l2")
        answer_table = Table(
            columns=columns,
            rows_ids=[str(i) for i in range(len(DatasetSummaryPerformanceConnection.DATASETS) *
                                            len(DatasetSummaryPerformanceConnection.METRICS) *
                                            DatasetSummaryPerformanceConnection.REPEAT_SUMMARY * 2
                                            # the algorithm is stochastic, so get several outcomes
                                            )])
        running_index = 0
        start_exp_time = time()
        for algo_index, algo in enumerate([LasVegasSummary, GeneticSummary]):
            for metric_name, metric in DatasetSummaryPerformanceConnection.METRICS.items():
                for dataset_name, dataset in DatasetSummaryPerformanceConnection.DATASETS.items():
                    run_size = range(DatasetSummaryPerformanceConnection.REPEAT_SUMMARY) if algo_index == 0 else range(
                        1)
                    for _ in run_size:
                        print("Working metric = {}, dataset = {}, finised {} iteration during {} seconds".format(
                            metric_name,
                            dataset_name,
                            running_index,
                            time() - start_exp_time))
                        # run the performance metric
                        summary, performance = self._performance_test(
                            dataset=dataset,
                            algo=GeneticSummary,
                            metric=metric,
                            desired_row_size=DatasetSummaryPerformanceConnection.SUMMARY_ROW_SIZE,
                            desired_col_size=DatasetSummaryPerformanceConnection.SUMMARY_COL_SIZE,
                            max_iter=DatasetSummaryPerformanceConnection.MAX_ITER)
                        # run the stability metric
                        stability = self._stability_test(dataset=dataset,
                                                         algo=GeneticSummary,
                                                         metric=metric,
                                                         desired_row_size=DatasetSummaryPerformanceConnection.SUMMARY_ROW_SIZE,
                                                         desired_col_size=DatasetSummaryPerformanceConnection.SUMMARY_COL_SIZE,
                                                         noise_function=DatasetSummaryPerformanceConnection.NOISE_FUNC,
                                                         noise=DatasetSummaryPerformanceConnection.NOISE_FACTOR,
                                                         max_iter=DatasetSummaryPerformanceConnection.MAX_ITER)
                        data = {"performance": performance,
                                "stability": stability,
                                "metric": metric_name,
                                "l2": math.sqrt(performance**2 + stability**2)}
                        ds_profile = DatasetPropertiesMeasurements.get_dataset_profile(dataset=dataset)
                        for name, value in ds_profile.items():
                            data["ds_{}".format(name)] = value
                        ds_profile = DatasetPropertiesMeasurements.get_dataset_profile(dataset=summary)
                        for name, value in ds_profile.items():
                            data["ds_{}".format(name)] = value
                        # save the result in the table
                        answer_table.add_row(row_id=str(running_index),
                                             data=data)
                        # count this, go to the next row
                        running_index += 1
                        # move table to file so even a break in some iteration we have the file ready up to this point
                        answer_table.to_csv(
                            save_path=os.path.join(DatasetSummaryPerformanceConnection.RESULT_PATH, "answer.csv"))

    def _performance_test(self,
                          dataset,
                          algo,
                          metric,
                          desired_row_size: int,
                          desired_col_size: int,
                          max_iter: int):
        summary, converge_report = algo.run(dataset=dataset,
                                            desired_row_size=desired_row_size,
                                            desired_col_size=desired_col_size,
                                            evaluate_score_function=metric,
                                            is_return_indexes=False,
                                            max_iter=max_iter)
        return summary, metric(dataset=dataset, summary=summary)

    def _stability_test(self,
                        dataset: pd.DataFrame,
                        algo,
                        metric,
                        noise_function,
                        desired_row_size: int,
                        desired_col_size: int,
                        noise: float,
                        max_iter: int):
        # calc baseline to compare with
        base_line_summary, converge_report = algo.run(dataset=dataset,
                                                      desired_row_size=desired_row_size,
                                                      desired_col_size=desired_col_size,
                                                      evaluate_score_function=metric,
                                                      is_return_indexes=False,
                                                      max_iter=max_iter)

        noised_dataset = noise_function(dataset=dataset,
                                        noise=noise)
        summary, converge_report = algo.run(dataset=noised_dataset,
                                            desired_row_size=desired_row_size,
                                            desired_col_size=desired_col_size,
                                            evaluate_score_function=metric,
                                            is_return_indexes=False,
                                            max_iter=max_iter)
        # calc stability score
        try:
            stability_score = abs(
                metric(dataset=base_line_summary, summary=summary) / metric(dataset=dataset,
                                                                            summary=noised_dataset))
        except:
            stability_score = abs(metric(dataset=base_line_summary, summary=summary))
        return stability_score


if __name__ == '__main__':
    exp = DatasetSummaryPerformanceConnection()
    exp.run()
