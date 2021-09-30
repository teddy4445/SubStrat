# library import
import os
import numpy as np
import pandas as pd
from glob import glob
from time import time

# project import
from ds.table import Table
from methods.summary_wellness_scores import SummaryWellnessScores
from summary_algorithms.las_vegas_summary_algorithm import LasVegasSummary
from experiments.stability_experiment import StabilityExperiment, prepare_dataset
from summary_algorithms.genetic_algorithm_summary_algorithm import GeneticSummary
from summary_algorithms.combined_greedy_summary_algorithm import CombinedGreedySummary


class PerformanceStabilityConnection:
    """
    Run over the stability of an algorithm in finding the best summary and than learning summaries
    stability in a algo-free context
    """

    DATASETS = {os.path.basename(path).replace(".csv", ""): prepare_dataset(pd.read_csv(path))
                for path in glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*.csv"))}

    METRICS = {
        "mean_entropy": SummaryWellnessScores.mean_entropy,
        "coefficient_of_anomaly": SummaryWellnessScores.coefficient_of_anomaly,
        "coefficient_of_variation": SummaryWellnessScores.coefficient_of_variation,
        "mean_pearson_corr": SummaryWellnessScores.mean_pearson_corr
    }

    ALGOS = {
        "combined_greedy": CombinedGreedySummary,
        "genetic": GeneticSummary,
        "las_vegas": LasVegasSummary
    }

    # STABILITY TEST FACTORS
    NOISE_FUNC = StabilityExperiment.add_dataset_subset_pick_noise
    NOISE_FACTOR = 0.05
    REPEAT_NOISE = 3
    REPEAT_START_CONDITION = 3
    REPEAT_SUMMARY = 10

    # PERFORMANCE TEST FACTORS
    REPEAT_PERFORMANCE = 5
    MAX_ITER = 30
    SUMMARY_ROW_SIZE = 20
    SUMMARY_COL_SIZE = 5

    # IO VALUES #
    RESULT_FOLDER_NAME = "performance_stability_results"
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
            os.mkdir(PerformanceStabilityConnection.RESULT_PATH)
        except Exception as error:
            pass
        algo_table = Table(
            columns=["algo", "metric", "dataset", "mean_performance", "std_performance", "mean_stability",
                     "std_stability"],
            rows_ids=[str(i) for i in range(len(PerformanceStabilityConnection.DATASETS) *
                                            len(PerformanceStabilityConnection.ALGOS) *
                                            len(PerformanceStabilityConnection.METRICS))])

        running_index = 0
        start_exp_time = time()
        for algo_name, algo in PerformanceStabilityConnection.ALGOS.items():
            for metric_name, metric in PerformanceStabilityConnection.METRICS.items():
                for dataset_name, dataset in PerformanceStabilityConnection.DATASETS.items():
                    print("Working on algo = {}, metric = {}, dataset = {}, finised {} iteration during {} seconds".format(
                        algo_name,
                        metric_name,
                        dataset_name,
                        running_index,
                        time()-start_exp_time))
                    # run the performance metric
                    mean_performance, std_performance = self._performance_test(
                        metric_name=metric_name,
                        dataset_name=dataset_name,
                        dataset=dataset,
                        algo=algo,
                        metric=metric,
                        desired_row_size=PerformanceStabilityConnection.SUMMARY_ROW_SIZE,
                        desired_col_size=PerformanceStabilityConnection.SUMMARY_COL_SIZE,
                        repeat=PerformanceStabilityConnection.REPEAT_PERFORMANCE,
                        max_iter=PerformanceStabilityConnection.MAX_ITER)
                    # run the stability metric
                    mean_stability, std_stability = self._stability_test(dataset=dataset,
                                                                         algo=algo,
                                                                         metric=metric,
                                                                         desired_row_size=PerformanceStabilityConnection.SUMMARY_ROW_SIZE,
                                                                         desired_col_size=PerformanceStabilityConnection.SUMMARY_COL_SIZE,
                                                                         noise_function=PerformanceStabilityConnection.NOISE_FUNC,
                                                                         noise=PerformanceStabilityConnection.NOISE_FACTOR,
                                                                         repeats=PerformanceStabilityConnection.REPEAT_NOISE,
                                                                         start_condition_repeat=PerformanceStabilityConnection.REPEAT_START_CONDITION,
                                                                         max_iter=PerformanceStabilityConnection.MAX_ITER)
                    # save the result in the table
                    algo_table.add_row(row_id=str(running_index),
                                       data={"algo": algo_name,
                                             "metric": metric_name,
                                             "dataset": dataset_name,
                                             "mean_performance": mean_performance,
                                             "std_performance": std_performance,
                                             "mean_stability": mean_stability,
                                             "std_stability": std_stability})
                    # count this, go to the next row
                    running_index += 1
                    # move table to file so even a break in some iteration we have the file ready up to this point
                    algo_table.to_csv(save_path=os.path.join(PerformanceStabilityConnection.RESULT_PATH,
                                                             "algo_table.csv"))

        running_index = 0  # resent row counter
        summary_table = Table(columns=["metric", "dataset", "performance", "stability"],
                              rows_ids=[str(i) for i in range(len(self.summaries))])
        for summary, metric_name, dataset_name in self.summaries:
            # calc the properties of the summary
            performance, stability = self._summary_stability_test(
                summary=summary,
                dataset=PerformanceStabilityConnection.DATASETS[dataset_name],
                metric=PerformanceStabilityConnection.METRICS[metric_name],
                noise_function=PerformanceStabilityConnection.NOISE_FUNC,
                noise=PerformanceStabilityConnection.NOISE_FACTOR,
                repeats=PerformanceStabilityConnection.REPEAT_SUMMARY)
            # add answer to the table
            summary_table.add_row(row_id=str(running_index),
                                  data={"metric": metric_name,
                                        "dataset": dataset_name,
                                        "performance": performance,
                                        "stability": stability})
            # count this, go to the next row
            running_index += 1
            # move table to file so even a break in some iteration we have the file ready up to this point
            summary_table.to_csv(save_path=os.path.join(PerformanceStabilityConnection.RESULT_PATH,
                                                        "summary_table.csv"))

    def _performance_test(self,
                          metric_name: str,
                          dataset_name: str,
                          dataset,
                          algo,
                          metric,
                          desired_row_size: int,
                          desired_col_size: int,
                          repeat: int,
                          max_iter: int):
        scores = []
        for _ in range(repeat):
            # run the summary and obtain result and coverage report
            summary, converge_report = algo.run(dataset=dataset,
                                                desired_row_size=desired_row_size,
                                                desired_col_size=desired_col_size,
                                                evaluate_score_function=metric,
                                                is_return_indexes=False,
                                                max_iter=max_iter)
            # add summary to later analysis
            self.summaries.append((summary, metric_name, dataset_name))
            # recall this result
            scores.append(metric(dataset=dataset, summary=summary))
        return np.nanmean(scores), np.nanstd(scores)

    def _stability_test(self,
                        dataset: pd.DataFrame,
                        algo,
                        metric,
                        noise_function,
                        desired_row_size: int,
                        desired_col_size: int,
                        noise: float,
                        repeats: int,
                        start_condition_repeat: int,
                        max_iter: int):
        # calc baseline to compare with
        base_line_summary, converge_report = algo.run(dataset=dataset,
                                                      desired_row_size=desired_row_size,
                                                      desired_col_size=desired_col_size,
                                                      evaluate_score_function=metric,
                                                      is_return_indexes=False,
                                                      max_iter=max_iter)

        scores = []
        for repeat in range(repeats):
            noised_dataset = noise_function(dataset=dataset,
                                            noise=noise)
            for start_condition in range(start_condition_repeat):
                # calc summary and coverage report
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
                scores.append(stability_score)
        return np.nanmean(scores), np.nanstd(scores)

    def _summary_stability_test(self,
                                summary: pd.DataFrame,
                                dataset: pd.DataFrame,
                                metric,
                                noise_function,
                                noise: float,
                                repeats: int):
        scores = []
        for repeat in range(repeats):
            noised_summary = noise_function(dataset=dataset,
                                            noise=noise)
            try:
                stability_score = abs(metric(dataset=dataset, summary=noised_summary) / metric(dataset=summary, summary=noised_summary))
            except:
                stability_score = abs(metric(dataset=dataset, summary=noised_summary))
            scores.append(stability_score)
        return metric(dataset=dataset, summary=summary), np.nanmean(scores)


if __name__ == '__main__':
    exp = PerformanceStabilityConnection()
    exp.run()
