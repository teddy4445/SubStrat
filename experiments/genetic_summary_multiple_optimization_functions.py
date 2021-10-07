# library import
import os
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
from summary_algorithms.genetic_algorithm_summary_algorithm import GeneticSummary


class GeneticSummaryMultipleOptimizationMetrics:
    """
    This class manage the experiment where we change the current best (genetic) algorithm's fittness function and aiming
    to study the results
    """

    MARKERS = ["o", "^", "P", "s", "*", "+", "X", "D", "d"]
    COLORS = ["black", "blue", "red", "green", "yellow", "purple", "orange", "gray", "peru", "aqua", "violet", "crimson", "indigo", "lime", "darkolivegreen"]

    DATASETS = {os.path.basename(path).replace(".csv", ""): prepare_dataset(pd.read_csv(path))
                for path in glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*.csv"))}

    METRICS = {
        "entropy": SummaryWellnessScores.mean_entropy,
        "mean_entropy_stability": SummaryWellnessScores.mean_mean_entropy_stability,
        "hmean_entropy_stability": SummaryWellnessScores.hmean_mean_entropy_stability,
        "coefficient_of_anomaly": SummaryWellnessScores.coefficient_of_anomaly,
        "mean_coefficient_of_anomaly_stability": SummaryWellnessScores.mean_coefficient_of_anomaly_stability,
        "hmean_coefficient_of_anomaly_stability": SummaryWellnessScores.hmean_coefficient_of_anomaly_stability,
        "coefficient_of_variation": SummaryWellnessScores.coefficient_of_variation,
        "mean_coefficient_of_variation_stability": SummaryWellnessScores.mean_coefficient_of_variation_stability,
        "hmean_coefficient_of_variation_stability": SummaryWellnessScores.hmean_coefficient_of_variation_stability
    }

    # STABILITY TEST FACTORS
    NOISE_FUNC = SummaryWellnessScores.add_dataset_subset_pick_noise
    NOISE_FACTOR = 0.05
    REPEAT_NOISE = 3
    REPEAT_START_CONDITION = 3
    REPEAT_SUMMARY = 1

    # ALGORITHM HYPER-PARAMETERS
    MAX_ITER = 25
    SUMMARY_ROW_SIZE = 20
    SUMMARY_COL_SIZE = 5

    # IO VALUES #
    RESULT_FOLDER_NAME = "multi_metric_optimization"
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
            os.mkdir(GeneticSummaryMultipleOptimizationMetrics.RESULT_PATH)
        except Exception as error:
            pass
        answer_table = Table(
            columns=["dataset", "metric", "performance", "stability"],
            rows_ids=[str(i) for i in range(len(GeneticSummaryMultipleOptimizationMetrics.DATASETS) *
                                            len(GeneticSummaryMultipleOptimizationMetrics.METRICS) *
                                            GeneticSummaryMultipleOptimizationMetrics.REPEAT_SUMMARY
                                            # the algorithm is stochastic, so get several outcomes
                                            )])
        running_index = 0
        start_exp_time = time()
        for metric_name, metric in GeneticSummaryMultipleOptimizationMetrics.METRICS.items():
            for dataset_name, dataset in GeneticSummaryMultipleOptimizationMetrics.DATASETS.items():
                for i in range(GeneticSummaryMultipleOptimizationMetrics.REPEAT_SUMMARY):
                    print("Working metric = {}, dataset = {}, finised {} iteration during {} seconds".format(
                        metric_name,
                        dataset_name,
                        running_index,
                        time() - start_exp_time))
                    # run the performance metric
                    performance = self._performance_test(
                        metric_name=metric_name,
                        dataset_name=dataset_name,
                        dataset=dataset,
                        algo=GeneticSummary,
                        metric=metric,
                        desired_row_size=GeneticSummaryMultipleOptimizationMetrics.SUMMARY_ROW_SIZE,
                        desired_col_size=GeneticSummaryMultipleOptimizationMetrics.SUMMARY_COL_SIZE,
                        repeat=1,
                        max_iter=GeneticSummaryMultipleOptimizationMetrics.MAX_ITER)
                    # run the stability metric
                    stability = self._stability_test(dataset=dataset,
                                                     algo=GeneticSummary,
                                                     metric=metric,
                                                     desired_row_size=GeneticSummaryMultipleOptimizationMetrics.SUMMARY_ROW_SIZE,
                                                     desired_col_size=GeneticSummaryMultipleOptimizationMetrics.SUMMARY_COL_SIZE,
                                                     noise_function=GeneticSummaryMultipleOptimizationMetrics.NOISE_FUNC,
                                                     noise=GeneticSummaryMultipleOptimizationMetrics.NOISE_FACTOR,
                                                     repeats=GeneticSummaryMultipleOptimizationMetrics.REPEAT_NOISE,
                                                     start_condition_repeat=GeneticSummaryMultipleOptimizationMetrics.REPEAT_START_CONDITION,
                                                     max_iter=GeneticSummaryMultipleOptimizationMetrics.MAX_ITER)
                    # save the result in the table
                    answer_table.add_row(row_id=str(running_index),
                                         data={"metric": metric_name,
                                               "dataset": dataset_name,
                                               "performance": performance,
                                               "stability": stability})
                    # count this, go to the next row
                    running_index += 1
                    # move table to file so even a break in some iteration we have the file ready up to this point
                    answer_table.to_csv(
                        save_path=os.path.join(GeneticSummaryMultipleOptimizationMetrics.RESULT_PATH, "answer.csv"))
        # prepare plot
        answer_df = answer_table.to_dataframe()
        ds_names = list(GeneticSummaryMultipleOptimizationMetrics.DATASETS.keys())
        for row_index, row in answer_df.iterrows():
            plt.scatter(row["performance"],
                        row["stability"],
                        marker="o" if "hmean" in row["metric"] else "^" if "mean" in row["metric"] else "P",
                        color=GeneticSummaryMultipleOptimizationMetrics.COLORS[ds_names.index(row["dataset"])])
        plt.xlabel("Performance [1]")
        plt.xlabel("Stability [1]")
        plt.legend(handels=[mlines.Line2D([], [], color='black', marker='o', markersize=15, linewidth=0, label='Harmonic Mean'),
                            mlines.Line2D([], [], color='black', marker='^', markersize=15, linewidth=0, label='Mean'),
                            mlines.Line2D([], [], color='black', marker='P', markersize=15, linewidth=0, label='Only Performance')])
        plt.grid(alpha=0.1)
        plt.savefig(os.path.join(GeneticSummaryMultipleOptimizationMetrics.RESULT_PATH, "answer_scatter.png"))
        plt.close()

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
        return np.nanmean(scores)

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
        return np.nanmean(scores)


if __name__ == '__main__':
    exp = GeneticSummaryMultipleOptimizationMetrics()
    exp.run()
