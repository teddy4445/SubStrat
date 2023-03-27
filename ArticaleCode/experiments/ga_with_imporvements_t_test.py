# library imports
import os
import math
import random
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime

# project imports
from ds.converge_report import ConvergeReport
from methods.summary_wellness_scores import SummaryWellnessScores
from summary_algorithms.genetic_algorithm_summary_algorithm import GeneticSummary
from summary_algorithms.stabler_genetic_algorithm_summary_algorithm import StablerGeneticSummary


class GAwithImprovmentsTtest:
    """
    Test if the machine learning metric works fine
    """

    # CONSTS #

    ITERS = 500
    GA_MAX_ITER = 30
    METRIC = SummaryWellnessScores.sklearn_model

    NOISE_FACTOR = 0.05
    REPEAT_NOISE = 3
    REPEAT_START_CONDITION = 3
    REPEAT_SUMMARY = 1

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run():
        performance_normal_ga = []
        performance_improved_ga = []
        stability_normal_ga = []
        stability_improved_ga = []

        for sample_index in range(GAwithImprovmentsTtest.ITERS):
            rows_count, cols_count = random.randint(100, 100000), random.randint(4, 40)
            data = np.random.normal(random.randint(-100, 100), random.randint(0, 100), rows_count * cols_count).reshape(
                rows_count, cols_count)

            summary, converge_report = StablerGeneticSummary.run(dataset=data,
                                                                 desired_row_size=round(math.sqrt(rows_count)),
                                                                 desired_col_size=round(cols_count / 2),
                                                                 evaluate_score_function=GAwithImprovmentsTtest.METRIC,
                                                                 is_return_indexes=False,
                                                                 max_iter=GAwithImprovmentsTtest.GA_MAX_ITER)
            performance_improved_ga.append(converge_report.final_total_score())
            stability = GAwithImprovmentsTtest._stability_test(dataset=data,
                                                               desired_row_size=round(math.sqrt(rows_count)),
                                                               desired_col_size=round(cols_count / 2),
                                                               noise=GAwithImprovmentsTtest.NOISE_FACTOR,
                                                               repeats=GAwithImprovmentsTtest.REPEAT_NOISE,
                                                               start_condition_repeat=GAwithImprovmentsTtest.REPEAT_START_CONDITION,
                                                               max_iter=GAwithImprovmentsTtest.GA_MAX_ITER)
            stability_improved_ga.append(stability)

            summary, converge_report = GeneticSummary.run(dataset=data,
                                                          desired_row_size=round(math.sqrt(rows_count)),
                                                          desired_col_size=round(cols_count / 2),
                                                          evaluate_score_function=GAwithImprovmentsTtest.METRIC,
                                                          is_return_indexes=False,
                                                          max_iter=GAwithImprovmentsTtest.GA_MAX_ITER)
            performance_normal_ga.append(converge_report.final_total_score())
            stability = GAwithImprovmentsTtest._stability_test(dataset=data,
                                                               desired_row_size=round(math.sqrt(rows_count)),
                                                               desired_col_size=round(cols_count / 2),
                                                               noise=GAwithImprovmentsTtest.NOISE_FACTOR,
                                                               repeats=GAwithImprovmentsTtest.REPEAT_NOISE,
                                                               start_condition_repeat=GAwithImprovmentsTtest.REPEAT_START_CONDITION,
                                                               max_iter=GAwithImprovmentsTtest.GA_MAX_ITER)
            stability_normal_ga.append(stability)

        combine_normal_ga = [math.sqrt(performance_normal_ga[i] ** 2 + stability_normal_ga[i] ** 2) for i in
                             range(len(performance_normal_ga))]
        combine_improved_ga = [math.sqrt(performance_improved_ga[i] ** 2 + stability_improved_ga[i] ** 2) for i in
                               range(len(performance_improved_ga))]

        performance_p_value = stats.ttest_ind(performance_normal_ga, performance_improved_ga)
        stability_p_value = stats.ttest_ind(stability_normal_ga, stability_improved_ga)
        combine_p_value = stats.ttest_ind(combine_normal_ga, combine_improved_ga)
        print("{}, {}, {}".format(performance_p_value, stability_p_value, combine_p_value))

    @staticmethod
    def _stability_test(dataset: pd.DataFrame,
                        algo,
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
                                                      evaluate_score_function=GAwithImprovmentsTtest.METRIC,
                                                      is_return_indexes=False,
                                                      max_iter=max_iter)

        scores = []
        for repeat in range(repeats):
            noised_dataset = SummaryWellnessScores.add_dataset_subset_pick_noise(dataset=dataset,
                                                                                 noise=noise)
            for start_condition in range(start_condition_repeat):
                # calc summary and coverage report
                summary, converge_report = algo.run(dataset=noised_dataset,
                                                    desired_row_size=desired_row_size,
                                                    desired_col_size=desired_col_size,
                                                    evaluate_score_function=GAwithImprovmentsTtest.METRIC,
                                                    is_return_indexes=False,
                                                    max_iter=max_iter)
                # calc stability score
                try:
                    stability_score = abs(
                        GAwithImprovmentsTtest.METRIC(dataset=base_line_summary,
                                                      summary=summary) / GAwithImprovmentsTtest.METRIC(dataset=dataset,
                                                                                                       summary=noised_dataset))
                except:
                    stability_score = abs(GAwithImprovmentsTtest.METRIC(dataset=base_line_summary, summary=summary))
                scores.append(stability_score)
        return np.nanmean(scores)


if __name__ == '__main__':
    GAwithImprovmentsTtest.run()
