# library import
import os
import random
import numpy as np
import pandas as pd
from glob import glob
from pandas.core.dtypes.common import is_numeric_dtype

# project import
from ds.table import Table
from ds.converge_report import ConvergeReport
from methods.summary_wellness_scores import SummaryWellnessScores
from plots.analysis_converge_process import AnalysisConvergeProcess
from summary_algorithms.greedy_summary_algorithm import GreedySummary
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

    # PERFORMANCE TEST FACTORS
    REPREAT_PERFORMANCE = 5

    # IO VALUES #
    RESULT_FOLDER_NAME = "performance_stability_results"
    RESULT_PATH = os.path.join(os.path.dirname(__file__), RESULT_FOLDER_NAME)
    # END - IO VALUES #

    def __init__(self):
        pass

    @staticmethod
    def run():
        """
        Single entry point, writing the results to csvs
        """
        # make sure we have the folder for the answers
        try:
            os.mkdir(PerformanceStabilityConnection.RESULT_PATH)
        except Exception as error:
            pass
        algo_table = Table(columns=["algo", "metric", "dataset", "mean_performance", "std_performance", "mean_stability", "std_stability"])




if __name__ == '__main__':
    PerformanceStabilityConnection.run()
