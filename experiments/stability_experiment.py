# library imports
import os
import random
import numpy as np
import pandas as pd
from glob import glob
from pandas.core.dtypes.common import is_numeric_dtype

# project imports
from ds.table import Table
from ds.converge_report import ConvergeReport
from summary_algorithms.greedy_summary_algorithm import GreedySummary
from summary_algorithms.las_vegas_summary_algorithm import LasVegasSummary
from summary_algorithms.genetic_algorithm_summary_algorithm import GeneticSummary
from methods.summary_wellness_scores import SummaryWellnessScores
from plots.analysis_converge_process import AnalysisConvergeProcess
from methods.summary_process_score_functions import SummaryProcessScoreFunctions


class StabilityExperiment:
    """
    This class generates a summary table of a summary's algorithm performance over multiple score functions, datasets, and summary sizes
    """

    METRIC = SummaryWellnessScores.mean_pearson_corr
    METRIC_NAME = "mean_pearson_corr"

    def __init__(self):
        pass

    @staticmethod
    def run(datasets: dict,
            algorithms: dict,
            summaries_sizes: list,
            main_save_folder_path: str,
            noise_function,
            max_iter: int = 30):
        """
        Generate multiple tables of: _rows -> dataset, columns ->  summary score metrics, divided by summary sizes
        :param datasets: a dict of datasets names (as keys) and pandas' dataframe (as values)
        :param algorithms: a dict of algorithms one wants to test one vs. the other
        :param summaries_sizes: a list of tuples (with 2 elements) - the number of _rows and the number of columns
        :param main_save_folder_path: the folder name where we wish to write the results in
        :param noise_function: a noise function to use in the experiment
        :param max_iter: the maximum number of iteration we allow to do for one combination of dataset, metric, summary size
        :return: None, save csv files and plots to the folder
        """
        # make sure the save folder is exits
        try:
            os.mkdir(main_save_folder_path)
        except:
            pass

        for summary_size in summaries_sizes:
            desired_row_size = summary_size[0]
            desired_col_size = summary_size[1]

            # prepare the table
            summary_table = Table(columns=[dataset_name for dataset_name in datasets],
                                  rows_ids=[algo_name for algo_name in algorithms])

            # run over all the algorithms
            for dataset_name, df in datasets.items():
                # edge cases in the summary size
                if desired_row_size < 1:
                    desired_row_size = df.shape[0]
                if desired_col_size < 1:
                    desired_col_size = df.shape[1]

                # run over all the algorithms
                for algo_name, algo in algorithms.items():

                    summary_table.add(column=dataset_name,
                                      row_id=algo_name,
                                      data_point=StabilityExperiment._stability_test(dataset_name=dataset_name,
                                                                                     noise_function=noise_function,
                                                                                     algo_name=algo_name,
                                                                                     dataset=df,
                                                                                     summary_row_size=desired_row_size,
                                                                                     summary_col_size=desired_col_size,
                                                                                     summary_algorithm=algo,
                                                                                     noise=0.1,
                                                                                     repeats=3,
                                                                                     start_condition_repeat=1,
                                                                                     max_iter=max_iter,
                                                                                     save_converge_report=os.path.join(
                                                                                         main_save_folder_path,
                                                                                         "{}_{}_{}_{}X{}.json".format(
                                                                                             StabilityExperiment.METRIC_NAME,
                                                                                             algo_name,
                                                                                             dataset_name,
                                                                                             desired_row_size,
                                                                                             desired_col_size))))

            summary_table.to_csv(save_path=os.path.join(main_save_folder_path,
                                                        "summary_table_{}X{}.csv".format(desired_row_size,
                                                                                         desired_col_size)))

    @staticmethod
    def _stability_test(dataset_name: str,
                        noise_function,
                        algo_name: str,
                        dataset: pd.DataFrame,
                        summary_row_size: int,
                        summary_col_size: int,
                        summary_algorithm,
                        noise: float,
                        repeats: int,
                        start_condition_repeat: int,
                        max_iter: int,
                        save_converge_report: str):
        # save path folder
        inner_folder = "{}_{}_{}X{}_to_{}X{}".format(dataset_name,
                                                     algo_name,
                                                     dataset.shape[0],
                                                     dataset.shape[1],
                                                     summary_row_size,
                                                     summary_col_size)
        try:
            os.mkdir(os.path.join(os.path.dirname(save_converge_report), inner_folder))
        except Exception as error:
            pass

        base_line_summary, converge_report = summary_algorithm.run(dataset=dataset,
                                                                   desired_row_size=summary_row_size,
                                                                   desired_col_size=summary_col_size,
                                                                   evaluate_score_function=StabilityExperiment.METRIC,
                                                                   is_return_indexes=False,
                                                                   save_converge_report="",
                                                                   max_iter=max_iter)

        scores = []
        for repeat in range(repeats):
            noised_dataset = noise_function(dataset=dataset,
                                            noise=noise)
            for start_condition in range(start_condition_repeat):
                print("Dataset '{}' and algo '{}' Repeat {}/{} with start condition repeat {}/{}".format(dataset_name,
                                                                                                         algo_name,
                                                                                                         repeat + 1,
                                                                                                         repeats,
                                                                                                         start_condition + 1,
                                                                                                         start_condition_repeat))
                # save path
                save_path = os.path.join(os.path.dirname(save_converge_report), inner_folder,
                                         os.path.basename(save_converge_report).replace(".json",
                                                                                        "_{}_{}.json".format(repeat,
                                                                                                             start_condition)))

                # calc summary and coverage report
                summary, converge_report = summary_algorithm.run(dataset=noised_dataset,
                                                                 desired_row_size=summary_row_size,
                                                                 desired_col_size=summary_col_size,
                                                                 evaluate_score_function=StabilityExperiment.METRIC,
                                                                 is_return_indexes=False,
                                                                 save_converge_report=save_path,
                                                                 max_iter=max_iter)

                # generate and save a score of the metric over iterations plot
                AnalysisConvergeProcess.converge_scores(rows_scores=converge_report["rows_score"],
                                                        cols_scores=converge_report["cols_score"],
                                                        total_scores=converge_report["total_score"],
                                                        save_path=save_path.replace(".json", ".png"),
                                                        y_label="'{}' algorithm's error [1]".format(algo_name))
                # calc stability score
                try:
                    stability_score = abs(StabilityExperiment.METRIC(dataset=base_line_summary, summary=summary) / StabilityExperiment.METRIC(dataset=dataset, summary=noised_dataset))
                except:
                    stability_score = abs(StabilityExperiment.METRIC(dataset=base_line_summary, summary=summary))
                print("Obtain stability score={:.5f}".format(stability_score))
                scores.append(stability_score)
        return "{} \\pm {}".format(np.nanmean(scores), np.nanstd(scores))

    @staticmethod
    def _add_dataset_gaussian_noise(dataset: pd.DataFrame,
                                    noise: float) -> pd.DataFrame:
        """
        A noise function that takes the dataset and adds to each value a Gaussian noise with a given STD = noise
        :param dataset: The dataset we want to add noise on
        :param noise: the noise hyper-parameter
        :return: noisy dataframe as Pandas' DataFrame
        """
        return dataset + np.random.normal(0, noise, [dataset.shape[0], dataset.shape[1]])

    @staticmethod
    def _add_dataset_subset_pick_noise(dataset: pd.DataFrame,
                                       noise: float) -> pd.DataFrame:
        """
        A noise function that takes a random subset of size (1-noise) from the original dataset for both the rows and columns
        :param dataset: The dataset we want to add noise on
        :param noise: the noise hyper-parameter
        :return: noisy dataframe as Pandas' DataFrame
        """
        row_indexes = list(range(dataset.shape[0]))
        col_indexes = list(range(dataset.shape[1]))
        return dataset.iloc[random.sample(row_indexes, round((1-noise) * len(row_indexes))), random.sample(col_indexes, round((1-noise) * len(col_indexes)))]


def prepare_dataset(df):
    # remove what we do not need
    df.drop([col for col in list(df) if not is_numeric_dtype(df[col])], axis=1, inplace=True)
    # remove _rows with nan
    df.dropna(inplace=True)
    # get only max number of _rows to work with
    return df.iloc[:200, -5:]


def run_test():
    # load and prepare datasets
    datasets = {os.path.basename(path).replace(".csv", ""): prepare_dataset(pd.read_csv(path))
                for path in glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*.csv"))}

    StabilityExperiment.run(datasets=datasets,
                            noise_function=StabilityExperiment._add_dataset_subset_pick_noise,
                            algorithms={
                                "las_vegas": LasVegasSummary,
                                "greedy": GreedySummary,
                                "genetic": GeneticSummary,
                            },
                            summaries_sizes=[(10, 0), (20, 0), (30, 0)],
                            main_save_folder_path=os.path.join(os.path.dirname(__file__), "stability_results"),
                            max_iter=30)


if __name__ == '__main__':
    run_test()
