# library imports
import os
import numpy as np
import pandas as pd
from glob import glob
from pandas.core.dtypes.common import is_numeric_dtype

# project imports
from ds.table import Table
from ds.converge_report import ConvergeReport
from summary_algorithms.greedy_summary_algorithm import GreedySummary
from summary_algorithms.las_vegas_summary_algorithm import LasVegasSummary
from methods.summary_wellness_scores import SummaryWellnessScores
from plots.analysis_converge_process import AnalysisConvergeProcess
from methods.summary_process_score_functions import SummaryProcessScoreFunctions


class StabilityExperiment:
    """
    This class generates a summary table of a summary's algorithm performance over multiple score functions, datasets, and summary sizes
    """

    METRIC = SummaryWellnessScores.mean_entropy

    def __init__(self):
        pass

    @staticmethod
    def run(datasets: dict,
            algorithms: dict,
            summaries_sizes: list,
            main_save_folder_path: str,
            max_iter: int = 30):
        """
        Generate multiple tables of: rows -> dataset, columns ->  summary score metrics, divided by summary sizes
        :param datasets: a dict of datasets names (as keys) and pandas' dataframe (as values)
        :param algorithms: a dict of algorithms one wants to test one vs. the other
        :param summaries_sizes: a list of tuples (with 2 elements) - the number of rows and the number of columns
        :param main_save_folder_path: the folder name where we wish to write the results in
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

            for dataset_name, df in datasets.items():
                for algo_name, algo in algorithms.items():
                    summary_table.add(column=dataset_name,
                                      row_id=algo_name,
                                      data_point=StabilityExperiment._stability_test(dataset=df,
                                                                                     summary_col_size=desired_row_size,
                                                                                     summary_row_size=desired_col_size,
                                                                                     summary_algorithm=algo,
                                                                                     noise=0.1,
                                                                                     repeats=3,
                                                                                     start_condition_repeat=3,
                                                                                     max_iter=30,
                                                                                     save_converge_report=os.path.join(
                                                                                         main_save_folder_path,
                                                                                         "greedy_converge_scores_{}_{}_summary_{}X{}.png".format(
                                                                                             "mean_entropy",
                                                                                             dataset_name,
                                                                                             desired_row_size,
                                                                                             desired_col_size))))

            summary_table.to_csv(save_path=os.path.join(main_save_folder_path, "summary_table_{}X{}.csv".format(desired_row_size, desired_col_size)))

    @staticmethod
    def _stability_test(dataset: pd.DataFrame,
                        summary_row_size: int,
                        summary_col_size: int,
                        summary_algorithm,
                        noise: float,
                        repeats: int,
                        start_condition_repeat: int,
                        max_iter: int,
                        save_converge_report: str):
        scores = []
        for repeat in range(repeats):
            noised_dataset = StabilityExperiment._add_dataset_gussian_noise(dataset=dataset,
                                                                            noise=noise)
            for start_condition in range(start_condition_repeat):
                summary, converge_report = summary_algorithm.run(dataset=noised_dataset,
                                                                 desired_row_size=summary_row_size,
                                                                 desired_col_size=summary_col_size,
                                                                 evaluate_score_function=StabilityExperiment.METRIC,
                                                                 is_return_indexes=False,
                                                                 save_converge_report=os.path.join(
                                                                     save_converge_report),
                                                                 max_iter=max_iter)
                scores.append(StabilityExperiment.METRIC(dataset=noised_dataset,
                                                         summary=summary))
        return np.nanmean(scores), np.nanstd(scores)

    @staticmethod
    def _add_dataset_gussian_noise(dataset: pd.DataFrame,
                                   noise: float) -> pd.DataFrame:
        return dataset + np.random.normal(0, noise, [dataset.shape[0], dataset.shape[1]])


def prepare_dataset(df):
    # remove what we do not need
    df.drop([col for col in list(df) if not is_numeric_dtype(df[col])], axis=1, inplace=True)
    # remove rows with nan
    df.dropna(inplace=True)
    # get only max number of rows to work with
    return df.iloc[:200, :]


def run_test():
    # load and prepare datasets
    datasets = {os.path.basename(path): prepare_dataset(pd.read_csv(path))
                for path in glob(os.path.join(os.path.dirname(__file__), "data", "*.csv"))}

    StabilityExperiment.run(datasets=datasets,
                            algorithms={
                                "las_vegas": LasVegasSummary,
                                "greedy": GreedySummary
                            },
                            summaries_sizes=[(10, 3), (20, 3), (10, 5), (20, 5)],
                            main_save_folder_path=os.path.join(os.path.dirname(__file__), "results"),
                            max_iter=30)


if __name__ == '__main__':
    run_test()
