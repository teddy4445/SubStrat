# library imports
import os
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime

# project imports
from table import Table
from greedy_summary_algorithm import GreedySummary
from summary_wellness_scores import SummaryWellnessScores
from analysis_converge_process import AnalysisConvergeProcess
from summary_process_score_functions import SummaryProcessScoreFunctions


class MultiScoreMultiDatasetExperiment:
    """
    This class generates a summary table of a summary's algorithm performance over multiple score functions, datasets, and summary sizes
    """

    def __init__(self):
        pass

    @staticmethod
    def run(datasets: dict,
            score_metrics: dict,
            summaries_sizes: list,
            evaluation_metric,
            main_save_folder_path: str,
            max_iter: int = 30):
        """
        Generate multiple tables of: rows -> dataset, columns ->  summary score metrics, divided by summary sizes
        :param datasets: a dict of datasets names (as keys) and pandas' dataframe (as values)
        :param score_metrics: a dict of metric functions object getting dataset (pandas' dataframe) and summary (pandas' dataframe) and give a score (float) where the function name is the key and the function object is the value.
        :param summaries_sizes: a list of tuples (with 2 elements) - the number of rows and the number of columns
        :param main_save_folder_path: the folder name where we wish to write the results in
        :param evaluation_metric:  a function object getting dataset (pandas' dataframe) and summary (pandas' dataframe) and give a score (float) to the entire summary
        :param max_iter: the maximum number of iteration we allow to do for one combination of dataset, metric, summary size
        :return: None, save csv files and plots to the folder
        """
        # make sure the save folder is exits
        try:
            os.mkdir(main_save_folder_path)
        except:
            pass

        # run over all summary size wanted
        for summary_size in summaries_sizes:
            desired_row_size = summary_size[0]
            desired_col_size = summary_size[1]

            # init a table instance so it will be easy to populate a pd.dataframe
            df_table_scores = Table(columns=[metric_name for metric_name in score_metrics],
                                    rows_ids=[dataset_name for dataset_name in datasets])
            df_table_step_compute = Table(columns=[metric_name for metric_name in score_metrics],
                                          rows_ids=[dataset_name for dataset_name in datasets])
            df_table_process_time = Table(columns=[metric_name for metric_name in score_metrics],
                                          rows_ids=[dataset_name for dataset_name in datasets])

            # run over all the metrics
            for metric_name, score_metric_function in score_metrics.items():
                # run over all the datasets
                for dataset_name, dataset in datasets.items():
                    # run the summary and obtain result and coverage report
                    summary, converge_report = GreedySummary.run(dataset=dataset,
                                                                 desired_row_size=desired_row_size,
                                                                 desired_col_size=desired_col_size,
                                                                 row_score_function=score_metric_function,
                                                                 evaluate_score_function=evaluation_metric,
                                                                 is_return_indexes=False,
                                                                 save_converge_report=os.path.join(
                                                                     main_save_folder_path,
                                                                     "converge_report_{}_{}_summary_{}X{}.json".format(
                                                                         metric_name,
                                                                         dataset_name,
                                                                         desired_row_size,
                                                                         desired_col_size)),
                                                                 max_iter=max_iter)
                    # save results into a table
                    df_table_scores.add(column=metric_name,
                                        row_id=dataset_name,
                                        data_point=converge_report.g)
            # TODO : save the table into a csv file
            # TODO : covert the table into a heatmap and save as an image


def run_test():
    datasets = {os.path.basename(path): pd.read_csv(path)
                for path in glob(os.path.join(os.path.dirname(__file__), "data", "*.csv"))}

    MultiScoreMultiDatasetExperiment.run(datasets=datasets,
                                         score_metrics={
                                             "mean_entropy": SummaryWellnessScores.mean_entropy,
                                             "coefficient_of_anomaly": SummaryWellnessScores.coefficient_of_anomaly,
                                             "coefficient_of_variation": SummaryWellnessScores.coefficient_of_variation,
                                             "mean_pearson_corr": SummaryWellnessScores.mean_pearson_corr
                                         },
                                         summaries_sizes=[(10, 10), (20, 10), (10, 20), (20, 20)],
                                         evaluation_metric=SummaryWellnessScores.mean_entropy,
                                         main_save_folder_path=os.path.join(os.path.dirname(__file__), "results"),
                                         max_iter=30)


if __name__ == '__main__':
    run_test()
