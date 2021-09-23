# library imports
import os
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from datetime import datetime
from pandas.core.dtypes.common import is_numeric_dtype

# project imports
from ds.table import Table
from ds.converge_report import ConvergeReport
from summary_algorithms.greedy_summary_algorithm import GreedySummary
from summary_algorithms.las_vegas_summary_algorithm import LasVegasSummary
from summary_algorithms.genetic_algorithm_summary_algorithm import GeneticSummary
from methods.summary_wellness_scores import SummaryWellnessScores
from plots.analysis_converge_process import AnalysisConvergeProcess


class MultiScoreMultiDatasetExperiment:
    """
    This class generates a summary table of a summary's algorithm performance over multiple score functions, datasets, and summary sizes
    """

    # CONSTS #
    SUMMARY_ALGORITHMS = {"genetic": GeneticSummary,
                          "las_vegas": LasVegasSummary,
                          "greedy": GreedySummary, }
    # END - CONSTS #

    # CHANGEABLE #
    SUMMARY_ALGORITHM = GreedySummary

    # END - CHANGEABLE #

    def __init__(self):
        pass

    @staticmethod
    def run(datasets: dict,
            score_metrics: dict,
            summaries_sizes: list,
            main_save_folder_path: str,
            max_iter: int = 30):
        """
        Generate multiple tables of: _rows -> dataset, columns ->  summary score metrics, divided by summary sizes
        :param datasets: a dict of datasets names (as keys) and pandas' dataframe (as values)
        :param score_metrics: a dict of metric functions object getting dataset (pandas' dataframe) and summary (pandas' dataframe) and give a score (float) where the function name is the key and the function object is the value.
        :param summaries_sizes: a list of tuples (with 2 elements) - the number of _rows and the number of columns
        :param main_save_folder_path: the folder name where we wish to write the results in
        :param max_iter: the maximum number of iteration we allow to do for one combination of dataset, metric, summary size
        :return: None, save csv files and plots to the folder
        """
        # make sure the save folder is exits
        try:
            os.mkdir(main_save_folder_path)
        except:
            pass

        # init tables for aggregated analysis #
        # prepare _rows
        summary_rows_ids = []
        for summary_size in summaries_sizes:
            for dataset_name in datasets:
                summary_rows_ids.append("{}_{}X{}".format(dataset_name, summary_size[0], summary_size[1]))
        summary_table = Table(
            columns=["steps",
                     "first_row_score", "first_col_score", "first_global_score",
                     "last_row_score", "last_col_score", "last_global_score",
                     "minimal_row_score", "minimal_col_score", "minimal_global_score",
                     "maximal_row_score", "maximal_col_score", "maximal_global_score",
                     "mean_row_score", "mean_col_score", "mean_global_score",
                     "first_row_time", "first_col_time", "first_global_time",
                     "last_row_time", "last_col_time", "last_global_time",
                     "minimal_row_time", "minimal_col_time", "minimal_global_time",
                     "maximal_row_time", "maximal_col_time", "maximal_global_time",
                     "mean_row_time", "mean_col_time", "mean_global_time"],
            rows_ids=summary_rows_ids)

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
                    # alert the user
                    print("For summary size ({}X{}), working on dataset '{}' with metric '{}'".format(desired_row_size,
                                                                                                      desired_col_size,
                                                                                                      dataset_name,
                                                                                                      metric_name))
                    # run the summary and obtain result and coverage report
                    summary, converge_report = MultiScoreMultiDatasetExperiment.SUMMARY_ALGORITHM.run(dataset=dataset,
                                                                                                      desired_row_size=desired_row_size,
                                                                                                      desired_col_size=desired_col_size,
                                                                                                      row_score_function=score_metric_function,
                                                                                                      evaluate_score_function=score_metric_function,
                                                                                                      is_return_indexes=False,
                                                                                                      save_converge_report=os.path.join(
                                                                                                          main_save_folder_path,
                                                                                                          "converge_report_{}_{}_summary_{}X{}.json".format(
                                                                                                              metric_name,
                                                                                                              dataset_name,
                                                                                                              desired_row_size,
                                                                                                              desired_col_size)),
                                                                                                      max_iter=max_iter)

                    # generate and save a score of the metric over iterations plot
                    AnalysisConvergeProcess.converge_scores(rows_scores=converge_report["rows_score"],
                                                            cols_scores=converge_report["cols_score"],
                                                            total_scores=converge_report["total_score"],
                                                            y_label="Error function value [1]",
                                                            save_path=os.path.join(
                                                                main_save_folder_path,
                                                                "converge_scores_{}_{}_summary_{}X{}.png".format(
                                                                    metric_name,
                                                                    dataset_name,
                                                                    desired_row_size,
                                                                    desired_col_size)))

                    # save results into a table
                    df_table_scores.add(column=metric_name,
                                        row_id=dataset_name,
                                        data_point=converge_report.final_total_score())
                    df_table_step_compute.add(column=metric_name,
                                              row_id=dataset_name,
                                              data_point=converge_report.steps_count())
                    df_table_process_time.add(column=metric_name,
                                              row_id=dataset_name,
                                              data_point=converge_report.compute_time())

                    # add data to the summay table
                    MultiScoreMultiDatasetExperiment._add_summary_table_row(summary_table=summary_table,
                                                                            converge_report=converge_report,
                                                                            row_id="{}_{}X{}".format(dataset_name,
                                                                                                     desired_row_size,
                                                                                                     desired_col_size))

                # same the summary table
                summary_table.to_csv(
                    save_path=os.path.join(main_save_folder_path, "summary_table_{}.csv".format(metric_name)))

            # convert tables into dataframes
            dfs = {name: table.to_dataframe() for name, table in {"scores": df_table_scores,
                                                                  "process_time": df_table_process_time,
                                                                  "step_compute": df_table_step_compute}.items()}
            # save the tables into a CSVs files
            [df.to_csv(
                os.path.join(main_save_folder_path, "{}X{}_{}.csv".format(desired_row_size, desired_col_size, df_name)))
             for df_name, df in dfs.items()]
            # covert the table into a heatmap and save as an image
            [AnalysisConvergeProcess.heatmap(df=df,
                                             save_path=os.path.join(main_save_folder_path,
                                                                    "heatmap_{}X{}_{}.png".format(desired_row_size,
                                                                                                  desired_col_size,
                                                                                                  df_name)))
             for df_name, df in dfs.items()]

    @staticmethod
    def _add_summary_table_row(summary_table: Table,
                               converge_report: ConvergeReport,
                               row_id: str):
        """
        This function responsible to add a summary table row to the summary table
        :param summary_table: the summary table instance
        :param converge_report: the new converge report
        :param row_id: the dataset and sizes of the converge report's name
        :return: none, modify the summary table
        """
        converge_report_total_times = [converge_report.total_time(index=index) for index in range(converge_report.steps_count())]

        summary_table.add(column="steps",
                          row_id=row_id,
                          data_point=converge_report.steps_count())
        summary_table.add(column="first_row_time",
                          row_id=row_id,
                          data_point=converge_report.rows_calc_time[0])
        summary_table.add(column="first_col_time",
                          row_id=row_id,
                          data_point=converge_report.cols_calc_time[0])
        summary_table.add(column="first_global_time",
                          row_id=row_id,
                          data_point=converge_report.total_time(index=0))
        summary_table.add(column="last_row_time",
                          row_id=row_id,
                          data_point=converge_report.rows_calc_time[-1])
        summary_table.add(column="last_col_time",
                          row_id=row_id,
                          data_point=converge_report.cols_calc_time[-1])
        summary_table.add(column="last_global_time",
                          row_id=row_id,
                          data_point=converge_report.total_time(index=-1))
        summary_table.add(column="minimal_row_time",
                          row_id=row_id,
                          data_point=min(converge_report.rows_calc_time))
        summary_table.add(column="minimal_col_time",
                          row_id=row_id,
                          data_point=min(converge_report.cols_calc_time))
        summary_table.add(column="minimal_global_time",
                          row_id=row_id,
                          data_point=min(converge_report_total_times))
        summary_table.add(column="maximal_row_time",
                          row_id=row_id,
                          data_point=max(converge_report.rows_calc_time))
        summary_table.add(column="maximal_col_time",
                          row_id=row_id,
                          data_point=max(converge_report.cols_calc_time))
        summary_table.add(column="maximal_global_time",
                          row_id=row_id,
                          data_point=max(converge_report_total_times))
        summary_table.add(column="mean_row_time",
                          row_id=row_id,
                          data_point=np.nanmean(converge_report.rows_calc_time))
        summary_table.add(column="mean_col_time",
                          row_id=row_id,
                          data_point=np.nanmean(converge_report.rows_calc_time))
        summary_table.add(column="mean_global_time",
                          row_id=row_id,
                          data_point=np.nanmean(converge_report_total_times))
        summary_table.add(column="first_row_score",
                          row_id=row_id,
                          data_point=converge_report.rows_score[0])
        summary_table.add(column="first_col_score",
                          row_id=row_id,
                          data_point=converge_report.cols_score[0])
        summary_table.add(column="first_global_score",
                          row_id=row_id,
                          data_point=converge_report.total_score[0])
        summary_table.add(column="last_row_score",
                          row_id=row_id,
                          data_point=converge_report.rows_score[-1])
        summary_table.add(column="last_col_score",
                          row_id=row_id,
                          data_point=converge_report.cols_score[-1])
        summary_table.add(column="last_global_score",
                          row_id=row_id,
                          data_point=converge_report.total_score[-1])
        summary_table.add(column="minimal_row_score",
                          row_id=row_id,
                          data_point=min(converge_report.rows_score))
        summary_table.add(column="minimal_col_score",
                          row_id=row_id,
                          data_point=min(converge_report.cols_score))
        summary_table.add(column="minimal_global_score",
                          row_id=row_id,
                          data_point=min(converge_report.total_score))
        summary_table.add(column="maximal_row_score",
                          row_id=row_id,
                          data_point=max(converge_report.rows_score))
        summary_table.add(column="maximal_col_score",
                          row_id=row_id,
                          data_point=max(converge_report.cols_score))
        summary_table.add(column="maximal_global_score",
                          row_id=row_id,
                          data_point=max(converge_report.total_score))
        summary_table.add(column="mean_row_score",
                          row_id=row_id,
                          data_point=np.nanmean(converge_report.rows_score))
        summary_table.add(column="mean_col_score",
                          row_id=row_id,
                          data_point=np.nanmean(converge_report.cols_score))
        summary_table.add(column="mean_global_score",
                          row_id=row_id,
                          data_point=np.nanmean(converge_report.total_score))


def prepare_dataset(df):
    # remove what we do not need
    df.drop([col for col in list(df) if not is_numeric_dtype(df[col])], axis=1, inplace=True)
    # remove _rows with nan
    df.dropna(inplace=True)
    # get only max number of _rows to work with
    return df.iloc[:200, :]


def run_test_multiple_summary_algorithms():
    # run over all the algorithms in the list
    for algo_name, summary_algo in MultiScoreMultiDatasetExperiment.SUMMARY_ALGORITHMS.items():
        # set the right algorithm
        MultiScoreMultiDatasetExperiment.SUMMARY_ALGORITHM = summary_algo
        # run and save the results in a new inner folder
        run_test(inner_result_folder="multi_db_multi_metric_initial_results_{}".format(algo_name))


def run_test(inner_result_folder: str = "multi_db_multi_metric_initial_results"):
    # load and prepare datasets
    datasets = {os.path.basename(path): prepare_dataset(pd.read_csv(path))
                for path in glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*.csv"))}

    MultiScoreMultiDatasetExperiment.run(datasets=datasets,
                                         score_metrics={
                                             "mean_entropy": SummaryWellnessScores.mean_entropy,
                                             "coefficient_of_anomaly": SummaryWellnessScores.coefficient_of_anomaly,
                                             "coefficient_of_variation": SummaryWellnessScores.coefficient_of_variation,
                                             "mean_pearson_corr": SummaryWellnessScores.mean_pearson_corr
                                         },
                                         summaries_sizes=[(10, 3), (20, 3), (10, 5), (20, 5)],
                                         main_save_folder_path=os.path.join(os.path.dirname(__file__),
                                                                            inner_result_folder),
                                         max_iter=30)


if __name__ == '__main__':
    run_test_multiple_summary_algorithms()
