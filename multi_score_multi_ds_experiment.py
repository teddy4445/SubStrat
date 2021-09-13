# library imports
import os
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from datetime import datetime
from pandas.core.dtypes.common import is_numeric_dtype

# project imports
from table import Table
from converge_report import ConvergeReport
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

        # init tables for aggregated analysis #
        # prepare rows
        summary_rows_ids = []
        for summary_size in summaries_sizes:
            for dataset_name in datasets:
                summary_rows_ids.append("{}_{}X{}".format(dataset_name, summary_size[0], summary_size[1]))
        summary_table = Table(
            columns=["steps", "first_row_score", "first_col_score", "minimal_score", "maximal_score", "mean_score",
                     "first_row_time", "first_col_time", "minimal_time", "maximal_time", "mean_time"],
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
                    print("Working on dataset '{}' with metric '{}'".format(dataset_name, metric_name))
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

                    # generate and save a process plot
                    AnalysisConvergeProcess.iou_greedy_converge(rows_list=converge_report["rows"],
                                                                cols_list=converge_report["cols"],
                                                                save_path=os.path.join(
                                                                    main_save_folder_path,
                                                                    "greedy_converge_{}_{}_summary_{}X{}.png".format(
                                                                        metric_name,
                                                                        dataset_name,
                                                                        desired_row_size,
                                                                        desired_col_size)))
                    # generate and save a score of the metric over iterations plot
                    AnalysisConvergeProcess.greedy_converge_scores(rows_scores=converge_report["rows_score"],
                                                                   cols_scores=converge_report["cols_score"],
                                                                   total_scores=converge_report["total_score"],
                                                                   save_path=os.path.join(
                                                                       main_save_folder_path,
                                                                       "greedy_converge_scores_{}_{}_summary_{}X{}.png".format(
                                                                           metric_name,
                                                                           dataset_name,
                                                                           desired_row_size,
                                                                           desired_col_size)))
                    # generate and save a time of compute over iterations plot
                    AnalysisConvergeProcess.greedy_converge_times(rows_compute_time=converge_report["rows_calc_time"],
                                                                  cols_compute_time=converge_report["cols_calc_time"],
                                                                  save_path=os.path.join(
                                                                      main_save_folder_path,
                                                                      "greedy_converge_times_{}_{}_summary_{}X{}.png".format(
                                                                          metric_name,
                                                                          dataset_name,
                                                                          desired_row_size,
                                                                          desired_col_size)))
                    # make a summary video
                    AnalysisConvergeProcess.picking_summary_video(rows_list=converge_report["rows"],
                                                                  cols_list=converge_report["cols"],
                                                                  original_data_set_shape=dataset.shape,
                                                                  save_path_folder=os.path.join(
                                                                      main_save_folder_path,
                                                                      "video_{}_{}_{}X{}.png".format(
                                                                          metric_name,
                                                                          dataset_name,
                                                                          desired_row_size,
                                                                          desired_col_size)),
                                                                  fps=1)

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
        summary_table.add(column="steps",
                          row_id=row_id,
                          data_point=converge_report.steps_count())
        summary_table.add(column="first_row_time",
                          row_id=row_id,
                          data_point=converge_report.rows_calc_time[0])
        summary_table.add(column="first_col_score",
                          row_id=row_id,
                          data_point=converge_report.cols_calc_time[0])
        summary_table.add(column="minimal_score",
                          row_id=row_id,
                          data_point=min([min(converge_report.rows_calc_time), min(converge_report.cols_calc_time)]))
        summary_table.add(column="maximal_score",
                          row_id=row_id,
                          data_point=max([max(converge_report.rows_calc_time), max(converge_report.cols_calc_time)]))
        summary_table.add(column="mean_score",
                          row_id=row_id,
                          data_point=np.nanmean(
                              [np.nanmean(converge_report.rows_calc_time), np.nanmean(converge_report.cols_calc_time)]))
        summary_table.add(column="first_row_score",
                          row_id=row_id,
                          data_point=converge_report.rows_score[0])
        summary_table.add(column="first_col_score",
                          row_id=row_id,
                          data_point=converge_report.cols_score[0])
        summary_table.add(column="minimal_score",
                          row_id=row_id,
                          data_point=min([min(converge_report.rows_score), min(converge_report.cols_score)]))
        summary_table.add(column="maximal_score",
                          row_id=row_id,
                          data_point=max([max(converge_report.rows_score), max(converge_report.cols_score)]))
        summary_table.add(column="mean_score",
                          row_id=row_id,
                          data_point=np.nanmean(
                              [np.nanmean(converge_report.rows_score), np.nanmean(converge_report.cols_score)]))


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

    MultiScoreMultiDatasetExperiment.run(datasets=datasets,
                                         score_metrics={
                                             "mean_entropy": SummaryWellnessScores.mean_entropy,
                                             "coefficient_of_anomaly": SummaryWellnessScores.coefficient_of_anomaly,
                                             "coefficient_of_variation": SummaryWellnessScores.coefficient_of_variation,
                                             "mean_pearson_corr": SummaryWellnessScores.mean_pearson_corr
                                         },
                                         summaries_sizes=[(10, 3), (20, 3), (10, 5), (20, 5)],
                                         evaluation_metric=SummaryWellnessScores.mean_entropy,
                                         main_save_folder_path=os.path.join(os.path.dirname(__file__), "results"),
                                         max_iter=30)


if __name__ == '__main__':
    run_test()
