# library imports
import os
import numpy as np
import pandas as pd
from datetime import datetime

# project imports
from ds.converge_report import ConvergeReport
from methods.summary_wellness_scores import SummaryWellnessScores
from plots.analysis_converge_process import AnalysisConvergeProcess
from summary_algorithms.stabler_genetic_algorithm_summary_algorithm import StablerGeneticSummary


class TestMachineLearningMetric:
    """
    Test if the machine learning metric works fine
    """

    SUMMARY_ALGORITHM = StablerGeneticSummary
    SUMMARY_ALGORITHM_NAME = "StablerGeneticSummary"

    # CONSTS #
    RESULTS_FOLDER_NAME = "sklearn_metric_test"
    FILE_TIME_FORMAT = "%m_%d_%Y__%H_%M_%S"
    SUMMARY_REPORT_FILE_PATH = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME,
                                            "converge_report_{}.json".format(
                                                datetime.now().strftime(FILE_TIME_FORMAT)))
    SUMMARY_REPORT_PLOT_FILE_PATH = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME,
                                                 "converge_report_{}.png".format(
                                                     datetime.now().strftime(FILE_TIME_FORMAT)))
    SUMMARY_REPORT_SCORE_PLOT_FILE_PATH = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME,
                                                       "scores_converge_report_{}.png".format(
                                                           datetime.now().strftime(FILE_TIME_FORMAT)))
    SUMMARY_REPORT_TIMES_PLOT_FILE_PATH = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME,
                                                       "times_converge_report_{}.png".format(
                                                           datetime.now().strftime(FILE_TIME_FORMAT)))
    PICKING_SUMMARY_VIDEO_FOLDER = os.path.join(os.path.dirname(__file__), RESULTS_FOLDER_NAME,
                                                "{}".format(datetime.now().strftime(FILE_TIME_FORMAT)))

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(data_file_path: str,
            data_row_working_size: int,
            data_col_working_size: int,
            data_rows_name_to_delete: list,
            desired_row_size: int,
            desired_col_size: int,
            result_file_path: str):
        """
        Single entry point to the class
        :param data_file_path: the path to the file with the tabular data
        :param data_row_working_size: the number of _rows we want to reduce the original file (so we could work with in reasonable time)
        :param data_col_working_size: the number of _cols we want to reduce the original file (so we could work with in reasonable time)
        :param data_rows_name_to_delete: list of columns name we want to remove from the original dataset
        :param desired_row_size: the number of _rows we want in the summary
        :param desired_col_size: the number of columns we want in the summary
        :param result_file_path: the path to the folder we want to save the results in
        :return: None, same several files
        """
        # make sure the IO needed is in place
        TestMachineLearningMetric._prepare()
        # read the data and make needed reductions in size
        data = TestMachineLearningMetric._read_data(data_file_path=data_file_path,
                                                    data_row_working_size=data_row_working_size,
                                                    data_col_working_size=data_col_working_size,
                                                    data_rows_name_to_delete=data_rows_name_to_delete)
        # run the summary algorithm and get the summary and coverge report
        summary, converge_report = TestMachineLearningMetric.get_summary(data=data,
                                                                         desired_row_size=desired_row_size,
                                                                         desired_col_size=desired_col_size)
        # calculate cool graphs on the summary process and save all the results
        TestMachineLearningMetric.save_results(result_file_path=result_file_path,
                                               summary=summary,
                                               data_shape=data.shape,
                                               converge_report=converge_report)

    @staticmethod
    def _prepare():
        """
        Make sure the IO folders and other system stuff ready for the process
        """
        try:
            os.mkdir(os.path.join(os.path.dirname(__file__), TestMachineLearningMetric.RESULTS_FOLDER_NAME))
        except:
            pass
        # update save paths
        TestMachineLearningMetric.SUMMARY_REPORT_FILE_PATH = os.path.join(os.path.dirname(__file__),
                                                                          TestMachineLearningMetric.RESULTS_FOLDER_NAME,
                                                                          "converge_report_{}.json".format(
                                                                              datetime.now().strftime(
                                                                                  TestMachineLearningMetric.FILE_TIME_FORMAT)))
        TestMachineLearningMetric.SUMMARY_REPORT_PLOT_FILE_PATH = os.path.join(os.path.dirname(__file__),
                                                                               TestMachineLearningMetric.RESULTS_FOLDER_NAME,
                                                                               "converge_report_{}.png".format(
                                                                                   datetime.now().strftime(
                                                                                       TestMachineLearningMetric.FILE_TIME_FORMAT)))
        TestMachineLearningMetric.SUMMARY_REPORT_SCORE_PLOT_FILE_PATH = os.path.join(os.path.dirname(__file__),
                                                                                     TestMachineLearningMetric.RESULTS_FOLDER_NAME,
                                                                                     "scores_converge_report_{}.png".format(
                                                                                         datetime.now().strftime(
                                                                                             TestMachineLearningMetric.FILE_TIME_FORMAT)))
        TestMachineLearningMetric.PICKING_SUMMARY_VIDEO_FOLDER = os.path.join(os.path.dirname(__file__),
                                                                              TestMachineLearningMetric.RESULTS_FOLDER_NAME,
                                                                              "{}".format(datetime.now().strftime(
                                                                                  TestMachineLearningMetric.FILE_TIME_FORMAT)))
        TestMachineLearningMetric.SUMMARY_REPORT_TIMES_PLOT_FILE_PATH = os.path.join(os.path.dirname(__file__),
                                                                                     TestMachineLearningMetric.RESULTS_FOLDER_NAME,
                                                                                     "times_converge_report_{}.png".format(
                                                                                         datetime.now().strftime(
                                                                                             TestMachineLearningMetric.FILE_TIME_FORMAT)))

    @staticmethod
    def _read_data(data_file_path: str,
                   data_row_working_size: int,
                   data_col_working_size: int,
                   data_rows_name_to_delete: list) -> pd.DataFrame:
        """
        Responsible to read and reduce size of the dataset we need to work with
        :param data_file_path: the path to the file with the tabular data
        :param data_row_working_size: the number of _rows we want to reduce the original file (so we could work with in reasonable time)
        :param data_col_working_size: the number of _cols we want to reduce the original file (so we could work with in reasonable time)
        :param data_rows_name_to_delete: list of columns name we want to remove from the original dataset
        :return: the reduced dataset as pandas' dataframe object
        """
        # read the data
        df = pd.read_csv(data_file_path)
        # remove unwanted columns
        df.drop(data_rows_name_to_delete, axis=1, inplace=True)
        # remove nans
        df.dropna(inplace=True)
        # down sample the dataset so we can work with - take the first _rows (random decision, can be changes later)
        df = df.iloc[:data_row_working_size, :data_col_working_size]
        return df

    @staticmethod
    def get_summary(data: pd.DataFrame,
                    desired_row_size: int,
                    desired_col_size: int) -> pd.DataFrame:
        """
        Activates the summary algorithm
        :param data: the dataset we will use in the summary process
        :param desired_row_size: the number of _rows we want in the summary
        :param desired_col_size: the number of columns we want in the summary
        :return: the summary (pandas' dataframe) and the converge process (dict)
        """
        return TestMachineLearningMetric.SUMMARY_ALGORITHM.run(dataset=data,
                                                               desired_row_size=desired_row_size,
                                                               desired_col_size=desired_col_size,
                                                               evaluate_score_function=SummaryWellnessScores.sklearn_model,
                                                               is_return_indexes=False,
                                                               save_converge_report=TestMachineLearningMetric.SUMMARY_REPORT_FILE_PATH,
                                                               max_iter=30)

    @staticmethod
    def save_results(result_file_path: str,
                     data_shape: tuple,
                     summary: pd.DataFrame,
                     converge_report: ConvergeReport) -> None:
        """
        Calculates some important graphs from the converge process and save them with the resulted summary
        :param result_file_path: the path to the folder we want to save the results in (str)
        :param data_shape: the shape (sizes) of the original dataset (tuple)
        :param summary: the summary of the dataset (pandas' dataframe)
        :param converge_report: a dictinary with the main data of the process of the summary algorithm
        :return: None, saves data in files
        """
        # save the summary for a file
        summary.to_csv(result_file_path, index=False)
        # generate and save a process plot
        AnalysisConvergeProcess.iou_converge(rows_list=converge_report["_rows"],
                                             cols_list=converge_report["_cols"],
                                             save_path=TestMachineLearningMetric.SUMMARY_REPORT_PLOT_FILE_PATH)
        # generate and save a score of the metric over iterations plot
        AnalysisConvergeProcess.converge_scores(rows_scores=converge_report["rows_score"],
                                                cols_scores=converge_report["cols_score"],
                                                total_scores=converge_report["total_score"],
                                                y_label="'{}' algorithm's error [1]".format(
                                                    TestMachineLearningMetric.SUMMARY_ALGORITHM_NAME),
                                                save_path=TestMachineLearningMetric.SUMMARY_REPORT_SCORE_PLOT_FILE_PATH)
        # generate and save a time of compute over iterations plot
        AnalysisConvergeProcess.converge_times(rows_compute_time=converge_report["rows_calc_time"],
                                               cols_compute_time=converge_report["cols_calc_time"],
                                               save_path=TestMachineLearningMetric.SUMMARY_REPORT_TIMES_PLOT_FILE_PATH)

        # upper bound this process as it generates a lot of images and can memory-out
        if converge_report.steps_count() < 100:
            # make a summary video
            AnalysisConvergeProcess.picking_summary_video(rows_list=converge_report["_rows"],
                                                          cols_list=converge_report["_cols"],
                                                          original_data_set_shape=data_shape,
                                                          save_path_folder=TestMachineLearningMetric.PICKING_SUMMARY_VIDEO_FOLDER,
                                                          fps=4)


if __name__ == '__main__':
    row_size = 79500
    col_size = 7
    print("Starting to work on size: {}X{}".format(row_size, col_size))
    TestMachineLearningMetric.run(data_file_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                              "big_data",
                                                              "dataset_25.csv"),
                                  data_row_working_size=row_size,
                                  data_col_working_size=col_size,
                                  data_rows_name_to_delete=[],
                                  desired_row_size=282,
                                  desired_col_size=7,
                                  result_file_path=os.path.join(os.path.dirname(__file__),
                                                                TestMachineLearningMetric.RESULTS_FOLDER_NAME,
                                                                "summary_{}.csv".format(datetime.now().strftime(
                                                                    TestMachineLearningMetric.FILE_TIME_FORMAT))))
