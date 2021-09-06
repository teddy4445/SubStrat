# library imports
import os
import numpy as np
import pandas as pd
from datetime import datetime

# project imports
from greedy_summary_algorithm import greedy_summary
from summary_score_functions import SummaryScoreFunctions
from analysis_converge_process import AnalysisConvergeProcess


class Main:
    """
    Manage the running of the simulation with analysis of results for the paper and IO operations
    """

    FILE_TIME_FORMAT = "%m_%d_%Y__%H_%M_%S"
    SUMMARY_REPORT_FILE_PATH = os.path.join(os.path.dirname(__file__), "results", "greedy_converge_report_{}.json".format(datetime.now().strftime(FILE_TIME_FORMAT)))
    SUMMARY_REPORT_PLOT_FILE_PATH = os.path.join(os.path.dirname(__file__), "results", "greedy_converge_report_{}.png".format(datetime.now().strftime(FILE_TIME_FORMAT)))
    PICKING_SUMMARY_VIDEO_FOLDER = os.path.join(os.path.dirname(__file__), "results", "{}".format(datetime.now().strftime(FILE_TIME_FORMAT)))

    def __init__(self):
        pass

    @staticmethod
    def run(data_file_path: str,
            data_row_working_size: int,
            data_rows_name_to_delete: list,
            desired_row_size: int,
            desired_col_size: int,
            result_file_path: str):
        data = Main.read_data(data_file_path=data_file_path,
                              data_row_working_size=data_row_working_size,
                              data_rows_name_to_delete=data_rows_name_to_delete)
        summary, converge_report = Main.get_summary(data=data,
                                                    desired_row_size=desired_row_size,
                                                    desired_col_size=desired_col_size)
        Main.save_results(result_file_path=result_file_path,
                          summary=summary,
                          data_shape=data.shape,
                          converge_report=converge_report)

    @staticmethod
    def read_data(data_file_path: str,
                  data_row_working_size: int,
                  data_rows_name_to_delete: list) -> pd.DataFrame:
        # read the data
        df = pd.read_csv(data_file_path)
        # remove unwanted columns
        df.drop(data_rows_name_to_delete, axis=1, inplace=True)
        # down sample the dataset so we can work with - take the first rows (random decision, can be changes later)
        df = df.iloc[:data_row_working_size, :]
        return df

    @staticmethod
    def get_summary(data: pd.DataFrame,
                    desired_row_size: int,
                    desired_col_size: int) -> pd.DataFrame:
        return greedy_summary(dataset=data,
                              desired_row_size=desired_row_size,
                              desired_col_size=desired_col_size,
                              row_score_function=SummaryScoreFunctions.row_entropy,
                              is_return_indexes=False,
                              save_converge_report=Main.SUMMARY_REPORT_FILE_PATH,
                              max_iter=30)

    @staticmethod
    def save_results(result_file_path: str,
                     data_shape: tuple,
                     summary: pd.DataFrame,
                     converge_report: dict) -> None:
        # save the summary for a file
        summary.to_csv(result_file_path, index=False)
        # generate and save a process plot
        AnalysisConvergeProcess.iou_greedy_converge(rows_list=converge_report["rows"],
                                                    cols_list=converge_report["cols"],
                                                    save_path=Main.SUMMARY_REPORT_PLOT_FILE_PATH)
        # make a summary video
        AnalysisConvergeProcess.picking_summary_video(rows_list=converge_report["rows"],
                                                      cols_list=converge_report["cols"],
                                                      original_data_set_shape=data_shape,
                                                      save_path_folder=Main.PICKING_SUMMARY_VIDEO_FOLDER,
                                                      fps=4)


if __name__ == '__main__':
    Main.run(data_file_path=os.path.join(os.path.dirname(__file__), "data", "data.csv"),
             data_row_working_size=50,
             data_rows_name_to_delete=["id", "species", "genus"],
             desired_row_size=20,
             desired_col_size=20,
             result_file_path=os.path.join(os.path.dirname(__file__), "results", "summary.csv".format(datetime.now().strftime(Main.FILE_TIME_FORMAT))))
