# library imports
import json
import time
import random
import pandas as pd
from itertools import combinations

# project import
from ds.converge_report import ConvergeReport
from summary_algorithms.base_summary_algorithm import BaseSummary


class BruteForceSummary(BaseSummary):
    """
    This class is a brute-force type algorithm for dataset summarization
    """

    def __init__(self):
        BaseSummary.__init__(self)

    @staticmethod
    def run(dataset: pd.DataFrame,
            desired_row_size: int,
            desired_col_size: int,
            evaluate_score_function,
            save_converge_report: str = "",
            row_score_function = None,
            is_return_indexes: bool = False,
            max_iter: int = -1):
        """
        A row-columns greedy dataset summary algorithm
        :param dataset: the dataset we work on (pandas' dataframe)
        :param desired_row_size: the size of the summary as the number of _rows (int)
        :param desired_col_size: the size of the summary as the number of columns (int)
        :param row_score_function: a function object getting dataset (pandas' dataframe) and summary (pandas' dataframe) and give the score of row\column optimization process
        :param evaluate_score_function: a function object getting dataset (pandas' dataframe) and summary (pandas' dataframe) and give a score (float) to the entire summary
        :param save_converge_report: a path to write the converge report to (default - do not write)
        :param is_return_indexes:  boolean flag to return summary's _rows indexes of after applying to the dataset itself
        :param max_iter: the maximum number of iteration we allow to do (default - unlimited)
        :return: the summary of the dataset with converge report (dict)
        """
        # make sure we have steps to run
        if max_iter < 1:
            raise Exception("Error at LasVegasSummary.run: the max_iter argument must be larger than 1")

        # setting the round count to the beginning of the process
        round_count = 1
        # if requested, init empty converge report
        converge_report = ConvergeReport()

        # init all the vars we need in the process
        best_score = 9999  # TODO: can be done better
        best_rows = []
        best_columns = []

        # all size of rows and columns to pick from
        old_pick_rows = list(range(dataset.shape[0]))
        old_pick_columns = list(range(dataset.shape[1]))
        pick_rows = list(range(dataset.shape[0]))  # all _rows
        pick_columns = list(range(dataset.shape[1]))  # all columns

        for row_index, current_rows in enumerate(combinations(pick_rows, desired_row_size)):
            for col_index, current_columns in enumerate(combinations(pick_columns, desired_col_size)):

                print("Working in row combination = {} and col combination = {}".format(row_index, col_index))

                # pick _rows
                start_rows_calc = time.time()  # just for time measurement tasks
                start_cols_calc = time.time()  # just for time measurement tasks

                # convert from set to list
                current_rows = sorted(list(current_rows))
                current_columns = sorted(list(current_columns))

                # compute scores
                rows_summary_score = evaluate_score_function(dataset, dataset.iloc[current_rows, old_pick_columns])
                cols_summary_score = evaluate_score_function(dataset, dataset.iloc[old_pick_rows, current_columns])
                total_score = evaluate_score_function(dataset, dataset.iloc[current_rows, current_columns])

                end_rows_calc = time.time()  # just for time measurement tasks
                end_cols_calc = time.time()  # just for time measurement tasks

                # Add the data for the report
                converge_report.add_step(row=current_rows,
                                         col=current_columns,
                                         row_score=rows_summary_score,
                                         col_score=cols_summary_score,
                                         row_calc_time=(end_rows_calc - start_rows_calc) / 2,
                                         col_calc_time=(end_cols_calc - start_cols_calc) / 2,
                                         total_score=total_score)

                # check we this summary is better
                if total_score < best_score:
                    best_score = total_score
                    best_rows = current_rows
                    best_columns = current_columns

                # count this step
                round_count += 1

                # recall last step's _rows and columns indexes
                old_pick_rows = pick_rows.copy()
                old_pick_columns = pick_columns.copy()

        # if requested, save the converge report
        if save_converge_report != "":
            json.dump(converge_report.to_dict(), open(save_converge_report, "w"), indent=2)

        # full return logic
        if is_return_indexes:
            return best_rows, best_columns, converge_report
        return dataset.iloc[best_rows, best_columns], converge_report
