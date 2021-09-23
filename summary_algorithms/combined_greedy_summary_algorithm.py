# library imports
import json
import time
import random
import collections
import pandas as pd

# project import
from ds.converge_report import ConvergeReport
from summary_algorithms.base_summary_algorithm import BaseSummary


class CombinedGreedySummary(BaseSummary):
    """
    This class is a greedy over the rows and columns in the same time type algorithm for dataset summarization.
    It extends logically the "GreedySummary" algorithm by picking both rows and columns in the same greedy loop
    """

    # GLOBAL PARMS #
    PREVENT_LOOP = True
    # END - GLOBAL PARMS #

    def __init__(self):
        BaseSummary.__init__(self)

    @staticmethod
    def run(dataset: pd.DataFrame,
            desired_row_size: int,
            desired_col_size: int,
            evaluate_score_function,
            save_converge_report: str = "",
            is_return_indexes: bool = False,
            max_iter: int = -1):
        """
        A row-columns greedy dataset summary algorithm
        :param dataset: the dataset we work on (pandas' dataframe)
        :param desired_row_size: the size of the summary as the number of _rows (int)
        :param desired_col_size: the size of the summary as the number of columns (int)
        :param evaluate_score_function: a function object getting dataset (pandas' dataframe) and summary (pandas' dataframe) and give a score (float) to the entire summary
        :param save_converge_report: a path to write the converge report to (default - do not write)
        :param is_return_indexes:  boolean flag to return summary's _rows indexes of after applying to the dataset itself
        :param max_iter: the maximum number of iteration we allow to do (default - unlimited)
        :return: the summary of the dataset with converge report (dict)
        """

        # setting the round count to the beginning of the process
        round_count = 1
        # if requested, init empty converge report
        converge_report = ConvergeReport()
        # init all the vars we need in the process
        old_pick_rows = []
        old_pick_columns = []
        pick_rows = list(range(dataset.shape[0]))  # all _rows
        pick_columns = list(range(dataset.shape[1]))  # all columns

        # when no other swap is taken place, this is the equilibrium and we can stop searching
        # Note: we introduce the "max_iter" stop condition in order stop the run after enough steps
        while (round_count < max_iter or max_iter == -1) and (collections.Counter(old_pick_rows) != collections.Counter(pick_rows) or collections.Counter(old_pick_columns) != collections.Counter(pick_columns)):
            # recall last step's _rows and columns indexes
            old_pick_rows = pick_rows.copy()
            old_pick_columns = pick_columns.copy()

            # optimize over the _rows
            start_rows_calc = time.time()  # just for time measurement tasks
            start_cols_calc = time.time()  # just for time measurement tasks
            pick_rows, pick_cols = CombinedGreedySummary._greedy_row_col_summary(dataset=dataset.iloc[:, old_pick_columns],
                                                                                 desired_row_size=desired_row_size,
                                                                                 desired_col_size=desired_col_size,
                                                                                 score_function=evaluate_score_function)
            end_rows_calc = time.time()  # just for time measurement tasks
            end_cols_calc = time.time()  # just for time measurement tasks

            # sort the indexes - just for easy review later
            pick_rows = sorted(pick_rows)
            pick_columns = sorted(pick_columns)

            # Add the data for the report
            converge_report.add_step(row=pick_rows,
                                     col=pick_columns,
                                     row_score=evaluate_score_function(dataset, dataset.iloc[pick_rows, old_pick_columns]),
                                     col_score=evaluate_score_function(dataset, dataset.iloc[old_pick_rows, pick_columns]),
                                     row_calc_time=(end_rows_calc - start_rows_calc),
                                     col_calc_time=(end_cols_calc - start_cols_calc),
                                     total_score=evaluate_score_function(dataset, dataset.iloc[pick_rows, pick_columns]))

            # in order to prevent loops, once found, we want to jump to other, random start condition
            if CombinedGreedySummary.PREVENT_LOOP and round_count >= 2:
                for previous_step in range(round_count - 1):
                    if pick_rows == converge_report.step_get("_rows", previous_step) and pick_columns == converge_report.step_get("_cols", previous_step):
                        # pick randomly new start
                        pick_rows = []
                        while len(pick_rows) < desired_row_size:
                            new_value = random.choice(list(range(dataset.shape[0])))
                            if new_value not in pick_rows:
                                pick_rows.append(new_value)
                        pick_columns = []
                        while len(pick_rows) < desired_col_size:
                            new_value = random.choice(list(range(dataset.shape[1])))
                            if new_value not in pick_rows:
                                pick_columns.append(new_value)

                        # sort just for easier review later
                        pick_rows = sorted(pick_rows)
                        pick_columns = sorted(pick_columns)

                        # add these changes in the converge report
                        converge_report.add_step(row=pick_rows,
                                                 col=pick_columns,
                                                 row_score=evaluate_score_function(dataset, dataset.iloc[pick_rows, old_pick_columns]),
                                                 col_score=evaluate_score_function(dataset, dataset.iloc[old_pick_rows, pick_columns]),
                                                 row_calc_time=0,  # TODO: maybe can be done better - later
                                                 col_calc_time=0,  # TODO: maybe can be done better - later
                                                 total_score=evaluate_score_function(dataset, dataset.iloc[pick_rows, pick_columns]))
                        break

            # count this step
            round_count += 1

        # if requested, save the converge report
        if save_converge_report != "":
            json.dump(converge_report.to_dict(), open(save_converge_report, "w"), indent=2)

        # full return logic
        if is_return_indexes:
            return pick_rows, pick_columns, converge_report
        return dataset.iloc[pick_rows, pick_columns], converge_report

    @staticmethod
    def _greedy_row_col_summary(dataset: pd.DataFrame,
                                desired_row_size: int,
                                desired_col_size: int,
                                score_function):
        """
        The greedy algorithm for only the _rows (columns when transposed matrix)
        :param dataset: the dataset we work on (pandas' dataframe)
        :param desired_row_size: the size of the summary as the number of _rows (int)
        :param desired_col_size: the size of the summary as the number of _cols (int)
        :param score_function: a function object getting dataset (pandas' dataframe) and summary (pandas' dataframe)
                                and give a score (float) to the summary
        """
        # find all the _rows indexes
        all_rows_indexes = set(list(range(dataset.shape[0])))
        # find all the _cols indexes
        all_cols_indexes = set(list(range(dataset.shape[1])))
        # init vars
        sample_rows_indexes = []
        sample_cols_indexes = []

        # run until we have the desired number of _rows in the summary
        for i in range(min(desired_row_size, desired_col_size)):
            # init to max to replace in the first time
            best_new_row_index = -1
            best_new_row_score = float("inf")
            # get only the relevant _rows
            search_rows_indexes = all_rows_indexes - set(sample_rows_indexes)
            # run over all relevant _rows and calc the score
            for check_row_index in search_rows_indexes:
                check_row_summary = sample_rows_indexes.copy()
                check_row_summary.append(check_row_index)
                if len(sample_cols_indexes) > 0:
                    check_summary_score = score_function(dataset=dataset,
                                                         summary=dataset.iloc[check_row_summary, sample_cols_indexes])
                else:

                    check_summary_score = score_function(dataset=dataset,
                                                         summary=dataset.iloc[check_row_summary, :])
                # if better score, we want this row to add
                if check_summary_score < best_new_row_score:
                    best_new_row_score = check_summary_score
                    best_new_row_index = check_row_index
            # add the best row to the summary and recall the best score for the converge report later
            sample_rows_indexes.append(best_new_row_index)

            # init to max to replace in the first time
            best_new_col_index = -1
            best_new_col_score = float("inf")
            # get only the relevant _rows
            search_cols_indexes = all_cols_indexes - set(all_cols_indexes)
            # run over all relevant _rows and calc the score
            for check_col_index in search_cols_indexes:
                check_col_summary = sample_cols_indexes.copy()
                check_col_summary.append(check_col_index)
                if len(sample_cols_indexes) > 0:
                    check_summary_score = score_function(dataset=dataset,
                                                         summary=dataset.iloc[sample_rows_indexes, check_col_summary])
                else:
                    check_summary_score = score_function(dataset=dataset,
                                                         summary=dataset.iloc[:, check_col_summary])
                # if better score, we want this row to add
                if check_summary_score < best_new_col_score:
                    best_new_col_score = check_summary_score
                    best_new_col_index = check_col_index
            # add the best row to the summary and recall the best score for the converge report later
            sample_cols_indexes.append(best_new_col_index)

        # run on the remaining set
        if desired_row_size > desired_col_size:
            for i in range(desired_row_size-desired_col_size):
                # init to max to replace in the first time
                best_new_row_index = -1
                best_new_row_score = float("inf")
                # get only the relevant _rows
                search_rows_indexes = all_rows_indexes - set(sample_rows_indexes)
                # run over all relevant _rows and calc the score
                for check_row_index in search_rows_indexes:
                    check_summary = sample_rows_indexes.copy()
                    check_summary.append(check_row_index)
                    check_summary_score = score_function(dataset=dataset,
                                                         summary=dataset.iloc[check_summary, sample_cols_indexes])
                    # if better score, we want this row to add
                    if check_summary_score < best_new_row_score:
                        best_new_row_score = check_summary_score
                        best_new_row_index = check_row_index
                # add the best row to the summary and recall the best score for the converge report later
                sample_rows_indexes.append(best_new_row_index)
        else:
            for i in range(desired_col_size-desired_row_size):
                # init to max to replace in the first time
                best_new_col_index = -1
                best_new_col_score = float("inf")
                # get only the relevant _rows
                search_cols_indexes = all_cols_indexes - set(sample_cols_indexes)
                # run over all relevant _rows and calc the score
                for check_row_index in search_cols_indexes:
                    check_col_summary = sample_rows_indexes.copy()
                    check_col_summary.append(check_row_index)
                    check_summary_score = score_function(dataset=dataset,
                                                         summary=dataset.iloc[sample_rows_indexes, check_col_summary])
                    # if better score, we want this row to add
                    if check_summary_score < best_new_col_score:
                        best_new_col_score = check_summary_score
                        best_new_col_index = check_row_index
                # add the best row to the summary and recall the best score for the converge report later
                sample_rows_indexes.append(best_new_col_index)

        # return answers
        return sample_rows_indexes, sample_cols_indexes
