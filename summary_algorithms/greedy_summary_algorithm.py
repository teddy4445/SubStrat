# library imports
import json
import time
import random
import collections
import pandas as pd

# project import
from ds.converge_report import ConvergeReport
from summary_algorithms.base_summary_algorithm import BaseSummary


class GreedySummary(BaseSummary):
    """
    This class is a wrapper over the greedy summary algorithm
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
            row_score_function = None,
            is_return_indexes: bool = False,
            max_iter: int = -1):
        """
        A row-columns greedy dataset summary algorithm
        :param dataset: the dataset we work on (pandas' dataframe)
        :param desired_row_size: the size of the summary as the number of rows (int)
        :param desired_col_size: the size of the summary as the number of columns (int)
        :param row_score_function: a function object getting dataset (pandas' dataframe) and summary (pandas' dataframe) and give the score of row\column optimization process
        :param evaluate_score_function: a function object getting dataset (pandas' dataframe) and summary (pandas' dataframe) and give a score (float) to the entire summary
        :param save_converge_report: a path to write the converge report to (default - do not write)
        :param is_return_indexes:  boolean flag to return summary's rows indexes of after applying to the dataset itself
        :param max_iter: the maximum number of iteration we allow to do (default - unlimited)
        :return: the summary of the dataset with converge report (dict)
        """
        # just to make sure we have the row_function and evaluate_score function to be the same one
        if row_score_function is None:
            row_score_function = evaluate_score_function

        # setting the round count to the beginning of the process
        round_count = 1
        # if requested, init empty converge report
        converge_report = ConvergeReport()
        # init all the vars we need in the process
        old_pick_rows = []
        old_pick_columns = []
        pick_rows = list(range(dataset.shape[0]))  # all rows
        pick_columns = list(range(dataset.shape[1]))  # all columns
        # calc once the transpose of the dataset (matrix) for the second step in each iteration
        dataset_transposed = dataset.transpose()

        # when no other swap is taken place, this is the equilibrium and we can stop searching
        # Note: we introduce the "max_iter" stop condition in order stop the run after enough steps
        while (round_count < max_iter or max_iter == -1) and (collections.Counter(old_pick_rows) != collections.Counter(pick_rows) or collections.Counter(old_pick_columns) != collections.Counter(pick_columns)):
            # recall last step's rows and columns indexes
            old_pick_rows = pick_rows.copy()
            old_pick_columns = pick_columns.copy()

            # optimize over the rows
            start_rows_calc = time.time()  # just for time measurement tasks
            pick_rows, rows_score = GreedySummary._greedy_row_summary(dataset=dataset.iloc[:, old_pick_columns],
                                                                      desired_row_size=desired_row_size,
                                                                      score_function=row_score_function,
                                                                      is_return_indexes=True)
            end_rows_calc = time.time()  # just for time measurement tasks

            # optimize over the columns
            start_cols_calc = time.time()  # just for time measurement tasks
            pick_columns, cols_score = GreedySummary._greedy_row_summary(dataset=dataset_transposed.iloc[:, old_pick_rows],
                                                                         desired_row_size=desired_col_size,
                                                                         score_function=row_score_function,
                                                                         is_return_indexes=True)
            end_cols_calc = time.time()  # just for time measurement tasks

            # sort the indexes - just for easy review later
            pick_rows = sorted(pick_rows)
            pick_columns = sorted(pick_columns)

            # Add the data for the report
            converge_report.add_step(row=pick_rows,
                                     col=pick_columns,
                                     row_score=rows_score,
                                     col_score=cols_score,
                                     row_calc_time=(end_rows_calc - start_rows_calc),
                                     col_calc_time=(end_cols_calc - start_cols_calc),
                                     total_score=evaluate_score_function(dataset, dataset.iloc[pick_rows, pick_columns]))

            # in order to prevent loops, once found, we want to jump to other, random start condition
            if GreedySummary.PREVENT_LOOP and round_count >= 2:
                for previous_step in range(round_count - 1):
                    if pick_rows == converge_report.step_get("rows", previous_step) and pick_columns == converge_report.step_get("cols", previous_step):
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
                                                 row_score=row_score_function(dataset, dataset.iloc[pick_rows, :]),
                                                 col_score=row_score_function(dataset_transposed, dataset_transposed.iloc[pick_columns, :]),
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
    def _greedy_row_summary(dataset: pd.DataFrame,
                            desired_row_size: int,
                            score_function,
                            is_return_indexes: bool = False):
        """
        The greedy algorithm for only the rows (columns when transposed matrix)
        :param dataset: the dataset we work on (pandas' dataframe)
        :param desired_row_size: the size of the summary as the number of rows (int)
        :param score_function: a function object getting dataset (pandas' dataframe) and summary (pandas' dataframe)
                                and give a score (float) to the summary
        :param is_return_indexes: boolean flag to return summary's rows indexes of after applying to the dataset itself
        :return: summary and converge report mean score (over all the rows)
        """
        # find all the rows indexes
        all_rows_indexes = set(list(range(dataset.shape[0])))
        # init vars
        sample_rows_indexes = []
        best_scores = []
        # run until we have the desired number of rows in the summary
        for i in range(desired_row_size):
            # init to max to replace in the first time
            best_new_row_index = -1
            best_new_row_score = float("inf")
            # get only the relevant rows
            search_rows_indexes = all_rows_indexes - set(sample_rows_indexes)
            # run over all relevant rows and calc the score
            for check_row_index in search_rows_indexes:
                check_summary = sample_rows_indexes.copy()
                check_summary.append(check_row_index)
                check_summary_score = score_function(dataset=dataset,
                                                     summary=dataset.iloc[check_summary, :])
                # if better score, we want this row to add
                if check_summary_score < best_new_row_score:
                    best_new_row_score = check_summary_score
                    best_new_row_index = check_row_index
            # add the best row to the summary and recall the best score for the converge report later
            sample_rows_indexes.append(best_new_row_index)
            best_scores.append(best_new_row_score)

        # return answers
        final_summary = dataset.iloc[sample_rows_indexes, :]
        summary_score = score_function(dataset, final_summary)
        if is_return_indexes:
            return sample_rows_indexes, summary_score
        return final_summary, summary_score
