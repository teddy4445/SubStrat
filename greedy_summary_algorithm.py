# library imports
import json
import collections
import pandas as pd


def greedy_summary(dataset: pd.DataFrame,
                   desired_row_size: int,
                   desired_col_size: int,
                   row_score_function,
                   is_return_indexes: bool = False,
                   save_converge_report: str = ""):
    """

    :param dataset:
    :param desired_row_size:
    :param desired_col_size:
    :param row_score_function:
    :param is_return_indexes:
    :param save_converge_report:
    :return:
    """
    # TODO: REMOVE LATER - JUST FOR DEBUG
    round_count = 1

    # if requested, init empty converge report
    if save_converge_report != "":
        converge_report = {"rows": [],
                           "cols": []}

    dataset_transposed = dataset.transpose()
    old_pick_rows = []
    old_pick_columns = []
    pick_rows = list(range(dataset.shape[0]))  # all rows
    pick_columns = list(range(dataset.shape[1]))  # all columns
    # when no other swap is taken place, this is the equilibrium and we can stop searching
    while collections.Counter(old_pick_rows) != collections.Counter(pick_rows) \
            or collections.Counter(old_pick_columns) != collections.Counter(pick_columns):
        # TODO: REMOVE LATER - JUST FOR DEBUG
        print("greedy_summary: we are starting with round #{}".format(round_count))

        old_pick_rows = pick_rows.copy()
        old_pick_columns = pick_columns.copy()
        pick_rows = greedy_row_summary(dataset=dataset.iloc[:, old_pick_columns],
                                       desired_row_size=desired_row_size,
                                       score_function=row_score_function,
                                       is_return_indexes=True)
        # TODO: REMOVE LATER - JUST FOR DEBUG
        print("\n{}\n".format("-"*100), end="")

        pick_columns = greedy_row_summary(dataset=dataset_transposed.iloc[:, old_pick_rows],
                                          desired_row_size=desired_col_size,
                                          score_function=row_score_function,
                                          is_return_indexes=True)
        # just for easy review later
        pick_rows = sorted(pick_rows)
        pick_columns = sorted(pick_columns)

        # if requested, add the data for the report
        if save_converge_report != "":
            converge_report["rows"].append(pick_rows)
            converge_report["cols"].append(pick_columns)

        # TODO: REMOVE LATER - JUST FOR DEBUG
        print("\n\nRound #{}\nPick rows = {}\nPick columns = {}".format(round_count, pick_rows, pick_columns), end="\n")
        round_count += 1
    # TODO: REMOVE LATER - JUST FOR DEBUG
    print("\n\n{}\nPick rows = {}\nPick columns = {}\n{}\n\n".format("-" * 100,
                                                                     pick_rows,
                                                                     pick_columns,
                                                                     "-" * 100), end="")
    # if requested, save the converge report
    if save_converge_report != "":
        json.dump(converge_report, open(save_converge_report, "w"), indent=2)

    # full return logic
    if save_converge_report:
        if is_return_indexes:
            return pick_rows, pick_columns, converge_report
        return dataset.iloc[pick_rows, pick_columns], converge_report
    else:
        if is_return_indexes:
            return pick_rows, pick_columns
        return dataset.iloc[pick_rows, pick_columns]


def greedy_row_summary(dataset: pd.DataFrame,
                       desired_row_size: int,
                       score_function,
                       is_return_indexes: bool = False):
    """

    :param dataset:
    :param desired_row_size:
    :param score_function:
    :param is_return_indexes:
    :return:
    """
    all_rows_indexes = set(list(range(dataset.shape[0])))
    sample_rows_indexes = []
    for i in range(desired_row_size):
        best_new_row_index = -1
        best_new_row_score = float("inf")
        search_rows_indexes = all_rows_indexes - set(sample_rows_indexes)
        for check_row_index in search_rows_indexes:
            check_summary = sample_rows_indexes.copy()
            check_summary.append(check_row_index)
            check_summary_score = score_function(dataset=dataset,
                                                 summary=dataset.iloc[check_summary, :])
            if check_summary_score < best_new_row_score:
                best_new_row_score = check_summary_score
                best_new_row_index = check_row_index
        sample_rows_indexes.append(best_new_row_index)

        # TODO: REMOVE LATER - JUST FOR DEBUG
        print("greedy_row_summary: we have {}/{} ({:.2f}%) of the summary we need for dataset size = {}".format(i,
                                                                                                                desired_row_size,
                                                                                                                i / desired_row_size * 100,
                                                                                                                dataset.shape))
    if is_return_indexes:
        return sample_rows_indexes
    return dataset.iloc[sample_rows_indexes, :]
