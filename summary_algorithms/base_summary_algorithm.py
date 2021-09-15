# library imports
import pandas as pd


class BaseSummary:
    """
    The basic summary algorithm
    """

    def __init__(self):
        pass

    @staticmethod
    def run(dataset: pd.DataFrame,
            desired_row_size: int,
            desired_col_size: int,
            evaluate_score_function,
            save_converge_report: str = "",
            row_score_function = None,
            is_return_indexes: bool = False,
            max_iter: int = -1) -> pd.DataFrame:
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
        pass
