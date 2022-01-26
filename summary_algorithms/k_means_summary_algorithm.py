# library imports
import json
import time
import random
import pandas as pd
from itertools import combinations

# learning model
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# project import
from ds.converge_report import ConvergeReport
from summary_algorithms.base_summary_algorithm import BaseSummary


class KmeansSummary(BaseSummary):
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
            max_iter: int = 10):
        """
        A row-columns greedy dataset summary algorithm
        :param dataset: the dataset we work on (pandas' dataframe)
        :param desired_row_size: the size of the summary as the number of _rows (int)
        :param desired_col_size: the size of the summary as the number of columns (int)
        :param row_score_function: the distance function used as part of the K-means algorithm
        :param evaluate_score_function: a function object getting dataset (pandas' dataframe) and summary (pandas' dataframe) and give a score (float) to the entire summary
        :param save_converge_report: a path to write the converge report to (default - do not write)
        :param is_return_indexes:  boolean flag to return summary's _rows indexes of after applying to the dataset itself
        :param max_iter: the number of iteration we allow to do for the K-means algorithm (default - 10)
        :return: the summary of the dataset with converge report (dict)
        """
        # make sure we have steps to run
        if max_iter < 1:
            raise Exception("Error at KmeansSummary.run: the max_iter argument must be larger than 1")

        # break dataset into rows
        samples = dataset.values.tolist()
        # make model
        model = KMeans(n_clusters=desired_row_size,
                       n_init=max_iter,
                       algorithm=row_score_function if row_score_function is None else "elkan")
        # find the centers
        model.fit(samples)
        # find the closest samples to each centroid
        closest_indexes, _ = pairwise_distances_argmin_min(model.cluster_centers_, samples)
        # build the summary table

        # full return logic
        best_columns = list(range(len(list(dataset))))
        if is_return_indexes:
            return list(closest_indexes), best_columns, None
        return dataset.iloc[list(closest_indexes), best_columns], None
