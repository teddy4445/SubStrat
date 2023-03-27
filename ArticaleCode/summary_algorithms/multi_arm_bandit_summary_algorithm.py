# library imports
import numpy as np
import pandas as pd

# project import
from ds.converge_report import ConvergeReport
from summary_algorithms.base_summary_algorithm import BaseSummary
from summary_algorithms.ga.gene_population import SummaryGenePopulation, SummaryGene


class MultiArmBanditSummary(BaseSummary):
    """
    This class is a Multi arm Bandit type algorithm for dataset summarization
    """

    # SETTINGS #
    steps = 3000
    epsilon = 0.01
    # END - SETTINGS #

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

        # make sure we have steps to run
        if max_iter < 1:
            raise Exception("Error at MultiArmBanditSummary.run: the max_iter argument must be larger than 1")

        # if requested, init empty converge report
        converge_report = ConvergeReport()

        # init random population
        num_bandits=dataset.shape[0] + dataset.shape[1]
        choices = []
        wins = np.zeros(num_bandits)
        pulls = np.zeros(num_bandits)
        for n in range(max_iter):
            if np.random.choice() < MultiArmBanditSummary.epsilon:
                choice = np.argmax(wins / (pulls + 0.1))
            else:
                choice = np.random.choice(list(set(range(len(wins))) - {np.argmax(wins / (pulls + 0.1))}))
            choices.append(choice)
            payout = evaluate_score_function(dataset, dataset.iloc[choice,:] if choice < dataset.shape[0] else dataset.iloc[:,choice-dataset.shape[0]])
            wins[choice] += payout
            pulls[choice] += 1
        # get best
        best_rows = []
        best_columns = []
        added_count = 0
        while added_count < desired_row_size + desired_col_size:
            next_index = np.argmax(wins / (pulls + 0.1))
            if next_index < dataset.shape[0] and len(best_rows) < desired_row_size:
                best_rows.append(next_index)
                added_count += 1
            elif len(best_columns) < desired_col_size:
                best_columns.append(next_index)
                added_count += 1
            wins = np.delete(wins, next_index)
            pulls = np.pop(pulls, next_index)

        # full return logic
        if is_return_indexes:
            return best_rows, best_columns, converge_report
        return dataset.iloc[best_rows, best_columns], converge_report
