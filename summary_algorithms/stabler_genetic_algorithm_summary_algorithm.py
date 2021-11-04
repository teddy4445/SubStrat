# library imports
import json
import time
import random
import collections
import pandas as pd

# project import
from ds.converge_report import ConvergeReport
from summary_algorithms.base_summary_algorithm import BaseSummary
from summary_algorithms.ga.gene_population import SummaryGenePopulation, SummaryGene


class StablerGeneticSummary(BaseSummary):
    """
    This class is a genetic-coding type algorithm for dataset summarization
    """

    # SETTINGS #
    MUTATION_RATE = 0.05
    POPULATION_SIZE = 50
    ROYALTY_RATE = 0.02
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
            raise Exception("Error at LasVegasSummary.run: the max_iter argument must be larger than 1")

        # setting the round count to the beginning of the process
        round_count = 1
        # if requested, init empty converge report
        converge_report = ConvergeReport()

        # init all the vars we need in the process
        best_score = 9999  # TODO: can be done better
        best_rows = []
        best_columns = []

        # init random population
        gene_population = SummaryGenePopulation.random_population(row_count=dataset.shape[0],
                                                                  col_count=dataset.shape[1],
                                                                  summary_rows=desired_row_size,
                                                                  summary_cols=desired_col_size,
                                                                  population_size=StablerGeneticSummary.POPULATION_SIZE)

        # for metric eval
        old_pick_rows = list(range(dataset.shape[0]))
        old_pick_columns = list(range(dataset.shape[1]))

        while round_count <= max_iter:
            # optimize over the columns and rows
            start_rows_calc = time.time()  # just for time measurement tasks
            start_cols_calc = time.time()  # just for time measurement tasks
            gene_population.selection(royalty_rate=StablerGeneticSummary.ROYALTY_RATE)
            gene_population.crossover()
            gene_population.mutation(mutation_rate=StablerGeneticSummary.MUTATION_RATE)
            gene_population.fitness(dataset=dataset,
                                    fitness_function=evaluate_score_function)
            best_gene = gene_population.get_best_gene()
            end_rows_calc = time.time()  # just for time measurement tasks
            end_cols_calc = time.time()  # just for time measurement tasks

            # compute scores
            rows_summary_score = evaluate_score_function(dataset, dataset.iloc[best_gene.get_rows(), old_pick_columns])
            cols_summary_score = evaluate_score_function(dataset, dataset.iloc[old_pick_rows, best_gene.get_columns()])
            total_score = evaluate_score_function(dataset, dataset.iloc[best_gene.get_rows(), best_gene.get_columns()])

            # Add the data for the report
            converge_report.add_step(row=best_gene.get_rows(),
                                     col=best_gene.get_columns(),
                                     row_score=rows_summary_score,
                                     col_score=cols_summary_score,
                                     row_calc_time=(end_rows_calc - start_rows_calc)/2,
                                     col_calc_time=(end_cols_calc - start_cols_calc)/2,
                                     total_score=total_score)

            # check we this summary is better
            if total_score < best_score:
                best_score = total_score
                best_rows = best_gene.get_rows()
                best_columns = best_gene.get_columns()

            # count this step
            round_count += 1

            # recall last step's _rows and columns indexes
            old_pick_rows = best_gene.get_rows().copy()
            old_pick_columns = best_gene.get_columns().copy()

        # if requested, save the converge report
        if save_converge_report != "":
            json.dump(converge_report.to_dict(), open(save_converge_report, "w"), indent=2)

        # full return logic
        if is_return_indexes:
            return best_rows, best_columns, converge_report
        return dataset.iloc[best_rows, best_columns], converge_report
