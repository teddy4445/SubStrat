# library imports
import random
import pandas as pd

# project imports
from summary_algorithms.ga.gene_population import SummaryGenePopulation


class StablerSummaryGenePopulation(SummaryGenePopulation):
    """
    A data class for population of summary genes with the main GA operations on them.
    This class introduce heuristic function to the score provided by the user to obtain a more stable
    results compared to the original (SummaryGenePopulation) class
    """

    def __init__(self,
                 row_count: int,
                 col_count: int,
                 genes: list = None):
        super(StablerSummaryGenePopulation, self).__init__(row_count=row_count,
                                                           col_count=col_count,
                                                           genes=genes)

    # logic #

    def fitness(self,
                dataset,
                fitness_function):
        scores = []
        for gene in self._genes:
            try:
                scores.append((fitness_function(dataset, gene.get_summary(dataset=dataset)) + StablerSummaryGenePopulation.stable_property_fitness(dataset, gene.get_summary(dataset=dataset)))/2)
            except:
                scores.append(-1)
        max_score = max(scores)
        scores = [score if score != -1 else max_score + 1 for score in scores]
        self._scores = scores

    @staticmethod
    def stable_property_fitness(dataset: pd.DataFrame,
                                sumarry: pd.DataFrame) -> float:
        """
        :param dataset: the dataset (pandas' dataframe)
        :param sumarry: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        pass

    # end - logic #

    def __repr__(self):
        return "<Stabler Genetic summaries population>"

    def __str__(self):
        return "<Stabler Genetic summaries population | size = {}>".format(len(self._genes))
