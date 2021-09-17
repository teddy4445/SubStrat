# library imports
import random

# project imports
from summary_algorithms.ga.gene import SummaryGene


class SummaryGenePopulation:
    """
    A data class for population of summary genes with the main GA operations on them
    """

    def __init__(self,
                 row_count: int,
                 col_count: int,
                 genes: list = None):
        self._row_count = row_count
        self._col_count = col_count
        self._genes = genes if isinstance(genes, list) else []
        self._scores = [random.random() for _ in range(len(self._genes))]

    @staticmethod
    def random_population(row_count: int,
                          col_count: int,
                          summary_rows: int,
                          summary_cols: int,
                          population_size: int):
        row_set = list(range(row_count))  # all _rows
        col_set = list(range(col_count))  # all columns
        return SummaryGenePopulation(row_count=row_count,
                                     col_count=col_count,
                                     genes=[SummaryGene(rows=random.sample(row_set, k=summary_rows),
                                                        cols=random.sample(col_set, k=summary_cols))
                                            for _ in range(population_size)])

    # getters #

    def get_scores(self):
        return self._scores

    def get_best_gene(self) -> SummaryGene:
        return self._genes[self._scores.index(min(self._scores))]

    # end - getters #

    # logic #

    def fitness(self,
                dataset,
                fitness_function):
        scores = []
        for gene in self._genes:
            try:
                scores.append(fitness_function(dataset, gene.get_summary(dataset=dataset)))
            except:
                scores.append(-1)
        max_score = max(scores)
        scores = [score if score != -1 else max_score + 1 for score in scores]
        self._scores = scores

    def selection(self,
                  royalty_rate: float):
        # convert scores to probability
        max_fitness = max(self._scores)
        reverse_scores = [max_fitness - score for score in self._scores]
        sum_fitness = sum(reverse_scores)
        if sum_fitness > 0:
            fitness_probabilities = [score / sum_fitness for score in reverse_scores]
        else:
            fitness_probabilities = reverse_scores
        # sort the population by fitness
        genes_with_fitness = zip(fitness_probabilities, self._genes)
        genes_with_fitness = sorted(genes_with_fitness, key=lambda x: x[0], reverse=True)
        # pick the best royalty_rate anyway
        royalty = [val[1] for val in genes_with_fitness[:round(len(genes_with_fitness)*royalty_rate)]]
        # tournament around the other genes
        left_genes = [val[1] for val in genes_with_fitness[round(len(genes_with_fitness) * royalty_rate):]]
        left_fitness = [val[0] for val in genes_with_fitness[round(len(genes_with_fitness) * royalty_rate):]]
        pick_genes = []
        left_count = len(self._genes) - len(royalty)
        while len(pick_genes) < left_count:
            pick_gene = random.choices(left_genes, weights=left_fitness)
            pick_genes.append(pick_gene)
        # add the royalty
        pick_genes = list(pick_genes)
        pick_genes.extend(royalty)
        return pick_genes

    def crossover(self):
        # init source and target lists
        target_length = len(self._genes)
        new_genes = []

        # run over the population and get children
        while len(new_genes) < target_length:
            # pick two random parents in the population
            parent_gene_1 = self._genes.pop(random.randrange(len(self._genes)))
            parent_gene_2 = self._genes.pop(random.randrange(len(self._genes)))
            # get their children #
            # pick random split location in the rows and columns
            change_rows_index = round(random.random() * parent_gene_1.get_row_count())
            change_cols_index = round(random.random() * parent_gene_1.get_col_count())
            # create rows
            child_gene_1_rows = parent_gene_1.get_rows()[:change_rows_index]
            child_gene_1_rows.extend(parent_gene_2.get_rows()[change_rows_index:])
            child_gene_2_rows = parent_gene_2.get_rows()[:change_rows_index]
            child_gene_2_rows.extend(parent_gene_1.get_rows()[change_rows_index:])
            # create colums
            child_gene_1_cols = parent_gene_1.get_columns()[:change_cols_index]
            child_gene_1_cols.extend(parent_gene_2.get_columns()[change_cols_index:])
            child_gene_2_cols = parent_gene_2.get_columns()[:change_cols_index]
            child_gene_2_cols.extend(parent_gene_1.get_columns()[change_cols_index:])
            # wrap with the classes
            child_gene_1 = SummaryGene(rows=child_gene_1_rows,
                                       cols=child_gene_1_cols)
            child_gene_2 = SummaryGene(rows=child_gene_2_rows,
                                       cols=child_gene_2_cols)
            # add the children to the new population
            new_genes.append(child_gene_1)
            new_genes.append(child_gene_2)

        # replace to new gene population and return answer
        self._genes = new_genes

    def mutation(self,
                 mutation_rate: float):
        [gene.mutation(max_row_index=self._row_count,
                       max_col_index=self._col_count,
                       mutation_rate=mutation_rate)
         for gene in self._genes]

    # end - logic #

    def __repr__(self):
        return "<Genetic summaries population>"

    def __str__(self):
        return "<Genetic summaries population | size = {}>".format(len(self._genes))
