# library imports
import math
import numpy as np
import pandas as pd
from scipy.stats import entropy


class SummaryScoreFunctions:
    """
    A static class with methods for scoring dataset's summary
    """

    def __init__(self):
        pass

    @staticmethod
    def row_entropy(dataset: pd.DataFrame,
                    summary: pd.DataFrame) -> float:
        """

        :param dataset:
        :param summary:
        :return:
        """
        # make sure the number of columns are identical
        if dataset.shape[1] != summary.shape[1]:
            raise Exception("Error at SummaryScoreFunctions.row_entropy: the method assumes the"
                            " dataset and summary have the same number of columns")
        # calc column-entropy
        return math.sqrt(sum([math.pow(entropy(dataset[col]) - entropy(summary[col]), 2)
                              for col in list(dataset)]))
