# library imports
import math
import numpy as np
import pandas as pd
from scipy.stats import entropy


class SummaryWellnessScores:
    """
    A static class with methods for evaluating the wellness of a summary.
    """

    def __init__(self):
        pass

    @staticmethod
    def mean_entropy(dataset: pd.DataFrame,
                     summary: pd.DataFrame) -> float:
        """
        The L1 (Manhattan) distance between the mean entropy of both the dataset and its summary
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        # calc dataset entropy
        dataset_entropy = np.mean([entropy(dataset[col]) for col in list(dataset)])
        # calc summary entropy
        summary_entropy = np.mean([entropy(summary[col]) for col in list(summary)])
        # return L1 distance between them
        return math.fabs(dataset_entropy - summary_entropy)