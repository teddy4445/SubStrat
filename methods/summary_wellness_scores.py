# library imports
import math
import scipy
import numpy as np
import pandas as pd
from random import random, sample
from scipy.stats import entropy


class SummaryWellnessScores:
    """
    A static class with methods for evaluating the wellness of a summary.
    """

    def __init__(self):
        pass

    # MEAN PERFORMANCE-STABILITY METRICS #

    @staticmethod
    def mean_mean_entropy_stability(dataset: pd.DataFrame,
                                    summary: pd.DataFrame) -> float:
        """
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        return (SummaryWellnessScores.mean_entropy(dataset=dataset,
                                                   summary=summary) +
                SummaryWellnessScores.stability(dataset=dataset,
                                                summary=summary,
                                                performance_function=SummaryWellnessScores.mean_entropy))/2

    @staticmethod
    def hmean_mean_entropy_stability(dataset: pd.DataFrame,
                                    summary: pd.DataFrame) -> float:
        """
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        p = SummaryWellnessScores.mean_entropy(dataset=dataset,
                                               summary=summary)
        s = SummaryWellnessScores.stability(dataset=dataset,
                                            summary=summary,
                                            performance_function=SummaryWellnessScores.mean_entropy)
        return (2 * s * p)/(s+p)

    @staticmethod
    def mean_coefficient_of_anomaly_stability(dataset: pd.DataFrame,
                                              summary: pd.DataFrame) -> float:
        """
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        return (SummaryWellnessScores.coefficient_of_anomaly(dataset=dataset,
                                                             summary=summary) +
                SummaryWellnessScores.stability(dataset=dataset,
                                                summary=summary,
                                                performance_function=SummaryWellnessScores.coefficient_of_anomaly))/2

    @staticmethod
    def hmean_coefficient_of_anomaly_stability(dataset: pd.DataFrame,
                                               summary: pd.DataFrame) -> float:
        """
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        p = SummaryWellnessScores.coefficient_of_anomaly(dataset=dataset,
                                                         summary=summary)
        s = SummaryWellnessScores.stability(dataset=dataset,
                                            summary=summary,
                                            performance_function=SummaryWellnessScores.coefficient_of_anomaly)
        return (2 * s * p)/(s+p)

    @staticmethod
    def mean_coefficient_of_variation_stability(dataset: pd.DataFrame,
                                              summary: pd.DataFrame) -> float:
        """
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        return (SummaryWellnessScores.coefficient_of_variation(dataset=dataset,
                                                               summary=summary) +
                SummaryWellnessScores.stability(dataset=dataset,
                                                summary=summary,
                                                performance_function=SummaryWellnessScores.coefficient_of_variation))/2

    @staticmethod
    def hmean_coefficient_of_variation_stability(dataset: pd.DataFrame,
                                                summary: pd.DataFrame) -> float:
        """
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        p = SummaryWellnessScores.coefficient_of_variation(dataset=dataset,
                                                         summary=summary)
        s = SummaryWellnessScores.stability(dataset=dataset,
                                            summary=summary,
                                            performance_function=SummaryWellnessScores.coefficient_of_variation)
        return (2 * s * p)/(s+p)

    # END - MEAN PERFORMANCE-STABILITY METRICS #

    @staticmethod
    def mean_metrics(dataset: pd.DataFrame,
                     summary: pd.DataFrame,
                     property_function,
                     distance_metric_1,
                     distance_metric_2) -> float:
        """
        A general approach to the similarity between a dataset and a summary
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :param property_function: a function object that gets (pandas' dataframe) and returns a property of the matrix
        :param distance_metric_1: a function object that gets two arguments and returns a float as the distance between them
        :param distance_metric_2: a function object that gets two arguments and returns a float as the distance between them
        :return: the score between them ranging (0, inf)
        """
        try:
            return (distance_metric_1(property_function(dataset), property_function(summary)) + distance_metric_2(property_function(dataset), property_function(summary)))/2
        except:
            return 0

    @staticmethod
    def harmonic_mean_metrics(dataset: pd.DataFrame,
                              summary: pd.DataFrame,
                              property_function,
                              distance_metric_1,
                              distance_metric_2) -> float:
        """
        A general approach to the similarity between a dataset and a summary
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :param property_function: a function object that gets (pandas' dataframe) and returns a property of the matrix
        :param distance_metric_1: a function object that gets two arguments and returns a float as the distance between them
        :param distance_metric_2: a function object that gets two arguments and returns a float as the distance between them
        :return: the score between them ranging (0, inf)
        """
        try:
            val_1 = distance_metric_1(property_function(dataset), property_function(summary))
            val_2 = distance_metric_2(property_function(dataset), property_function(summary))
            return (2 * val_1 * val_2) / (val_1 + val_2)
        except:
            return 0

    @staticmethod
    def general_approach_metric(dataset: pd.DataFrame,
                                summary: pd.DataFrame,
                                property_function,
                                distance_metric) -> float:
        """
        A general approach to the similarity between a dataset and a summary
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :param property_function: a function object that gets (pandas' dataframe) and returns a property of the matrix
        :param distance_metric: a function object that gets two arguments and returns a float as the distance between them
        :return: the score between them ranging (0, inf)
        """
        try:
            return distance_metric(property_function(dataset), property_function(summary))
        except:
            return 0

    @staticmethod
    def stability(dataset: pd.DataFrame,
                  summary: pd.DataFrame,
                  performance_function,
                  noise: float = 0.05,
                  folds: int = 3) -> float:
        """
        The Lyapunov stability metric between the dataset and its summary in some measurement
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :param performance_function: the metric used to evaluate the summary goodness
        :param noise: the hyperparamter to the noise added to the dataset
        :param folds: the number of folds we do to get an approximation to the stability measurement
        :return: the score between them ranging (0, inf)
        """
        scores = []
        for fold in range(folds):
            noised_dataset = SummaryWellnessScores.add_dataset_subset_pick_noise(dataset=dataset,
                                                                                 noise=noise)
            # calc stability score
            try:
                stability_score = abs(performance_function(dataset=noised_dataset, summary=summary) / performance_function(dataset=dataset, summary=summary))
            except:
                stability_score = abs(performance_function(dataset=noised_dataset, summary=summary))
            scores.append(stability_score)
        return np.nanmean(scores)


    @staticmethod
    def mean_entropy(dataset: pd.DataFrame,
                     summary: pd.DataFrame) -> float:
        """
        The L1 (Manhattan) distance between the mean entropy of both the dataset and its summary
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        return SummaryWellnessScores.general_approach_metric(dataset=dataset,
                                                             summary=summary,
                                                             property_function=SummaryWellnessScores._mean_entropy,
                                                             distance_metric=SummaryWellnessScores._l1)

    @staticmethod
    def mean_pearson_corr(dataset: pd.DataFrame,
                          summary: pd.DataFrame) -> float:
        """
        The L1 (Manhattan) distance between the mean Pearson correlation of the matrices
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, 1)
        """
        return SummaryWellnessScores.general_approach_metric(dataset=dataset,
                                                             summary=summary,
                                                             property_function=SummaryWellnessScores._matrix_mean_pearson_correlation,
                                                             distance_metric=SummaryWellnessScores._l1)

    @staticmethod
    def coefficient_of_variation(dataset: pd.DataFrame,
                                 summary: pd.DataFrame) -> float:
        """
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        return SummaryWellnessScores.general_approach_metric(dataset=dataset,
                                                             summary=summary,
                                                             property_function=SummaryWellnessScores._average_coefficient_of_variation_of_feature,
                                                             distance_metric=SummaryWellnessScores._l1)

    @staticmethod
    def coefficient_of_anomaly(dataset: pd.DataFrame,
                               summary: pd.DataFrame) -> float:
        """
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, inf)
        """
        return SummaryWellnessScores.general_approach_metric(dataset=dataset,
                                                             summary=summary,
                                                             property_function=SummaryWellnessScores._average_coefficient_of_anomaly_of_feature,
                                                             distance_metric=SummaryWellnessScores._l1)

    @staticmethod
    def coverage(dataset: pd.DataFrame,
                 summary: pd.DataFrame) -> float:
        """
        :param dataset: the dataset (pandas' dataframe)
        :param summary: the summary of the dataset (pandas' dataframe)
        :return: the score between them ranging (0, 1)
        """
        answer = []
        for column in summary.columns:
            try:
                vc = dataset[column].value_counts()
                answer.append(sum([vc[b] for b in summary[column].unique()])/vc.sum())
            except:
                pass
        return 1 - np.nanmean(answer)

    # PROPERTY FUNCTIONS #

    @staticmethod
    def _mean_entropy(matrix: pd.DataFrame):
        """
        :param matrix: the matrix we want to evaluate
        :return: the mean entropy of the matrix columns (features)
        """
        return np.nanmean([entropy(matrix[col]) for col in list(matrix)])

    @staticmethod
    def _matrix_mean_pearson_correlation(matrix: pd.DataFrame):
        """
        :param matrix: the matrix we want to evaluate
        :return: It measures the average value of Pearson’s correlation coefficient between different features.
        """
        n = len(list(matrix))
        cols = list(matrix)
        return 2 * sum([np.nansum([scipy.stats.pearsonr(matrix[cols[column_i]],
                                                        matrix[cols[column_j]])[0]
                                   for column_j in range(column_i + 1, len(cols))])
                        for column_i in range(len(cols) - 1)]) / (n * (n - 1))

    @staticmethod
    def _average_coefficient_of_variation_of_feature(matrix: pd.DataFrame):
        """
        :param matrix: the matrix we want to evaluate
        :return: It measures the STD coefficient of variation by the ratio of the standard deviation and the mean of the feature values.
        """
        return np.nanmean([np.nanstd(matrix[column]) / np.nanmean(matrix[column]) if np.nanmean(
            matrix[column]) > 0 else np.nanstd(matrix[column]) for column in list(matrix)])

    @staticmethod
    def _average_coefficient_of_anomaly_of_feature(matrix: pd.DataFrame):
        """
        :param matrix: the matrix we want to evaluate
        :return: It measures the average coefficient of anomaly by the ratio of the mean and the standard deviation of the feature values.
        """
        return np.nanmean([np.nanmean(matrix[column]) / np.nanstd(matrix[column]) if np.nanstd(matrix[column]) > 0 else np.nanmean(matrix[column]) for column in list(matrix)])

    # END - PROPERTY FUNCTIONS #

    # DISTANCE FUNCTIONS #

    @staticmethod
    def _l1(value1: float,
            value2: float):
        return math.fabs(value1 - value2)

    @staticmethod
    def _vector_l1(value1: list,
                   value2: list):
        return sum([math.fabs(value1[i] - value2[i]) for i in range(len(value1))])

    @staticmethod
    def _vector_l2(value1: list,
                   value2: list):
        return math.sqrt(sum([math.pow(value1[i] - value2[i], 2) for i in range(len(value1))]))

    # END - DISTANCE FUNCTIONS #

    # NOISE FUNCTIONS #

    @staticmethod
    def add_dataset_subset_pick_noise(dataset: pd.DataFrame,
                                      noise: float) -> pd.DataFrame:
        """
        A noise function that takes a random subset of size (1-noise) from the original dataset for both the rows and columns
        :param dataset: The dataset we want to add noise on
        :param noise: the noise hyper-parameter
        :return: noisy dataframe as Pandas' DataFrame
        """
        row_indexes = list(range(dataset.shape[0]))
        col_indexes = list(range(dataset.shape[1]))
        noisy_dataset = dataset.iloc[sample(row_indexes, round((1 - noise) * len(row_indexes))), sample(col_indexes, round((1 - noise) * len(col_indexes)))]
        noisy_dataset.reset_index(inplace=True)
        return noisy_dataset

    @staticmethod
    def add_dataset_gaussian_noise(dataset: pd.DataFrame,
                                   noise: float) -> pd.DataFrame:
        """
        A noise function that takes the dataset and adds to each value a Gaussian noise with a given STD = noise
        :param dataset: The dataset we want to add noise on
        :param noise: the noise hyper-parameter
        :return: noisy dataframe as Pandas' DataFrame
        """
        return dataset + np.random.normal(0, noise, [dataset.shape[0], dataset.shape[1]])

    # END - NOISE FUNCTIONS #
