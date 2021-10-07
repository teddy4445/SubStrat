# library imports
import scipy
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA


class DatasetPropertiesMeasurements:
    """
    A collection of analysis for a tabular dataset
    A lot of ideas from: Reif, M., Shafait, F., Dengel, A., Meta-learning for evolutionary parameter optimization of classifiers. Machine Learning. 2012, 87:357-380. ---> (https://link.springer.com/content/pdf/10.1007/s10994-012-5286-7.pdf)

    Statistical features from: (Gama and Brazdil 1995; Vilalta et al. 2004; Brazdil et al. 1994; King et al. 1995)

    More ideas from: Shen, Z., Chen, X., Garibaldi, J. M., A Novel Meta Learning Framework for Feature Selection using Data Synthesis and Fuzzy Similarity. IEEE World Congress on Computational Intelligence. 2020. ----> (https://arxiv.org/pdf/2005.09856.pdf)
    """

    # consts #
    TARGET_COL_NAME = "target"
    IS_DEBUG = False
    # end - consts #

    def __init__(self):
        pass

    @staticmethod
    def get_columns():
        """
        Run all the measurements and summary them up in a dict
        """
        return ["row_count",
                "col_count",
                "row_over_class",
                "col_over_class",
                "row_over_col",
                "col_count",
                "col_count",
                "col_numerical_count",
                "col_categorical_count",
                "classes_count",
                "cancor",
                "kurtosis",
                "average_asymmetry_of_features",
                "average_linearly_to_target",
                "std_linearly_to_target",
                "average_correlation_between_features",
                "average_coefficient_of_variation_of_feature",
                "std_coefficient_of_variation_of_feature",
                "average_coefficient_of_anomaly",
                "std_coefficient_of_anomaly",
                "average_entropy_of_features",
                "std_entropy_of_features"
        ]

    @staticmethod
    def get_dataset_profile(dataset: pd.DataFrame):
        """
        Run all the measurements and summary them up in a dict
        """
        return {
            "row_count": DatasetPropertiesMeasurements.row_count(dataset=dataset),
            "col_count": DatasetPropertiesMeasurements.col_count(dataset=dataset),
            "row_over_class": DatasetPropertiesMeasurements.row_over_class(dataset=dataset),
            "col_over_class": DatasetPropertiesMeasurements.col_over_class(dataset=dataset),
            "row_over_col": DatasetPropertiesMeasurements.row_over_col(dataset=dataset),
            "col_numerical_count": DatasetPropertiesMeasurements.col_numerical_count(dataset=dataset),
            "col_categorical_count": DatasetPropertiesMeasurements.col_categorical_count(dataset=dataset),
            "classes_count": DatasetPropertiesMeasurements.classes_count(dataset=dataset),
            "cancor": DatasetPropertiesMeasurements.cancor(dataset=dataset),
            "kurtosis": DatasetPropertiesMeasurements.kurtosis(dataset=dataset),
            "average_asymmetry_of_features": DatasetPropertiesMeasurements.average_asymmetry_of_features(dataset=dataset),
            "average_linearly_to_target": DatasetPropertiesMeasurements.average_linearly_to_target(dataset=dataset),
            "std_linearly_to_target": DatasetPropertiesMeasurements.std_linearly_to_target(dataset=dataset),
            "average_correlation_between_features": DatasetPropertiesMeasurements.average_correlation_between_features(dataset=dataset),
            "average_coefficient_of_variation_of_feature": DatasetPropertiesMeasurements.average_coefficient_of_variation_of_feature(dataset=dataset),
            "std_coefficient_of_variation_of_feature": DatasetPropertiesMeasurements.std_coefficient_of_variation_of_feature(dataset=dataset),
            "average_coefficient_of_anomaly": DatasetPropertiesMeasurements.average_coefficient_of_anomaly(dataset=dataset),
            "std_coefficient_of_anomaly": DatasetPropertiesMeasurements.std_coefficient_of_anomaly(dataset=dataset),
            "average_entropy_of_features": DatasetPropertiesMeasurements.average_entropy_of_features(dataset=dataset),
            "std_entropy_of_features": DatasetPropertiesMeasurements.std_entropy_of_features(dataset=dataset)
        }

    @staticmethod
    def get_dataset_profile_vector(dataset: pd.DataFrame):
        """
        Run all the measurements and summary them up in a dict
        """
        return [DatasetPropertiesMeasurements.row_count(dataset=dataset),
                DatasetPropertiesMeasurements.col_count(dataset=dataset),
                DatasetPropertiesMeasurements.row_over_class(dataset=dataset),
                DatasetPropertiesMeasurements.col_over_class(dataset=dataset),
                DatasetPropertiesMeasurements.row_over_col(dataset=dataset),
                DatasetPropertiesMeasurements.col_numerical_count(dataset=dataset),
                DatasetPropertiesMeasurements.col_categorical_count(dataset=dataset),
                DatasetPropertiesMeasurements.classes_count(dataset=dataset),
                DatasetPropertiesMeasurements.cancor(dataset=dataset),
                DatasetPropertiesMeasurements.kurtosis(dataset=dataset),
                DatasetPropertiesMeasurements.average_asymmetry_of_features(dataset=dataset),
                DatasetPropertiesMeasurements.average_linearly_to_target(dataset=dataset),
                DatasetPropertiesMeasurements.std_linearly_to_target(dataset=dataset),
                DatasetPropertiesMeasurements.average_correlation_between_features(dataset=dataset),
                DatasetPropertiesMeasurements.average_coefficient_of_variation_of_feature(dataset=dataset),
                DatasetPropertiesMeasurements.std_coefficient_of_variation_of_feature(dataset=dataset),
                DatasetPropertiesMeasurements.average_coefficient_of_anomaly(dataset=dataset),
                DatasetPropertiesMeasurements.std_coefficient_of_anomaly(dataset=dataset),
                DatasetPropertiesMeasurements.average_entropy_of_features(dataset=dataset),
                DatasetPropertiesMeasurements.std_entropy_of_features(dataset=dataset)]

    @staticmethod
    def row_count(dataset: pd.DataFrame):
        # Idea from (Engels and Theusinger 1998)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.row_count running")
        return dataset.shape[0]

    @staticmethod
    def col_count(dataset: pd.DataFrame):
        # Idea from (Engels and Theusinger 1998)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.col_count running")
        return dataset.shape[1]

    @staticmethod
    def row_over_class(dataset: pd.DataFrame):
        # Idea from (Engels and Theusinger 1998)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.row_over_class running")
        try:
            return dataset.shape[0] / dataset[DatasetPropertiesMeasurements.TARGET_COL_NAME].nunique()
        except:
            return np.nan

    @staticmethod
    def col_over_class(dataset: pd.DataFrame):
        # Idea from (Engels and Theusinger 1998)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.col_over_class running")
        try:
            return dataset.shape[1] / dataset[DatasetPropertiesMeasurements.TARGET_COL_NAME].nunique()
        except:
            return np.nan

    @staticmethod
    def row_over_col(dataset: pd.DataFrame):
        # Idea from (Engels and Theusinger 1998)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.col_over_class running")
        try:
            return dataset.shape[0] / dataset.shape[1]
        except:
            return np.nan

    @staticmethod
    def col_numerical_count(dataset: pd.DataFrame,
                            max_values: int = 20):
        """
        :param dataset: the data set
        :param max_values: max number of values to be considered categorical
        :return: the number of categorical colums
        """
        # Idea from (Engels and Theusinger 1998)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.col_numerical_count running")
        try:
            return len([1 for column in list(dataset) if dataset[column].nunique() > max_values])
        except:
            return np.nan

    @staticmethod
    def col_categorical_count(dataset: pd.DataFrame,
                              max_values: int = 20):
        """
        :param dataset: the data set
        :param max_values: max number of values to be considered categorical
        :return: the number of categorical colums
        """
        # Idea from (Engels and Theusinger 1998)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.col_categorical_count running")
        try:
            return len(list(dataset)) - DatasetPropertiesMeasurements.col_numerical_count(dataset=dataset,
                                                                                          max_values=max_values)
        except:
            return np.nan

    @staticmethod
    def classes_count(dataset: pd.DataFrame):
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.classes_count running")
        try:
            return dataset[DatasetPropertiesMeasurements.TARGET_COL_NAME].nunique()
        except:
            return np.nan

    @staticmethod
    def average_linearly_to_target(dataset: pd.DataFrame):
        """
        :return: the average R^2 between the explainable features and the target feature
        """
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.average_linearly_to_target running")
        try:
            return np.nanmean([np.corrcoef(dataset[col], dataset[DatasetPropertiesMeasurements.TARGET_COL_NAME])[0, 1]**2
                            for col in list(dataset) if col != DatasetPropertiesMeasurements.TARGET_COL_NAME])
        except:
            return np.nan

    @staticmethod
    def std_linearly_to_target(dataset: pd.DataFrame):
        """
        :return: the std R^2 between the explainable features and the target feature
        """
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.std_linearly_to_target running")
        try:
            return np.nanstd([np.corrcoef(dataset[col], dataset[DatasetPropertiesMeasurements.TARGET_COL_NAME])[0, 1] ** 2
                           for col in list(dataset) if col != DatasetPropertiesMeasurements.TARGET_COL_NAME])
        except:
            return np.nan

    @staticmethod
    def cancor(dataset: pd.DataFrame):
        """
        :return: canonical correlation for the best single combination of features
        """
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.cancor running")
        try:
            cca = CCA(n_components=1)
            x = dataset.drop(DatasetPropertiesMeasurements.TARGET_COL_NAME, inplace=False, axis=1)
            Uc, Vc = cca.fit_transform(x, dataset[DatasetPropertiesMeasurements.TARGET_COL_NAME])
            return np.corrcoef(Uc.T, Vc.T)[0, 1]
        except:
            return np.nan

    @staticmethod
    def kurtosis(dataset: pd.DataFrame):
        """
        :return: mean peakedness of the probability distributions of the features
        """
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.kurtosis running")
        try:
            x = dataset.drop(DatasetPropertiesMeasurements.TARGET_COL_NAME, inplace=False, axis=1)
            return np.nanmean([scipy.stats.kurtosis(x[column]) for column in list(x)])
        except:
            return np.nan

    @staticmethod
    def average_asymmetry_of_features(dataset: pd.DataFrame):
        """
        :return: It measures the average value of the Pearson’s asymmetry coefficient.
        """
        # idea from (Shen et al., 2020)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.average_asymmetry_of_features running")
        try:
            return 3 * np.nansum([(np.nanmean(dataset[column]) - np.median(dataset[column]))/np.nanstd(dataset[column]) for column in list(dataset) if np.nanstd(dataset[column]) > 0]) / len(list(dataset))
        except:
            return np.nan

    @staticmethod
    def average_correlation_between_features(dataset: pd.DataFrame):
        """
        :return: It measures the average value of Pearson’s correlation coefficient between different features.
        """
        # idea from (Shen et al., 2020)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.average_correlation_between_features running")
        try:
            n = len(list(dataset))
            cols = list(dataset)
            return 2 * sum([np.nansum([scipy.stats.pearsonr(dataset[cols[column_i]],
                                                      dataset[cols[column_j]])[0]
                                 for column_j in range(column_i + 1, len(cols))])
                            for column_i in range(len(cols)-1)]) / (n * (n-1))
        except:
            return np.nan

    @staticmethod
    def average_coefficient_of_variation_of_feature(dataset: pd.DataFrame):
        """
        :return: It measures the STD coefficient of variation by the ratio of the standard deviation and the mean of the feature values.
        """
        # idea from (Shen et al., 2020)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.average_coefficient_of_variation_of_feature running")
        try:
            return np.nanmean([np.nanstd(dataset[column]) / np.nanmean(dataset[column]) for column in list(dataset)])
        except:
            return np.nan

    @staticmethod
    def std_coefficient_of_variation_of_feature(dataset: pd.DataFrame):
        """
        :return: It measures the STD coefficient of variation by the ratio of the standard deviation and the mean of the feature values.
        """
        # idea from (Shen et al., 2020)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.std_coefficient_of_variation_of_feature running")
        try:
            return np.nanstd([np.nanstd(dataset[column]) / np.nanmean(dataset[column]) for column in list(dataset)])
        except:
            return np.nan

    @staticmethod
    def average_coefficient_of_anomaly(dataset: pd.DataFrame):
        """
        :return: It measures the average coefficient of anomaly by the ratio of the mean and the standard deviation of the feature values.
        """
        # idea from (Shen et al., 2020)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.average_coefficient_of_anomaly running")
        try:
            return np.nanmean([np.nanmean(dataset[column]) / np.nanstd(dataset[column]) for column in list(dataset)])
        except:
            return np.nan

    @staticmethod
    def std_coefficient_of_anomaly(dataset: pd.DataFrame):
        """
        :return: It measures the STD coefficient of anomaly by the ratio of the mean and the standard deviation of the feature values.
        """
        # idea from (Shen et al., 2020)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.std_coefficient_of_anomaly running")
        try:
            return np.nanstd([np.nanmean(dataset[column]) / np.nanstd(dataset[column]) for column in list(dataset)])
        except:
            return np.nan

    @staticmethod
    def average_entropy_of_features(dataset: pd.DataFrame):
        """
        :return: It measures the average coefficient of anomaly by the ratio of the mean and the standard deviation of the feature values.
        """
        # idea from (Shen et al., 2020)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.average_entropy_of_features running")
        try:
            return np.nanmean([scipy.stats.entropy(dataset[column]) for column in list(dataset)
                              if scipy.stats.entropy(dataset[column]) != -np.inf])
        except:
            return np.nan

    @staticmethod
    def std_entropy_of_features(dataset: pd.DataFrame):
        """
        :return: It measures the STD coefficient of anomaly by the ratio of the mean and the standard deviation of the feature values.
        """
        # idea from (Shen et al., 2020)
        if DatasetPropertiesMeasurements.IS_DEBUG:
            print("DatasetPropertiesMeasurements.std_entropy_of_features running")
        try:
            return np.nanstd([scipy.stats.entropy(dataset[column]) for column in list(dataset)
                           if scipy.stats.entropy(dataset[column]) != -np.inf])
        except:
            return np.nan

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<DatasetPropertiesMeasurements>"
