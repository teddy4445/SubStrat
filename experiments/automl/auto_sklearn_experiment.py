# library imports
import os
import math
import json
import random
import numpy as np
import pandas as pd
from glob import glob
from time import time
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import autosklearn.classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas.core.dtypes.common import is_numeric_dtype

# project imports
from ds.table import Table
from experiments.stability_experiment import prepare_dataset
from methods.summary_wellness_scores import SummaryWellnessScores
from ds.dataset_properties_measurements import DatasetPropertiesMeasurements
from summary_algorithms.stabler_genetic_algorithm_summary_algorithm import StablerGeneticSummary


# so the colors will be the same
random.seed(73)


def prepare_dataset_full(df):
    # remove what we do not need
    df.drop([col for col in list(df) if not is_numeric_dtype(df[col])], axis=1, inplace=True)
    # remove _rows with nan
    df.dropna(inplace=True)
    # get only max number of _rows to work with
    return df


class AutoSKlearnExperiment:
    """
    An experiments to use the stable sub-table in the usage of auto ML experiment

    Cite:
    @inproceedings{feurer-neurips15a,
        title     = {Efficient and Robust Automated Machine Learning},
        author    = {Feurer, Matthias and Klein, Aaron and Eggensperger, Katharina  Springenberg, Jost and Blum, Manuel and Hutter, Frank},
        booktitle = {Advances in Neural Information Processing Systems 28 (2015)},
        pages     = {2962--2970},
        year      = {2015}
    }

    """

    # CONSTS #
    DATASETS = {os.path.basename(path).replace(".csv", ""): prepare_dataset_full(pd.read_csv(path))
                for path in glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*.csv"))}

    METRICS = {
        "entropy": SummaryWellnessScores.mean_entropy,
        "coefficient_of_anomaly": SummaryWellnessScores.coefficient_of_anomaly,
        "coefficient_of_variation": SummaryWellnessScores.coefficient_of_variation,
    }

    MARKERS = {
        "entropy": "o",
        "coefficient_of_anomaly": "X",
        "coefficient_of_variation": "^",
    }

    COLORS = {os.path.basename(path).replace(".csv", ""): "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                for path in glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*.csv"))}

    # STABILITY TEST FACTORS
    NOISE_FUNC = SummaryWellnessScores.add_dataset_subset_pick_noise
    NOISE_FACTOR = 0.05
    REPEAT_NOISE = 3
    REPEAT_START_CONDITION = 3
    REPEAT_SUMMARY = 5

    # ALGORITHM HYPER-PARAMETERS
    MAX_ITER = 25
    SUMMARY_ROW_SIZE = 20
    SUMMARY_COL_SIZE = 5

    COL_PORTION = 0.75

    # IO VALUES
    RESULT_FOLDER_NAME = "auto_ml_results"
    RESULT_PATH = os.path.join(os.path.dirname(__file__), RESULT_FOLDER_NAME)

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(target_feature_name: str = "target",
            test_portion: float = 0.1):
        # make sure we have the folder for the answers
        try:
            os.mkdir(AutoSKlearnExperiment.RESULT_PATH)
        except Exception as error:
            pass

        answer = {}
        for dataset_name, dataset in AutoSKlearnExperiment.DATASETS.items():
            answer[dataset_name] = {}
            for metric_name, metric in AutoSKlearnExperiment.METRICS.items():
                x, y = dataset.drop([target_feature_name], axis=1), dataset[target_feature_name]
                best_rows, best_columns, converge_report = StablerGeneticSummary.run(dataset=dataset,
                                                                                     desired_row_size=round(
                                                                                         math.sqrt(dataset.shape[0])),
                                                                                     desired_col_size=round(
                                                                                         AutoSKlearnExperiment.COL_PORTION *
                                                                                         dataset.shape[1]),
                                                                                     evaluate_score_function=metric,
                                                                                     is_return_indexes=True,
                                                                                     max_iter=AutoSKlearnExperiment.MAX_ITER)
                # get the summary
                if len(dataset.shape[1]) - 1 not in best_columns:
                    best_columns.append(len(dataset.shape[1]) - 1)
                summary = dataset.iloc[best_rows, best_columns]

                # split all
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_portion, random_state=73)

                # full data learning
                full_data_start = time()
                cls = autosklearn.classification.AutoSklearnClassifier()
                cls.fit(x_train, y_train)
                predictions = cls.predict(x_test)
                full_data_acc = accuracy_score(predictions, y_test)
                full_data_end = time()

                # sub-table learning
                x_summary, y_summary = summary.drop([target_feature_name], axis=1), summary[target_feature_name]
                x_train_summary, x_test_summary, y_train_summary, y_test_summary = train_test_split(x_summary, y_summary, test_size=test_portion, random_state=73)

                summary_start = time()
                cls = autosklearn.classification.AutoSklearnClassifier()
                cls.fit(x_train_summary, y_train_summary)
                predictions = cls.predict(x_test)
                summary_acc = accuracy_score(predictions, y_test)
                summary_end = time()

                # compute answer and save it
                answer[dataset_name][metric_name] = (full_data_end - full_data_start, summary_end - summary_start, full_data_acc, summary_acc)

                # safe the results at each point we need
                with open(os.path.join(AutoSKlearnExperiment.RESULT_PATH, "raw_data.json"), "w") as answer_file_json:
                    json.dump(answer, answer_file_json)

        # organize the results differently
        nice_answer = []
        for dataset_name in answer:
            for metric_name in answer[dataset_name]:
                relative_change_in_time = round(abs(answer[dataset_name][metric_name][0] - answer[dataset_name][metric_name][1]) / answer[dataset_name][metric_name][0], 2)
                relative_change_in_performance = round(abs(answer[dataset_name][metric_name][2] - answer[dataset_name][metric_name][3]) / answer[dataset_name][metric_name][2], 2)
                nice_answer.append([dataset_name, metric_name, relative_change_in_time, relative_change_in_performance])

        # print the result
        plt.scatter([val[2] for val in nice_answer],
                    [val[3] for val in nice_answer],
                    color=[AutoSKlearnExperiment.COLORS[val[0]] for val in nice_answer],
                    marker=[AutoSKlearnExperiment.MARKERS[val[1]] for val in nice_answer])
        plt.xlabel("Relative change in computation time [t]")
        plt.xlabel("Relative change in performance [1]")
        plt.savefig(os.path.join(AutoSKlearnExperiment.RESULT_PATH, "results.png"))
        plt.close()
