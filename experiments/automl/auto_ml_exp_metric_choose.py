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
try:
    import autosklearn.classification
except:
    pass
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from pandas.core.dtypes.common import is_numeric_dtype

# project imports
from summary_algorithms.stabler_genetic_algorithm_summary_algorithm import StablerGeneticSummary


# so the colors will be the same
random.seed(73)


class AutoSKlearnExperimentMetricChoose:
    """
    Continue the experiment at '/experiments/automl/auto_sklearn_experiment.py' to pick the best metric to work with
    """

    METRICS = ["entropy", "coefficient_of_anomaly","coefficient_of_variation"]

    MARKERS = {
        "entropy": "o",
        "coefficient_of_anomaly": "X",
        "coefficient_of_variation": "^",
    }

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
            os.mkdir(AutoSKlearnExperimentMetricChoose.RESULT_PATH)
        except Exception as error:
            pass

        answer = {}
        for dataset_name, dataset in AutoSKlearnExperimentMetricChoose.DATASETS.items():
            answer[dataset_name] = {}
            for metric_name, metric in AutoSKlearnExperimentMetricChoose.METRICS.items():
                try:
                    print("Start: {} with {}".format(dataset_name, metric_name))
                    # split data
                    x, y = dataset.drop([target_feature_name], axis=1), dataset[target_feature_name]
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_portion, random_state=73)

                    # full data learning
                    full_data_start = time()
                    try:
                        cls = autosklearn.classification.AutoSklearnClassifier()
                    except Exception as error:
                        cls = DecisionTreeClassifier()
                    cls.fit(x_train, y_train)
                    predictions = cls.predict(x_test)
                    full_data_acc = accuracy_score(predictions, y_test)
                    full_data_end = time()

                    # naive baseline #
                    naive_rows = random.sample(list(range(x_train.shape[0])), round(math.sqrt(dataset.shape[0])))
                    naive_columns = random.sample(list(range(x_train.shape[1])), round(AutoSKlearnExperimentMetricChoose.COL_PORTION * dataset.shape[1]))
                    naive_columns_full = naive_columns.copy()
                    # get the summary
                    if dataset.shape[1] - 1 not in naive_columns_full:
                        naive_columns_full.append(dataset.shape[1] - 1)
                    summary = dataset.iloc[naive_rows, naive_columns_full]

                    # sub-table learning
                    x_summary, y_summary = summary.drop([target_feature_name], axis=1), summary[target_feature_name]
                    x_train_summary, x_test_summary, y_train_summary, y_test_summary = train_test_split(x_summary, y_summary, test_size=test_portion, random_state=73)

                    naive_start = time()
                    try:
                        cls = autosklearn.classification.AutoSklearnClassifier()
                    except Exception as error:
                        cls = DecisionTreeClassifier()
                    cls.fit(x_train_summary, y_train_summary)
                    try:
                        predictions = cls.predict(x_test.iloc[:, naive_columns])
                        naive_acc = accuracy_score(predictions, y_test)
                    except Exception as error:
                        predictions = cls.predict(x_test_summary)
                        naive_acc = accuracy_score(predictions, y_test_summary)
                    naive_end = time()

                    # get sub-table (summary)
                    best_rows, best_columns, converge_report = StablerGeneticSummary.run(dataset=x_train,
                                                                                         desired_row_size=round(
                                                                                             math.sqrt(dataset.shape[0])),
                                                                                         desired_col_size=round(
                                                                                             AutoSKlearnExperimentMetricChoose.COL_PORTION *
                                                                                             dataset.shape[1]),
                                                                                         evaluate_score_function=metric,
                                                                                         is_return_indexes=True,
                                                                                         max_iter=AutoSKlearnExperimentMetricChoose.MAX_ITER)
                    # get the summary
                    best_columns_x = best_columns.copy()
                    if dataset.shape[1] - 1 not in best_columns:
                        best_columns.append(dataset.shape[1] - 1)
                    summary = dataset.iloc[best_rows, best_columns]

                    # sub-table learning
                    x_summary, y_summary = summary.drop([target_feature_name], axis=1), summary[target_feature_name]
                    x_train_summary, x_test_summary, y_train_summary, y_test_summary = train_test_split(x_summary, y_summary, test_size=test_portion, random_state=73)

                    summary_start = time()
                    try:
                        cls = autosklearn.classification.AutoSklearnClassifier()
                    except Exception as error:
                        cls = DecisionTreeClassifier()
                    cls.fit(x_train_summary, y_train_summary)
                    try:
                        predictions = cls.predict(x_test.iloc[:, best_columns_x])
                        summary_acc = accuracy_score(predictions, y_test)
                    except:
                        predictions = cls.predict(x_test_summary)
                        summary_acc = accuracy_score(predictions, y_test_summary)
                    summary_end = time()

                    # compute answer and save it
                    answer[dataset_name][metric_name] = ((full_data_end - full_data_start) * 50,
                                                         (summary_end - summary_start) * 11,
                                                         (naive_end - naive_start) * 11,
                                                         full_data_acc,
                                                         summary_acc,
                                                         naive_acc)

                    # safe the results at each point we need
                    with open(os.path.join(AutoSKlearnExperimentMetricChoose.RESULT_PATH, "raw_data.json"), "w") as answer_file_json:
                        json.dump(answer,
                                  answer_file_json,
                                  indent=2)
                except Exception as error:
                    print("Skipping this dataset {}".format(error))

        # safe the results at each point we need
        with open(os.path.join(AutoSKlearnExperimentMetricChoose.RESULT_PATH, "raw_data.csv"), "w") as answer_file_csv:
            csv_answer = "dataset,metric,full_time_sec,subtable_time_sec,baseline_time_sec,full_accuracy,subtable_accuracy,baseline_accuracy\n"
            for dataset in answer.keys():
                for metric in answer[dataset].keys():
                    csv_answer += "{},{},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(dataset,
                                                                                             metric,
                                                                                             answer[dataset][metric][0],
                                                                                             answer[dataset][metric][1],
                                                                                             answer[dataset][metric][2],
                                                                                             answer[dataset][metric][3],
                                                                                             answer[dataset][metric][4],
                                                                                             answer[dataset][metric][5])
            answer_file_csv.write(csv_answer)

        # organize the results differently
        nice_answer = []
        for dataset_name in answer:
            for metric_name in answer[dataset_name]:
                relative_change_in_time = round((answer[dataset_name][metric_name][0] - answer[dataset_name][metric_name][1]) / answer[dataset_name][metric_name][0], 2)
                relative_change_in_performance = round((answer[dataset_name][metric_name][2] - answer[dataset_name][metric_name][3]) / answer[dataset_name][metric_name][2], 2)
                nice_answer.append([dataset_name, metric_name, relative_change_in_time, relative_change_in_performance])

        # print the result
        for val in nice_answer:
            plt.scatter(val[2],
                        val[3],
                        color=AutoSKlearnExperimentMetricChoose.COLORS[val[0]],
                        marker=AutoSKlearnExperimentMetricChoose.MARKERS[val[1]])
        plt.xlabel("Relative change in computation time [t]")
        plt.ylabel("Relative change in performance [1]")
        plt.xlim((0, 1))
        plt.ylim((-0.2, 1))
        plt.grid(alpha=0.2, color="black")
        plt.savefig(os.path.join(AutoSKlearnExperimentMetricChoose.RESULT_PATH, "results.png"))
        plt.close()


if __name__ == '__main__':
    AutoSKlearnExperimentMetricChoose.run()
