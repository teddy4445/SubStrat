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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
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
                for path in
                glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "*.csv"))}

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

    COLORS = {
        os.path.basename(path).replace(".csv", ""): "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        for path in glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "*.csv"))}

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
    NAIVE_TIMES = 10

    COL_PORTION = 0.75

    # IO VALUES
    RESULT_FOLDER_NAME = "auto_ml_results"
    RESULT_PATH = os.path.join(os.path.dirname(__file__), RESULT_FOLDER_NAME)

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run_grid(target_feature_name: str = "target",
                 test_portion: float = 0.1):
        for row_portion in range(1, 11):
            for col_portion in range(1, 11):
                AutoSKlearnExperiment.run(target_feature_name=target_feature_name,
                                          test_portion=test_portion,
                                          row_portion=row_portion/10,
                                          col_portion=col_portion/10)

    @staticmethod
    def run(target_feature_name: str = "target",
            test_portion: float = 0.1,
            row_portion: float = 0,
            col_portion: float = 0):
        # make sure we have the folder for the answers
        try:
            os.mkdir(AutoSKlearnExperiment.RESULT_PATH)
        except Exception as error:
            pass

        answer = {}
        for dataset_name, dataset in AutoSKlearnExperiment.DATASETS.items():
            answer[dataset_name] = {}
            for metric_name, metric in AutoSKlearnExperiment.METRICS.items():
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
                    full_data_f1 = f1_score(predictions, y_test)
                    full_data_auc = roc_auc_score(predictions, y_test)
                    full_data_end = time()

                    naive_time_list = []
                    naive_acc_list = []
                    naive_f1_list = []
                    naive_auc_list = []

                    for _ in range(AutoSKlearnExperiment.NAIVE_TIMES):
                        # naive baseline #
                        naive_rows = random.sample(list(range(x_train.shape[0])), round(math.sqrt(dataset.shape[0])) if row_portion == 0 else round(dataset.shape[0] * row_portion))
                        naive_columns = random.sample(list(range(x_train.shape[1])),
                                                      round(AutoSKlearnExperiment.COL_PORTION * dataset.shape[1]) if col_portion == 0 else round(dataset.shape[1] * col_portion))
                        naive_columns_full = naive_columns.copy()
                        # get the summary
                        if dataset.shape[1] - 1 not in naive_columns_full:
                            naive_columns_full.append(dataset.shape[1] - 1)
                        summary = dataset.iloc[naive_rows, naive_columns_full]

                        # sub-table learning
                        x_summary, y_summary = summary.drop([target_feature_name], axis=1), summary[target_feature_name]
                        x_train_summary, x_test_summary, y_train_summary, y_test_summary = train_test_split(x_summary,
                                                                                                            y_summary,
                                                                                                            test_size=test_portion,
                                                                                                            random_state=73)

                        naive_start = time()
                        try:
                            cls = autosklearn.classification.AutoSklearnClassifier()
                        except Exception as error:
                            cls = DecisionTreeClassifier()
                        cls.fit(x_train_summary, y_train_summary)
                        try:
                            predictions = cls.predict(x_test.iloc[:, naive_columns])
                            naive_acc = accuracy_score(predictions, y_test)
                            naive_f1 = f1_score(predictions, y_test)
                            naive_auc = roc_auc_score(predictions, y_test)
                        except Exception as error:
                            predictions = cls.predict(x_test_summary)
                            naive_acc = accuracy_score(predictions, y_test_summary)
                            naive_f1 = f1_score(predictions, y_test_summary)
                            naive_auc = roc_auc_score(predictions, y_test_summary)
                        naive_end = time()
                        naive_time_list.append((naive_end - naive_start))
                        naive_acc_list.append(naive_acc)
                        naive_f1_list.append(naive_f1)
                        naive_auc_list.append(naive_auc)

                    # get sub-table (summary)
                    best_rows, best_columns, converge_report = StablerGeneticSummary.run(dataset=x_train,
                                                                                         desired_row_size=round(math.sqrt(dataset.shape[0])) if row_portion == 0 else round(dataset.shape[0] * row_portion),
                                                                                         desired_col_size=round(AutoSKlearnExperiment.COL_PORTION * dataset.shape[1]) if col_portion == 0 else round(dataset.shape[1] * col_portion),
                                                                                         evaluate_score_function=metric,
                                                                                         is_return_indexes=True,
                                                                                         max_iter=AutoSKlearnExperiment.MAX_ITER)
                    # get the summary
                    best_columns_x = best_columns.copy()
                    if dataset.shape[1] - 1 not in best_columns:
                        best_columns.append(dataset.shape[1] - 1)
                    summary = dataset.iloc[best_rows, best_columns]

                    # sub-table learning
                    x_summary, y_summary = summary.drop([target_feature_name], axis=1), summary[target_feature_name]
                    x_train_summary, x_test_summary, y_train_summary, y_test_summary = train_test_split(x_summary,
                                                                                                        y_summary,
                                                                                                        test_size=test_portion,
                                                                                                        random_state=73)

                    summary_start = time()
                    try:
                        cls = autosklearn.classification.AutoSklearnClassifier()
                    except Exception as error:
                        cls = DecisionTreeClassifier()
                    cls.fit(x_train_summary, y_train_summary)
                    try:
                        predictions = cls.predict(x_test.iloc[:, best_columns_x])
                        summary_acc = accuracy_score(predictions, y_test)
                        summary_f1 = f1_score(predictions, y_test)
                        summary_auc = roc_auc_score(predictions, y_test)
                    except:
                        predictions = cls.predict(x_test_summary)
                        summary_acc = accuracy_score(predictions, y_test_summary)
                        summary_f1 = f1_score(predictions, y_test_summary)
                        summary_auc = roc_auc_score(predictions, y_test_summary)
                    summary_end = time()

                    # compute answer and save it
                    answer[dataset_name][metric_name] = ((full_data_end - full_data_start),
                                                         (summary_end - summary_start),
                                                         np.mean(naive_time_list),
                                                         full_data_acc,
                                                         summary_acc,
                                                         np.mean(naive_acc_list),
                                                         full_data_f1,
                                                         summary_f1,
                                                         np.mean(naive_f1_list),
                                                         full_data_auc,
                                                         summary_auc,
                                                         np.mean(naive_auc_list))

                    # safe the results at each point we need
                    with open(os.path.join(AutoSKlearnExperiment.RESULT_PATH, "raw_data_row_{}_col_{}.json".format(row_portion,
                                                                                                                   col_portion)),
                              "w") as answer_file_json:
                        json.dump(answer,
                                  answer_file_json,
                                  indent=2)
                except Exception as error:
                    print("Skipping this dataset {}".format(error))

        # safe the results at each point we need
        with open(os.path.join(AutoSKlearnExperiment.RESULT_PATH, "raw_data_row_{}_col_{}.csv".format(row_portion,
                                                                                                      col_portion)), "w") as answer_file_csv:
            csv_answer = "dataset,metric,full_time_sec,subtable_time_sec,baseline_time_sec" \
                         ",full_accuracy,subtable_accuracy,baseline_accuracy" \
                         ",full_f1,subtable_f1,baseline_f1" \
                         ",full_auc,subtable_auc,baseline_auc\n"
            for dataset in answer.keys():
                for metric in answer[dataset].keys():
                    csv_answer += "{},{},{:.4f},{:.4f},{:.4f}" \
                                  ",{:.4f},{:.4f},{:.4f}" \
                                  ",{:.4f},{:.4f},{:.4f}" \
                                  ",{:.4f},{:.4f},{:.4f}\n".format(dataset,
                                                                   metric,
                                                                   answer[dataset][metric][0],
                                                                   answer[dataset][metric][1],
                                                                   answer[dataset][metric][2],
                                                                   answer[dataset][metric][3],
                                                                   answer[dataset][metric][4],
                                                                   answer[dataset][metric][5],
                                                                   answer[dataset][metric][6],
                                                                   answer[dataset][metric][7],
                                                                   answer[dataset][metric][8],
                                                                   answer[dataset][metric][9],
                                                                   answer[dataset][metric][10],
                                                                   answer[dataset][metric][11])
            answer_file_csv.write(csv_answer)


if __name__ == '__main__':
    AutoSKlearnExperiment.run()
