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
try:
    from tpot import TPOTClassifier
    from sklearn.pipeline import make_pipeline
    from tpot.builtins import StackingEstimator
    from tpot.export_utils import set_param_recursive
except:
    pass
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import HalvingRandomSearchCV

# project imports
from ds.table import Table
from methods.summary_wellness_scores import SummaryWellnessScores
from experiments.automl.auto_sklearn_experiment import prepare_dataset_full
from summary_algorithms.genetic_algorithm_summary_algorithm import GeneticSummary

# so the colors will be the same
random.seed(73)


class AutoSKlearnFullPipelineExperiment:
    """
    This class tries to see if we take an approach of doing warm start from the proposed sub-stable obtained from the
    full data set and than just hyper parameter tuning with
    """

    # CONSTS #
    DATASETS = {os.path.basename(path).replace(".csv", ""): prepare_dataset_full(pd.read_csv(path))
                for path in
                glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "*.csv"))}

    COLORS = {
        os.path.basename(path).replace(".csv", ""): "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
        for path in glob(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "*.csv"))}

    # STABILITY TEST FACTORS
    METRIC = SummaryWellnessScores.mean_entropy
    NOISE_FUNC = SummaryWellnessScores.add_dataset_subset_pick_noise
    NOISE_FACTOR = 0.05
    REPEAT_NOISE = 3
    REPEAT_START_CONDITION = 3
    REPEAT_SUMMARY = 5

    # ALGORITHM HYPER-PARAMETERS
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
                 test_portion: float = 0.1,
                 auto_ml_method: str = "tpot"):
        for row_portion in range(1, 11):
            for col_portion in range(1, 11):
                AutoSKlearnFullPipelineExperiment.run(target_feature_name=target_feature_name,
                                                      test_portion=test_portion,
                                                      auto_ml_method=auto_ml_method,
                                                      summary_generations=25,
                                                      row_portion=row_portion/10,
                                                      col_portion=col_portion/10)

    @staticmethod
    def generation_analysis(target_feature_name: str = "target",
                            test_portion: float = 0.1,
                            auto_ml_method: str = "tpot"):
        answer = "generation,avg_relative_time_change,avg_absolute_time_change,avg_relative_acc,avg_absolute_acc\n"
        for i in range(11):
            summary_generations = i * 5
            df = AutoSKlearnFullPipelineExperiment.run(target_feature_name=target_feature_name,
                                                       test_portion=test_portion,
                                                       auto_ml_method=auto_ml_method,
                                                       summary_generations=summary_generations)
            # compute the results
            answer += "{},{},{},{},{}\n".format(summary_generations,
                                          ((df["full_time_min"] - df["subtable_time_min"])/df["full_time_min"]).mean(),
                                          (df["full_time_min"] - df["subtable_time_min"]).mean(),
                                          ((df["full_accuracy"] - df["subtable_accuracy"])/df["full_accuracy"]).mean(),
                                          (df["full_accuracy"] - df["subtable_accuracy"]).mean())


    @staticmethod
    def run(target_feature_name: str = "target",
            auto_ml_method: str = "tpot",
            test_portion: float = 0.1,
            summary_generations: int = 25,
            row_portion: float = 0,
            col_portion: float = 0):
        # make sure we have the folder for the answers
        try:
            os.mkdir(AutoSKlearnFullPipelineExperiment.RESULT_PATH)
        except Exception as error:
            pass

        answer = {}
        for dataset_name, dataset in AutoSKlearnFullPipelineExperiment.DATASETS.items():
            answer[dataset_name] = None
            try:
                print("Start: db = {}".format(dataset_name))
                # split data
                full_data_start = time()
                x, y = dataset.drop([target_feature_name], axis=1), dataset[target_feature_name]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_portion, random_state=73)

                # full data learning
                try:
                    if auto_ml_method == "tpot":
                        cls = TPOTClassifier(generations=100, population_size=100, random_state=73)
                    else:
                        cls = autosklearn.classification.AutoSklearnClassifier()
                except Exception as error:
                    cls = DecisionTreeClassifier()
                cls.fit(x_train, y_train)
                predictions = cls.predict(x_test)
                full_data_acc = accuracy_score(predictions, y_test)
                full_data_f1 = f1_score(predictions, y_test)
                full_data_auc = roc_auc_score(predictions, y_test)
                full_data_end = time()

                # get sub-table (summary)
                summary_start = time()
                best_rows, best_columns, converge_report = GeneticSummary.run(dataset=x_train,
                                                                                     desired_row_size=round(math.sqrt(
                                                                                         dataset.shape[
                                                                                             0])) if row_portion == 0 else round(
                                                                                         dataset.shape[
                                                                                             0] * row_portion),
                                                                                     desired_col_size=round(
                                                                                         AutoSKlearnFullPipelineExperiment.COL_PORTION *
                                                                                         dataset.shape[
                                                                                             1]) if col_portion == 0 else round(
                                                                                         dataset.shape[
                                                                                             1] * col_portion),
                                                                                     evaluate_score_function=AutoSKlearnFullPipelineExperiment.METRIC,
                                                                                     is_return_indexes=True,
                                                                                     max_iter=summary_generations)
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

                try:
                    if auto_ml_method == "tpot":
                        cls = TPOTClassifier(generations=100, population_size=100, random_state=73)
                    else:
                        cls = autosklearn.classification.AutoSklearnClassifier()
                except Exception as error:
                    cls = DecisionTreeClassifier()
                cls.fit(x_train_summary, y_train_summary)
                # try again but this time only with the right classifier
                try:
                    if auto_ml_method == "tpot":
                        cls = make_pipeline(StackingEstimator(estimator=cls.best_model_))
                        set_param_recursive(cls.steps, 'random_state', 73)
                        cls.fit(x_train_summary, y_train_summary)
                    else:
                        cls = autosklearn.classification.AutoSklearnClassifier(
                            include={
                                'classifier': [cls.show_models().split(",")[0]]
                            },
                            exclude=None
                        )
                        cls.fit(x_train_summary, y_train_summary)
                except:
                    pass
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
                answer[dataset_name] = ((full_data_end - full_data_start) / 60,
                                        (summary_end - summary_start) / 60,
                                        full_data_acc,
                                        summary_acc,
                                        full_data_f1,
                                        summary_f1,
                                        full_data_auc,
                                        summary_auc)

                # safe the results at each point we need
                with open(os.path.join(AutoSKlearnFullPipelineExperiment.RESULT_PATH, "raw_data-{}.json".format(auto_ml_method)),
                          "w") as answer_file_json:
                    json.dump(answer,
                              answer_file_json,
                              indent=2)
            except Exception as error:
                print("Skipping this dataset {}".format(error))

        # safe the results at each point we need
        with open(os.path.join(AutoSKlearnFullPipelineExperiment.RESULT_PATH, "full_pipeline_raw_data-{}.csv".format(auto_ml_method)), "w") as answer_file_csv:
            csv_answer = "dataset,full_time_min,subtable_time_min" \
                         ",full_accuracy,subtable_accuracy" \
                         ",full_f1,subtable_f1" \
                         ",full_auc,subtable_auc\n"
            for dataset in answer.keys():
                csv_answer += "{},{:.2f},{:.2f}" \
                              ",{:.2f},{:.2f}" \
                              ",{:.2f},{:.2f}" \
                              ",{:.2f},{:.2f}\n".format(dataset,
                                                        answer[dataset][0],
                                                        answer[dataset][1],
                                                        answer[dataset][2],
                                                        answer[dataset][3],
                                                        answer[dataset][4],
                                                        answer[dataset][5],
                                                        answer[dataset][6],
                                                        answer[dataset][7])
            answer_file_csv.write(csv_answer)

        # read the table for more analysis
        return pd.read_csv(os.path.join(AutoSKlearnFullPipelineExperiment.RESULT_PATH, "full_pipeline_raw_data-{}.csv".format(auto_ml_method)))


if __name__ == '__main__':
    AutoSKlearnFullPipelineExperiment.run()
