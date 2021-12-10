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


# metric fitness property
def fitness(df: pd.DataFrame,
            acc_weight: float):
    # "full_time_sec,subtable_time_sec,full_accuracy,subtable_accuracy"
    fit_df = df.copy()
    fit_df["relative_time"] = (df["full_time_sec"] - df["subtable_time_sec"])/df["full_time_sec"]
    fit_df["relative_acc"] = (df["full_accuracy"] - df["subtable_accuracy"])/df["full_accuracy"]
    return fit_df["relative_time"].mean() * (1 - acc_weight) + fit_df["relative_acc"].mean() * acc_weight

# so the colors will be the same
random.seed(73)


class AutoSKlearnExperimentVisualize:
    """
    Continue the experiment at '/experiments/automl/auto_sklearn_experiment.py' to pick the best metric to work with
    """

    METRICS = ["entropy", "coefficient_of_anomaly","coefficient_of_variation"]

    MARKERS = {
        "entropy": "o",
        "coefficient_of_anomaly": "X",
        "coefficient_of_variation": "^",
    }

    COLORS = {"dataset_{}".format(index): "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
              for index in range(16, 26)}

    FITNESS_ACC_WEIGHT = 0.8

    # IO VALUES
    RESULT_FOLDER_NAME = "auto_ml_results"
    RESULT_PATH = os.path.join(os.path.dirname(__file__), RESULT_FOLDER_NAME)

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(file_name: str = "full_pipeline_raw_data.csv"):
        # get the data
        df = pd.read_csv(os.path.join(AutoSKlearnExperimentVisualize.RESULT_PATH, file_name))
        # answer the data
        answer = {key: 0 for key in AutoSKlearnExperimentVisualize.METRICS}

        for metric in answer:
            metric_df = df[df["metric"] == metric]
            answer[metric] = fitness(metric_df,
                                     acc_weight=AutoSKlearnExperimentVisualize.FITNESS_ACC_WEIGHT)

        plt.bar(list(range(3)), answer.values())
        plt.xlabel("Metric")
        plt.ylabel("Weighted average of mean time relative error (w={:.2f})\n and mean accuracy (w={:.2f})".format(1-AutoSKlearnExperimentVisualize.FITNESS_ACC_WEIGHT,
                                                                                                                 AutoSKlearnExperimentVisualize.FITNESS_ACC_WEIGHT))
        plt.xticks(list(range(len(answer))), list(answer.keys()), rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(AutoSKlearnExperimentVisualize.RESULT_PATH, "fitness of each metric.png"))
        plt.close()

        # print the result
        for metric in AutoSKlearnExperimentVisualize.METRICS:
            metric_df = df[df["metric"] == metric]
            plt.scatter((metric_df["full_time_sec"] - metric_df["subtable_time_sec"])/metric_df["full_time_sec"],
                        (metric_df["full_accuracy"] - metric_df["subtable_accuracy"])/metric_df["full_accuracy"])
            plt.xlabel("Relative change in computation time [t]")
            plt.ylabel("Relative change in performance [1]")
            plt.xlim((0, 1))
            plt.ylim((-0.2, 1))
            plt.grid(alpha=0.2, color="black")
            plt.title("Graph for metric '{}'".format(metric))
            plt.savefig(os.path.join(AutoSKlearnExperimentVisualize.RESULT_PATH, "results_metric_{}.png".format(metric)))
            plt.close()

    @staticmethod
    def pipeline(file_name: str = "full_pipeline_raw_data.csv",
                 full_metric_name: str = "full_accuracy",
                 subtable_metric_name: str = "subtable_accuracy"):
        # get the data
        df = pd.read_csv(os.path.join(AutoSKlearnExperimentVisualize.RESULT_PATH, file_name))

        # print the result
        plt.scatter((df["full_time_min"] - df["subtable_time_min"])/df["full_time_min"],
                    (df[full_metric_name] - df[subtable_metric_name])/df[full_metric_name],
                    color=[AutoSKlearnExperimentVisualize.COLORS[name] for name in df["dataset"]])
        plt.xlabel("Relative change in computation time in minutes [t]")
        plt.ylabel("Relative change in performance [1]")
        plt.xlim((-5, 1))
        plt.ylim((-0.2, 1))
        plt.grid(alpha=0.2, color="black")
        plt.savefig(os.path.join(AutoSKlearnExperimentVisualize.RESULT_PATH, "results_pipeline_scatter.png"))
        plt.close()

        # zoom in
        plt.scatter((df["full_time_min"] - df["subtable_time_min"])/df["full_time_min"],
                    (df[full_metric_name] - df[subtable_metric_name])/df[full_metric_name],
                    color=[AutoSKlearnExperimentVisualize.COLORS[name] for name in df["dataset"]])
        plt.xlabel("Relative change in computation time in minutes [t]")
        plt.ylabel("Relative change in performance [1]")
        plt.xlim((0, 0.4))
        plt.ylim((-0.1, 0.1))
        plt.grid(alpha=0.2, color="black")
        plt.savefig(os.path.join(AutoSKlearnExperimentVisualize.RESULT_PATH, "results_pipeline_scatter_zoom_in.png"))
        plt.close()


if __name__ == '__main__':
    AutoSKlearnExperimentVisualize.pipeline()
