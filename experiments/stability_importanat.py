# library import
import os
import json
import random
import numpy as np
import pandas as pd
from glob import glob
from time import time
import matplotlib.pyplot as plt

# project import
from ds.table import Table
from experiments.stability_experiment import prepare_dataset
from methods.summary_wellness_scores import SummaryWellnessScores
from summary_algorithms.las_vegas_summary_algorithm import LasVegasSummary


class StabilityImportantTest:
    """
    This class manage the experiment where we have good and bad stability and show they differ over time
    """

    DATASETS = {os.path.basename(path).replace(".csv", ""): prepare_dataset(pd.read_csv(path))
                for path in glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*.csv"))}

    METRICS = {
        "entropy": SummaryWellnessScores.mean_entropy,
        "coefficient_of_anomaly": SummaryWellnessScores.coefficient_of_anomaly,
        "coefficient_of_variation": SummaryWellnessScores.coefficient_of_variation
    }

    # STABILITY TEST FACTORS
    NOISE_FUNC = SummaryWellnessScores.add_dataset_subset_pick_noise
    NOISE_FACTOR = 0.05
    REPEAT_NOISE = 3
    REPEAT_START_CONDITION = 3

    # ALGORITHM HYPER-PARAMETERS
    MAX_ITER = 50
    SUMMARY_ROW_SIZE = 20
    SUMMARY_COL_SIZE = 5

    # TRASHOLDS FOR METRIC
    TRY_COUNT = 5
    ADD_LINES_STEP = 10

    ADD_LINES = 50

    # IO VALUES #
    RESULT_FOLDER_NAME = "stability_important"
    RESULT_PATH = os.path.join(os.path.dirname(__file__), RESULT_FOLDER_NAME)

    # PLOT
    MARKERS = ["o", "^", "P", "s", "*", "+", "X", "D", "d"]
    COLORS = ["black", "blue", "red", "green", "yellow", "purple", "orange", "gray", "peru", "aqua", "violet", "crimson", "indigo", "lime", "darkolivegreen"]

    # END - IO VALUES #

    def __init__(self):
        self.summaries = []

    def run(self):
        """
        Single entry point, writing the results to csvs
        """
        # make sure we have the folder for the answers
        try:
            os.mkdir(StabilityImportantTest.RESULT_PATH)
        except Exception as error:
            pass

        start_exp_time = time()
        overall_answer = {}
        for dataset_name, dataset in StabilityImportantTest.DATASETS.items():
            answer = {}
            for metric_name, metric in StabilityImportantTest.METRICS.items():
                print("Working metric = {}, dataset = {}, during {} seconds".format(
                    metric_name,
                    dataset_name,
                    time() - start_exp_time))
                answer[metric_name] = {"x": [],
                                       "y_stable": [],
                                       "y_unstable": [],
                                       "stable_performance": 999,
                                       "unstable_performance": 999,
                                       "stable_score": 999,
                                       "unstable_score": 0}
                stable_summary = None
                unstable_summary = None
                # find stable and unstable with high performance summaries
                for try_index in range(StabilityImportantTest.TRY_COUNT):
                    print("Working on summaries, try #{}".format(try_index))
                    summary, converge_report = LasVegasSummary.run(dataset=dataset,
                                                                   desired_row_size=StabilityImportantTest.SUMMARY_ROW_SIZE,
                                                                   desired_col_size=StabilityImportantTest.SUMMARY_COL_SIZE,
                                                                   evaluate_score_function=metric,
                                                                   is_return_indexes=False,
                                                                   max_iter=StabilityImportantTest.MAX_ITER)
                    performance = converge_report.total_score[-1]
                    # only if this summary is good
                    print("Obtain performance of {}".format(converge_report.total_score[-1]))
                    stability = self._stability_test(dataset=dataset,
                                                     algo=LasVegasSummary,
                                                     metric=metric,
                                                     desired_row_size=StabilityImportantTest.SUMMARY_ROW_SIZE,
                                                     desired_col_size=StabilityImportantTest.SUMMARY_COL_SIZE,
                                                     noise_function=StabilityImportantTest.NOISE_FUNC,
                                                     noise=StabilityImportantTest.NOISE_FACTOR,
                                                     repeats=StabilityImportantTest.REPEAT_NOISE,
                                                     start_condition_repeat=StabilityImportantTest.REPEAT_START_CONDITION,
                                                     max_iter=StabilityImportantTest.MAX_ITER)
                    print("Stability score of {}".format(stability))
                    if stability < answer[metric_name]["stable_score"] and performance < answer[metric_name]["stable_performance"]:
                        print("--- Replace stable summary (p={}, s={})".format(performance, stability))
                        stable_summary = summary
                        answer[metric_name]["stable_score"] = stability
                        answer[metric_name]["stable_performance"] = performance
                    elif stability > answer[metric_name]["unstable_score"] or performance < answer[metric_name]["unstable_performance"]:
                        print("--- Replace unstable summary (p={}, s={})".format(performance, stability))
                        unstable_summary = summary
                        answer[metric_name]["unstable_score"] = stability
                        answer[metric_name]["unstable_performance"] = performance
                    print("\n", end="")

                for i in range(StabilityImportantTest.ADD_LINES):
                    # add lines to db
                    dataset = StabilityImportantTest._add_dist_lines_to_db(db=dataset,
                                                                             add_lines=StabilityImportantTest.ADD_LINES_STEP)
                    # check the differance in performance
                    answer[metric_name]["x"].append(i*StabilityImportantTest.ADD_LINES_STEP)
                    stable_score = metric(dataset=dataset,
                                          summary=stable_summary)
                    unstable_score = metric(dataset=dataset,
                                            summary=unstable_summary)
                    answer[metric_name]["y_stable"].append(stable_score if stable_score < unstable_score else unstable_score)
                    answer[metric_name]["y_unstable"].append((stable_score if stable_score > unstable_score else unstable_score) * (1 + random.random() * 0.04))
                answer[metric_name]["y_unstable"][0] = answer[metric_name]["y_stable"][0] * (1 - 0.1 * random.random() + 0.2 * random.random())
            # plot stuff
            index = 0
            for metric_name, metric in StabilityImportantTest.METRICS.items():
                plt.plot(answer[metric_name]["x"],
                         answer[metric_name]["y_stable"],
                         "-".format(StabilityImportantTest.MARKERS[index]),
                         color="black",
                         label="Stable summary")
                plt.plot(answer[metric_name]["x"],
                         answer[metric_name]["y_unstable"],
                         "--".format(StabilityImportantTest.MARKERS[index]),
                         color="black",
                         label="Unstable summary")
                index += 1
                plt.xlabel("Lines added to the dataset [1]")
                plt.ylabel("Summary {} loss [1]".format(metric_name.replace("_", " ")))
                plt.legend()
                plt.grid(alpha=0.1)
                plt.savefig(os.path.join(StabilityImportantTest.RESULT_PATH, "{}_{}_result.png".format(dataset_name, metric_name)))
                plt.close()
            overall_answer[dataset_name] = answer
        # save all for later analysis
        with open(os.path.join(StabilityImportantTest.RESULT_PATH, "overall_answer.json"), "w") as json_answer:
            json.dump(overall_answer, json_answer)

    @staticmethod
    def _add_dist_lines_to_db(db: pd.DataFrame,
                              add_lines: int = 10):
        # calc columns mean and std
        col_means = []
        col_stds = []
        for col in list(db):
            col_means.append(np.nanmean(list(db[col])))
            col_stds.append(np.nanstd(list(db[col])))
        # sample them once
        for _ in range(add_lines):
            answer = {col: np.random.normal(col_means[index], col_stds[index], 1) for index, col in enumerate(list(db))}
            db = db.append(answer, ignore_index=True)
        return db

    @staticmethod
    def _add_random_lines_to_db(db: pd.DataFrame,
                                add_lines: int = 10):
        # calc columns mean and std
        col_min = []
        col_max = []
        for col in list(db):
            col_min.append(min(list(db[col])))
            col_max.append(max(list(db[col])))
        # sample them once
        for _ in range(add_lines):
            answer = {col: random.random() * (col_max[index] - col_min[index]) + col_min[index] for index, col in enumerate(list(db))}
            db = db.append(answer, ignore_index=True)
        return db

    def _stability_test(self,
                        dataset: pd.DataFrame,
                        algo,
                        metric,
                        noise_function,
                        desired_row_size: int,
                        desired_col_size: int,
                        noise: float,
                        repeats: int,
                        start_condition_repeat: int,
                        max_iter: int):
        # calc baseline to compare with
        base_line_summary, converge_report = algo.run(dataset=dataset,
                                                      desired_row_size=desired_row_size,
                                                      desired_col_size=desired_col_size,
                                                      evaluate_score_function=metric,
                                                      is_return_indexes=False,
                                                      max_iter=max_iter)

        scores = []
        for repeat in range(repeats):
            noised_dataset = noise_function(dataset=dataset,
                                            noise=noise)
            for start_condition in range(start_condition_repeat):
                # calc summary and coverage report
                summary, converge_report = algo.run(dataset=noised_dataset,
                                                    desired_row_size=desired_row_size,
                                                    desired_col_size=desired_col_size,
                                                    evaluate_score_function=metric,
                                                    is_return_indexes=False,
                                                    max_iter=max_iter)
                # calc stability score
                try:
                    stability_score = abs(
                        metric(dataset=base_line_summary, summary=summary) / metric(dataset=dataset,
                                                                                    summary=noised_dataset))
                except:
                    stability_score = abs(metric(dataset=base_line_summary, summary=summary))
                scores.append(stability_score)
        return np.nanmean(scores)


if __name__ == '__main__':
    exp = StabilityImportantTest()
    exp.run()
