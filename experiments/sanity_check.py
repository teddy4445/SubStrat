# library imports
import os
import math
import random
import numpy as np
import pandas as pd
from glob import glob
from pandas.core.dtypes.common import is_numeric_dtype

# project imports
from methods.summary_wellness_scores import SummaryWellnessScores
from experiments.automl.auto_sklearn_experiment import AutoSKlearnExperiment
from summary_algorithms.genetic_algorithm_summary_algorithm import GeneticSummary


class SanityCheck:
    """
    Check if the results we get makes sence
    """

    def __init__(self):
        pass

    @staticmethod
    def run():
        # fix data
        dfs = {os.path.basename(path).replace(".csv", ""): pd.read_csv(path)
               for path in glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "big_data", "*.csv"))}
        # clear data
        for name, df in dfs.items():
            # remove what we do not need
            df.drop([col for col in list(df) if not is_numeric_dtype(df[col])], axis=1, inplace=True)
            # remove _rows with nan
            df.dropna(inplace=True)
            print("Fixed dataset {}".format(name))
        # calc result
        answer = "dataset_name,dataset_score,summary_score\n"
        for name, df in dfs.items():
            answer += "{},".format(name)
            # calc score
            ds_score = SummaryWellnessScores._mean_entropy(df)
            answer += "{:.3f},".format(ds_score)
            print("Dataset {} score: {:.3f}".format(name, ds_score))
            # calc data subset (summary)
            summary = GeneticSummary.run(dataset=df,
                                         desired_row_size=round(math.sqrt(df.shape[0])),
                                         desired_col_size=round(df.shape[1] * 0.25),
                                         evaluate_score_function=SummaryWellnessScores.mean_entropy,
                                         is_return_indexes=False,
                                         max_iter=30)
            # calc the summary's score
            summary_score = SummaryWellnessScores._mean_entropy(summary)
            answer += "{:.3f},".format(summary_score)
            print("Dataset {} summary score: {:.3f}".format(name, summary_score))
            # calc pipeline thing
            AutoSKlearnExperiment.run(target_feature_name="target")
            answer += "\n"
        with open("sanity_check_results.csv", "w") as answer_file:
            answer_file.write(answer)


if __name__ == '__main__':
    SanityCheck.run()
