# library import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, make_scorer

# project import
from experiments.dataset_summary_performance_connection import DatasetSummaryPerformanceConnection


class SummariesConnectionXAI:
    """

    """

    SEED = 73
    CV = 5

    LOW_VAR_THRESHOLD = 0.1
    GOOD_PERFORMANCE_THRESHOLD = 0.0001
    GOOD_STABILITY_THRESHOLD = 1.01
    GOOD_L2_THRESHOLD = (GOOD_PERFORMANCE_THRESHOLD**2 + GOOD_STABILITY_THRESHOLD**2)**0.5
    Y_COLS = ["performance", "stability", "l2"]
    THRESHOLDS = [GOOD_PERFORMANCE_THRESHOLD, GOOD_STABILITY_THRESHOLD, GOOD_L2_THRESHOLD]

    def __init__(self):
        pass

    @staticmethod
    def run():
        df = pd.read_csv(r"C:\Users\lazeb\Desktop\stability_feature_selection_dataset_summary\experiments\dataset_summary_performance_connection\answer.csv")
        # clear df
        df.drop(["Unnamed: 0", "ds_row_over_class", "ds_col_over_class", "ds_classes_count", "ds_cancor", "ds_kurtosis",
                 "ds_average_linearly_to_target", "ds_std_linearly_to_target",
                 "summary_row_over_class", "summary_col_over_class", "summary_classes_count", "summary_cancor", "summary_kurtosis",
                 "summary_average_linearly_to_target", "summary_std_linearly_to_target"], axis=1, inplace=True)
        df.dropna(axis=0, inplace=True)
        # drop columns that are without information
        df.drop(df.std()[df.std() < SummariesConnectionXAI.LOW_VAR_THRESHOLD].index.values, axis=1, inplace=True)
        # fix metric column
        metrics = {name: index for index, name in enumerate(set(list(df["metric"])))}
        df["metric"] = df["metric"].apply(lambda x: metrics[x])
        for trashold_index, y_col_name in enumerate(SummariesConnectionXAI.Y_COLS):
            print("Working on '{}'".format(y_col_name))
            y = df[y_col_name]
            y = y.apply(lambda val: 1 if val > SummariesConnectionXAI.THRESHOLDS[trashold_index] else 0)
            x = df.drop(SummariesConnectionXAI.Y_COLS, axis=1)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SummariesConnectionXAI.SEED)
            model = RandomForestClassifier(max_depth=4, random_state=SummariesConnectionXAI.SEED)
            scores = cross_val_score(model, x_train, y_train, cv=SummariesConnectionXAI.CV, scoring=make_scorer(accuracy_score))
            print("Cross-validation: mean = {}, std = {}".format(np.mean(scores), np.std(scores)))
            model = RandomForestClassifier(max_depth=4, random_state=SummariesConnectionXAI.SEED)
            model.fit(x_train,
                      y_train)
            test_score = accuracy_score(y_true=y_test,
                                        y_pred=model.predict(x_test))
            print("Test score: {}".format(test_score))
            model = RandomForestClassifier(max_depth=4, random_state=SummariesConnectionXAI.SEED)
            model.fit(x,
                      y)
            forest_importances = pd.Series(model.feature_importances_, index=list(x))

            fig, ax = plt.subplots(figsize=(10, 10))
            forest_importances.plot.bar(ax=ax)
            ax.set_ylabel("Feature importance")
            plt.ylim((0, 0.2))
            plt.yticks([val/100 for val in range(21)])
            plt.tight_layout()
            plt.savefig(os.path.join(DatasetSummaryPerformanceConnection.RESULT_PATH, "feature_importance_for_{}.png".format(y_col_name)))
            plt.close()


if __name__ == '__main__':
    SummariesConnectionXAI.run()
