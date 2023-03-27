# library imports
import os
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import _tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer, r2_score
from sklearn.model_selection import train_test_split


class StabilityPerformanceLearnConnection:
    """
    This class analyze the connection between performance and stability in different summarization algorithms
    """

    # CONSTS #
    SEED = 73
    TEST_RATE = 0.2
    K_FOLD = 5
    MAX_DEPTH = 4
    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def dt_find_in_algo_connection(data_path: str,
                                   save_path: str):
        """
        Train a DT model and try to find connection between the
        """
        # add folder for the meta analysis results
        try:
            os.mkdir(os.path.dirname(save_path))
        except Exception as error:
            pass
        # read data
        df = pd.read_csv(data_path)
        # split it
        x = df.drop(["Unnamed: 0","std_performance","mean_stability","std_stability"], axis=1)
        maps = {}
        # fix columns to numbers
        for col in ["algo","metric","dataset"]:
            values = {name: i for i, name in enumerate(set(list(x[col])))}
            x[col] = x[col].apply(lambda x: values[x])
            maps[col] = values
        y = df["mean_stability"]
        # split to train and test
        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=StabilityPerformanceLearnConnection.TEST_RATE,
                                                            random_state=StabilityPerformanceLearnConnection.SEED)
        # train model
        clf = DecisionTreeRegressor(random_state=StabilityPerformanceLearnConnection.SEED,
                                    max_depth=StabilityPerformanceLearnConnection.MAX_DEPTH)
        cross_scores = cross_val_score(clf,
                                       x_train,
                                       y_train,
                                       cv=StabilityPerformanceLearnConnection.K_FOLD,
                                       scoring=make_scorer(mean_absolute_error))
        cross_str = "Cross validation model: mean = {}, std = {}".format(np.mean(cross_scores), np.std(cross_scores))
        print(cross_str)
        # train again
        clf = DecisionTreeRegressor(random_state=StabilityPerformanceLearnConnection.SEED,
                                    max_depth=StabilityPerformanceLearnConnection.MAX_DEPTH)
        clf.fit(x_train,
                y_train)
        # test the model
        y_pred = clf.predict(x_test)
        test_score = mean_absolute_error(y_true=y_test,
                                         y_pred=y_pred)
        test_str = "Test score: {}".format(test_score)
        print(test_str)
        # train on all the data (fitting)
        clf = DecisionTreeRegressor(random_state=StabilityPerformanceLearnConnection.SEED,
                                    max_depth=StabilityPerformanceLearnConnection.MAX_DEPTH)
        clf.fit(x,
                y)
        # showcase result to analyze
        dt_code = StabilityPerformanceLearnConnection.convert_decision_tree_python(name="general_{}".format(StabilityPerformanceLearnConnection.MAX_DEPTH),
                                                                                   dt=clf,
                                                                                   feature_names=list(x))
        # add meta data to string
        dt_code += "\n\nModels results:\n{}\n{}\n\nMaps:\n{}".format(cross_str, test_str, "\n".join(["{}: {}".format(name, val) for name, val in maps.items()]))
        # save it
        with open(save_path, "w") as answer_file:
            answer_file.write(dt_code)

        # test coronation for each algorithm
        algos = list(set(list(df["algo"])))
        for algo in algos:
            algo_df = df[df["algo"] == algo]
            algo_df_x = algo_df["mean_performance"]
            algo_df_y = algo_df["mean_stability"]
            reg = LinearRegression()
            reg.fit([[val] for val in list(algo_df_x)],
                    list(algo_df_y))
            predict_y = reg.predict([[val] for val in list(algo_df_x)])
            score = r2_score(algo_df_y, predict_y)
            print("For '{}' the model has R^2 = {}".format(algo, score))

    @staticmethod
    def convert_decision_tree_python(name: str,
                                     dt: tree.DecisionTreeClassifier,
                                     feature_names: list) -> str:
        """
        Convert a decision tree (sklearn) object to SAT problem (as string).
        :param dt: decision tree (sklearn.tree.DecisionTreeClassifier)
        :param name: the classification problem function name
        :param feature_names: the names of the features to make sure it is easy to analyze
        :return: Python code for the tree to run (string)
        """
        tree_ = dt.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        answer = "def predict_{}({}):".format(name, ", ".join(feature_names))

        def recurse(node, depth, rec_answer: str):
            indent = "\t" * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                rec_answer += "\n{}if {} <= {}:".format(indent, name, threshold)
                rec_answer = recurse(tree_.children_left[node], depth + 1, rec_answer)
                rec_answer += "\n{}else:  # if {} > {}".format(indent, name, threshold)
                rec_answer = recurse(tree_.children_right[node], depth + 1, rec_answer)
            else:
                rec_answer += "\n{}return {}".format(indent, np.argmax(tree_.value[node]))
            return rec_answer

        answer = recurse(0, 1, answer)
        return answer


if __name__ == '__main__':
    StabilityPerformanceLearnConnection.dt_find_in_algo_connection(
        data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "performance_stability_results",
                               "algo_table.csv"),
        save_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "performance_stability_results", "learning_connection",
                               "algo.txt")
    )
