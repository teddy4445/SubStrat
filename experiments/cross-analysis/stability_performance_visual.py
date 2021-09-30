# library imports
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


class StabilityPerformanceVisual:
    """
    This class analyze the connection between performance and stability in different summarization algorithms
    """

    # CONSTS #
    MARKERS = ["o", "^", "P", "s", "*", "+", "X", "D", "d"]
    COLORS = ["black", "blue", "red", "green", "yellow", "purple", "orange", "gray"]

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run_all(summary_data_path: str,
                summary_save_path: str,
                algo_data_path: str,
                algo_save_path: str):
        """
        StabilityPerformanceVisual.show_summary_performance_stability_avg(data_path=summary_data_path,
                                                                          save_path=summary_save_path)
        StabilityPerformanceVisual.show_summary_performance_stability_all(data_path=summary_data_path,
                                                                          save_path=summary_save_path)
                                                                          """
        StabilityPerformanceVisual.show_algo_performance_stability_all(data_path=algo_data_path,
                                                                       save_path=algo_save_path)

    @staticmethod
    def show_summary_performance_stability_avg(data_path: str,
                                               save_path: str):
        df = pd.read_csv(data_path)
        df = df.groupby(['metric', "dataset"]).mean()
        df["dataset"] = df["performance"].apply(lambda x: "mean")
        df["metric"] = df["performance"].apply(lambda x: "mean")
        # run all options
        StabilityPerformanceVisual.show_summary_performance_stability(data_path=data_path,
                                                                      save_path=os.path.join(
                                                                          os.path.dirname(save_path),
                                                                          "mean_{}".format(
                                                                              os.path.basename(save_path))),
                                                                      df=df)

    @staticmethod
    def show_summary_performance_stability_all(data_path: str,
                                               save_path: str):
        df = pd.read_csv(data_path)
        dbs = list(set(list(df["dataset"])))
        dbs.append("")  # add none value
        metrics = list(set(list(df["metric"])))
        metrics.append("")  # add none value
        # run all options
        [[StabilityPerformanceVisual.show_summary_performance_stability(data_path=data_path,
                                                                        save_path=os.path.join(
                                                                            os.path.dirname(save_path),
                                                                            "{}_{}_{}".format(filter_db,
                                                                                              filter_metric,
                                                                                              os.path.basename(
                                                                                                  save_path))),
                                                                        df=df,
                                                                        filter_db=filter_db,
                                                                        filter_metric=filter_metric)
          for filter_db in dbs]
         for filter_metric in metrics]

    @staticmethod
    def show_algo_performance_stability_all(data_path: str,
                                            save_path: str):
        df = pd.read_csv(data_path)
        algos = list(set(list(df["algo"])))
        # run all options
        [StabilityPerformanceVisual.show_algo_performance_stability(data_path=data_path,
                                                                    save_path=os.path.join(
                                                                        os.path.dirname(save_path),
                                                                        "{}_{}".format(filter_algo,
                                                                                       os.path.basename(
                                                                                           save_path))),
                                                                    df=df,
                                                                    filter_algo=filter_algo)
         for filter_algo in algos]

    @staticmethod
    def show_summary_performance_stability(data_path: str,
                                           save_path: str,
                                           filter_db: str = "",
                                           filter_metric: str = "",
                                           df=None):
        """
        Load the file from path and show the connection between the summary's performance and stability
        """
        # add folder for the meta analysis results
        try:
            os.mkdir(os.path.dirname(save_path))
        except Exception as error:
            pass
        # read data
        df = pd.read_csv(data_path) if df is None else df
        # filter if requested
        if filter_db != "":
            df = df[df["dataset"] == filter_db]
        if filter_metric != "":
            df = df[df["metric"] == filter_metric]
        # get scatter style markers
        dbs = list(set(list(df["dataset"])))
        metrics = list(set(list(df["metric"])))

        # print scatter plot
        for row_index, row in df.iterrows():
            plt.scatter(row["performance"],
                        row["stability"],
                        s=20,
                        marker=StabilityPerformanceVisual.MARKERS[metrics.index(row["metric"])],
                        color=StabilityPerformanceVisual.COLORS[dbs.index(row["dataset"])])
        plt.xlabel("Performance's loss [1]")
        plt.ylabel("Stability's loss [1]")
        plt.savefig(os.path.join(save_path))
        plt.close()

    @staticmethod
    def show_algo_performance_stability(data_path: str,
                                        save_path: str,
                                        filter_db: str = "",
                                        filter_metric: str = "",
                                        filter_algo: str = "",
                                        df=None):
        """
        Load the file from path and show the connection between the summary's performance and stability
        """
        # add folder for the meta analysis results
        try:
            os.mkdir(os.path.dirname(save_path))
        except Exception as error:
            pass
        # read data
        df = pd.read_csv(data_path) if df is None else df
        # filter if requested
        if filter_db != "":
            df = df[df["dataset"] == filter_db]
        if filter_metric != "":
            df = df[df["metric"] == filter_metric]
        if filter_algo != "":
            df = df[df["algo"] == filter_algo]
        # get scatter style markers
        dbs = list(set(list(df["dataset"])))
        metrics = list(set(list(df["metric"])))

        # print scatter plot
        for row_index, row in df.iterrows():
            plt.errorbar(row["mean_performance"],
                         row["mean_stability"],
                         yerr=row["std_stability"],
                         xerr=row["std_performance"],
                         marker=StabilityPerformanceVisual.MARKERS[metrics.index(row["metric"])],
                         color=StabilityPerformanceVisual.COLORS[dbs.index(row["dataset"])])
        plt.xlabel("Performance's loss [1]")
        plt.ylabel("Stability's loss [1]")
        plt.savefig(os.path.join(save_path))
        plt.close()


if __name__ == '__main__':
    StabilityPerformanceVisual.run_all(
        summary_data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "performance_stability_results",
                                       "summary_table.csv"),
        summary_save_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "performance_stability_results", "visual",
                                       "summary_visual.png"),
        algo_data_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "performance_stability_results",
                                    "algo_table.csv"),
        algo_save_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), "performance_stability_results",
                                    "visual",
                                    "algo_visual.png")
    )
