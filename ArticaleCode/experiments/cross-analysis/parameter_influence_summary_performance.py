# library imports
import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt


class ParameterInfluenceSummaryPerformance:
    """
    This class plots the results of the parameter influence of summary
    
    """

    # CONSTS #
    RESULTS_FOLDER_NAME = "meta_analysis_results"
    RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), RESULTS_FOLDER_NAME)

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(data_path_file: str,
            results_path_file: str):
        """
        Visualizing the affect of a parameter on the performance and stability of summaries
        :param data_path_file: the data file we want to print
        :param results_path_file: the path we would like to store the results
        :return: none - save a plot to a file
        """
        # add folder for the meta analysis results
        try:
            os.mkdir(ParameterInfluenceSummaryPerformance.RESULTS_PATH)
        except Exception as error:
            pass
        # read data
        df = pd.read_csv(data_path_file)
        # convert to something easy to work with
        parameter_values = list(set(list(df["parameter"])))
        # start plot
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # print each point after some computing
        for parameter in parameter_values:
            # get the data relevant value
            sub_df = df[df["parameter"] == parameter]
            # compute the data for the dot
            mean_performance = np.nanmean(sub_df["performance"])
            std_performance = np.nanstd(sub_df["performance"])
            mean_stability = np.nanmean(sub_df["stability"])
            std_stability = np.nanstd(sub_df["stability"])
            # plot the points
            ax.scatter([mean_performance],
                       [mean_stability],
                       [parameter],
                       alpha=0.5,
                       marker="o",
                       s=20,
                       cmap="coolwarm",
                       color="black")
            # plot the error bars
            ax.plot([mean_performance - std_performance, mean_performance + std_performance],
                    [mean_stability, mean_stability],
                    [parameter, parameter],
                    alpha=0.2,
                    marker="",
                    linewidth=2,
                    color='black')
            ax.plot([mean_performance, mean_performance],
                    [mean_stability - std_stability, mean_stability + std_stability],
                    [parameter, parameter],
                    alpha=0.2,
                    marker="",
                    linewidth=2,
                    color='k')
        ax.set_xlabel("Performance")
        ax.set_ylabel("Stability")
        ax.set_zlabel("Parameter")
        plt.xlim((0, max(df["performance"])))
        plt.ylim((6, max(df["stability"])))
        plt.savefig(os.path.join(results_path_file,
                                 "parameter_influence_summary_performance.png"))
        plt.close()

        # print each point after some computing
        for parameter in parameter_values:
            # get the data relevant value
            sub_df = df[df["parameter"] == parameter]
            # compute the data for the dot
            mean_performance = np.nanmean(sub_df["performance"])
            std_performance = np.nanstd(sub_df["performance"])
            mean_stability = np.nanmean(sub_df["stability"])
            std_stability = np.nanstd(sub_df["stability"])
            # plot the points
            plt.scatter([parameter],
                        [mean_performance],
                        alpha=1,
                        marker="o",
                        s=20,
                        color="black")
            # plot the error bars
            plt.plot([parameter, parameter],
                     [mean_performance - std_performance, mean_performance + std_performance],
                     alpha=0.8,
                     marker="",
                     linewidth=2,
                     color='black')
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Performance")
        plt.xlim((-0.05, 1.05))
        plt.ylim((0, max(df["performance"]) * 1.05))
        plt.savefig(os.path.join(results_path_file,
                                 "parameter_influence_summary_performance_only_performance.png"))
        plt.close()

        # print each point after some computing
        for parameter in parameter_values:
            # get the data relevant value
            sub_df = df[df["parameter"] == parameter]
            # compute the data for the dot
            mean_performance = np.nanmean(sub_df["performance"])
            std_performance = np.nanstd(sub_df["performance"])
            mean_stability = np.nanmean(sub_df["stability"])
            std_stability = np.nanstd(sub_df["stability"])
            # plot the points
            plt.scatter([parameter],
                        [mean_stability],
                        alpha=1,
                        marker="o",
                        s=20,
                        color="black")
            # plot the error bars
            plt.plot([parameter, parameter],
                     [mean_stability - std_stability, mean_stability + std_stability],
                     alpha=0.8,
                     marker="",
                     linewidth=2,
                     color='black')
        plt.xlabel("Parameter")
        plt.ylabel("Stability")
        plt.xlim((-0.05, 1.05))
        plt.ylim((0, max(df["stability"]) * 1.05))
        plt.savefig(os.path.join(results_path_file,
                                 "parameter_influence_summary_performance_only_stability.png"))
        plt.close()


if __name__ == '__main__':
    ParameterInfluenceSummaryPerformance.run(data_path_file=os.path.join(os.path.dirname(__file__), "data.csv"),
                                             results_path_file=os.path.dirname(__file__))
