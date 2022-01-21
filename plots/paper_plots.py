# library imports
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class PaperPlots:
    """
    Plots for the paper
    """

    def __init__(self):
        pass

    @staticmethod
    def run():
        """
        Plot all the figures for the paper from the results of the experiments
        """
        #PaperPlots.ga_generation_plot()
        PaperPlots.heatmap_one_d()
        PaperPlots.heatmap_one_d_time()
        #PaperPlots.heatmap()

    @staticmethod
    def heatmap_one_d_time():
        time_data = [[3.24, 3.35, 3.41, 3.17, 3.98, 4.35, 5.31, 5.88],
                    [12.26, 12.11, 12.78, 14.16, 15.71, 17.37, 18.83, 20.04],
                    [35.02, 35.93, 35.56, 35.2, 37.18, 41.1, 42.71, 43.8],
                    [39.31, 40.84, 39.72, 40.69, 43.17, 47.09, 50.78, 54.15],
                    [41.74, 42.07, 41.98, 43.1, 46.57, 51.14, 55.9, 56.89],
                    [44.27, 43.67, 44.12, 44.61, 53.76, 59.52, 60.79, 63.06],
                    [50.71, 51.2, 51.5, 53.71, 58.87, 68.07, 70.25, 75.4],
                    [71.41, 71.03, 71.79, 71.48, 76.05, 80.39, 83.66, 88.46]]
        time_data = [[(1-val/88.46) for val in row]for row in time_data]
        means = np.nanmean(time_data, axis=1)
        stds = np.nanstd(time_data, axis=1)
        ds_sizes = [129880, 15300, 1000, 10000, 8125, 57660, 7000, 1700, 795401, 1000000, 17415]
        n = sum(ds_sizes) / len(ds_sizes)
        x = [round(math.log(n)), round(math.sqrt(n)), round(0.01 * n), round(0.05 * n), round(0.1 * n), round(0.25 * n),
             round(0.5 * n), round(n)]
        names = ["$log_2 (n)$ (", "$\sqrt{n}$ (", "$0.01nv (", "$0.05n$ (", "$0.1n$ (", "$0.25n$ (", "$0.5n$ (",
                 "$n$ ("]
        ax = plt.subplot(111)
        plt.errorbar(x=x,
                     y=means,
                     yerr=stds,
                     fmt="-o",
                     color="black",
                     linewidth=2,
                     markersize=6,
                     capsize=2)
        plt.ylim((0, 1))
        plt.xlabel("Portion of samples")
        plt.ylabel("Average reduction in computation time")
        plt.grid(alpha=0.1,
                 color="black")
        # plt.yticks([0 + 0.01 * i for i in range(15)])
        plt.xticks(x, [names[index] + str(x[index]) + ")" for index in range(len(x))], rotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.legend()
        plt.savefig("samples_on_time_line_plot.png")
        plt.close()

        means = np.nanmean(time_data, axis=0)
        stds = np.nanstd(time_data, axis=0)
        ds_sizes = [23, 5, 171, 18, 23, 7, 9, 15, 123, 7, 15]
        n = sum(ds_sizes) / len(ds_sizes)
        x = [round(math.log(n)), round(math.sqrt(n)), 1, round(0.05 * n), round(0.1 * n), round(0.25 * n),
             round(0.5 * n), round(n)]
        names = ["$log_2 (n)$ (", "$\sqrt{n}$ (", "$0.01nv (", "$0.05n$ (", "$0.1n$ (", "$0.25n$ (", "$0.5n$ (",
                 "$n$ ("]
        new_x = [(x[index], names[index]) for index in range(len(x))]
        new_x = sorted(new_x, key=lambda x: x[0])
        x = [val[0] for val in new_x]
        names = [val[1] for val in new_x]
        ax = plt.subplot(111)
        plt.errorbar(x=x,
                     y=means,
                     yerr=stds,
                     fmt="-o",
                     color="black",
                     linewidth=2,
                     markersize=6,
                     capsize=2)
        plt.ylim((0, 1))
        plt.xlabel("Portion of features")
        plt.ylabel("Average reduction in computation time")
        plt.grid(alpha=0.1,
                 color="black")
        # plt.yticks([0 + 0.01 * i for i in range(15)])
        plt.xticks(x, [names[index] + str(x[index]) + ")" for index in range(len(x))], rotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.legend()
        plt.savefig("features_on_time_line_plot.png")
        plt.close()

    @staticmethod
    def heatmap_one_d():
        acc_data = [[0.42, 0.43, 0.43, 0.43, 0.58, 0.58, 0.58, 0.58],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.69, 0.69, 0.69],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.69, 0.69, 0.70],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.69, 0.70, 0.70],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.70, 0.70, 0.70],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.70, 0.70, 0.70],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.70, 0.70, 0.70],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.70, 0.70, 0.70]]
        acc_data = [[0.7 - val for val in row] for row in acc_data]
        means = np.nanmean(acc_data, axis=1)
        stds = np.nanstd(acc_data, axis=1)
        ds_sizes = [129880, 15300, 1000, 10000, 8125, 57660, 7000, 1700, 795401, 1000000, 17415]
        n = sum(ds_sizes) / len(ds_sizes)
        x = [round(math.log(n)), round(math.sqrt(n)), round(0.01 * n), round(0.05 * n), round(0.1 * n), round(0.25 * n),
             round(0.5 * n), round(n)]
        names = ["$log_2 (n)$ (", "$\sqrt{n}$ (", "$0.01nv (", "$0.05n$ (", "$0.1n$ (", "$0.25n$ (", "$0.5n$ (",
                 "$n$ ("]
        ax = plt.subplot(111)
        plt.errorbar(x=x,
                     y=means,
                     yerr=stds,
                     fmt="-o",
                     color="black",
                     linewidth=2,
                     markersize=6,
                     capsize=2)
        plt.ylim((0, 0.3))
        plt.xlabel("Portion of samples")
        plt.ylabel("Average reduction in accuracy")
        plt.grid(alpha=0.1,
                 color="black")
        # plt.yticks([0 + 0.01 * i for i in range(15)])
        plt.xticks(x, [names[index] + str(x[index]) + ")" for index in range(len(x))], rotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.legend()
        plt.savefig("samples_on_accuracy_line_plot.png")
        plt.close()

        means = np.nanmean(acc_data, axis=0)
        stds = np.nanstd(acc_data, axis=0)
        ds_sizes = [23, 5, 171, 18, 23, 7, 9, 15, 123, 7, 15]
        n = sum(ds_sizes) / len(ds_sizes)
        x = [round(math.log(n)), round(math.sqrt(n)), 1, round(0.05 * n), round(0.1 * n), round(0.25 * n),
             round(0.5 * n), round(n)]
        names = ["$log_2 (n)$ (", "$\sqrt{n}$ (", "$0.01nv (", "$0.05n$ (", "$0.1n$ (", "$0.25n$ (", "$0.5n$ (",
                 "$n$ ("]
        new_x = [(x[index], names[index]) for index in range(len(x))]
        new_x = sorted(new_x, key=lambda x: x[0])
        x = [val[0] for val in new_x]
        names = [val[1] for val in new_x]
        ax = plt.subplot(111)
        plt.errorbar(x=x,
                     y=means,
                     yerr=stds,
                     fmt="-o",
                     color="black",
                     linewidth=2,
                     markersize=6,
                     capsize=2)
        plt.ylim((0, 0.3))
        plt.xlabel("Portion of features")
        plt.ylabel("Average reduction in accuracy")
        plt.grid(alpha=0.1,
                 color="black")
        # plt.yticks([0 + 0.01 * i for i in range(15)])
        plt.xticks(x, [names[index] + str(x[index]) + ")" for index in range(len(x))], rotation=90)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.tight_layout()
        plt.legend()
        plt.savefig("features_on_accuracy_line_plot.png")
        plt.close()

    @staticmethod
    def heatmap():
        acc_data = [[0.42, 0.43, 0.43, 0.43, 0.58, 0.58, 0.58, 0.58],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.69, 0.69, 0.69],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.69, 0.69, 0.70],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.69, 0.70, 0.70],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.70, 0.70, 0.70],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.70, 0.70, 0.70],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.70, 0.70, 0.70],
                    [0.43, 0.43, 0.43, 0.43, 0.58, 0.70, 0.70, 0.70]]
        acc_data = [[100*val/0.7 for val in row]for row in acc_data]
        time_data = [[3.24, 3.35, 3.41, 3.17, 3.98, 4.35, 5.31, 5.88],
                    [12.26, 12.11, 12.78, 14.16, 15.71, 17.37, 18.83, 20.04],
                    [35.02, 35.93, 35.56, 35.2, 37.18, 41.1, 42.71, 43.8],
                    [39.31, 40.84, 39.72, 40.69, 43.17, 47.09, 50.78, 54.15],
                    [41.74, 42.07, 41.98, 43.1, 46.57, 51.14, 55.9, 56.89],
                    [44.27, 43.67, 44.12, 44.61, 53.76, 59.52, 60.79, 63.06],
                    [50.71, 51.2, 51.5, 53.71, 58.87, 68.07, 70.25, 75.4],
                    [71.41, 71.03, 71.79, 71.48, 76.05, 80.39, 83.66, 88.46]]
        time_data = [[100*(1-val/88.46) for val in row]for row in time_data]
        indexes = ["$log_2(n)$, $\sqrt{n}$, 0.01n, 0.05n, 0.1n, 0.25n, 0.5n, n".replace(" ", "").split(",")]
        cols = indexes.copy()
        df = pd.DataFrame(data=acc_data, index=indexes, columns=cols)
        ax = sns.heatmap(df, vmin=0, vmax=100, cmap="coolwarm")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.ylabel("Rows")
        plt.xlabel("Columns")
        plt.tight_layout()
        plt.savefig("acc_heatmap.png")
        plt.close()

        df = pd.DataFrame(data=time_data, index=indexes, columns=cols)
        ax = sns.heatmap(df, vmin=0, vmax=100, cmap="coolwarm")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.ylabel("Rows")
        plt.xlabel("Columns")
        plt.tight_layout()
        plt.savefig("time_heatmap.png")
        plt.close()

        acc_data = np.array(acc_data)
        time_data = np.array(time_data)
        df = pd.DataFrame(data=acc_data * 0.1 + time_data * 0.9, index=indexes, columns=cols)
        ax = sns.heatmap(df, vmin=0, vmax=100, cmap="coolwarm")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.ylabel("Rows")
        plt.xlabel("Columns")
        plt.tight_layout()
        plt.savefig("acc_01_time_09_average_heatmap.png")
        plt.close()
        df = pd.DataFrame(data=acc_data * 0.5 + time_data * 0.5, index=indexes, columns=cols)
        ax = sns.heatmap(df, vmin=0, vmax=100, cmap="coolwarm")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.ylabel("Rows")
        plt.xlabel("Columns")
        plt.tight_layout()
        plt.savefig("acc_05_time_05_average_heatmap.png")
        plt.close()
        df = pd.DataFrame(data=acc_data * 0.9 + time_data * 0.1, index=indexes, columns=cols)
        ax = sns.heatmap(df, vmin=0, vmax=100, cmap="coolwarm")
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        plt.ylabel("Rows")
        plt.xlabel("Columns")
        plt.tight_layout()
        plt.savefig("acc_09_time_01_average_heatmap.png")
        plt.close()

    @staticmethod
    def ga_generation_plot():
        df = {"GA save time ratio": [21.47, 14.15, 10.34, 08.46, 06.62, 05.89, 05.09, 04.56, 04.15, 03.59, 03.26, 02.96],
              "GA accuracy":        [0.135, 0.101, 0.087, 0.043, 0.011, 0.009, 0.007, 0.006, 0.005, 0.003, 0.002, 0.002],
              "GA std":             [0.021, 0.019, 0.017, 0.014, 0.012, 0.010, 0.006, 0.004, 0.004, 0.003, 0.001, 0.001],
              "MAB save time ratio": [27.47, 14.14, 10.89, 08.43, 05.26, 04.12, 03.48, 02.87],
              "MAB accuracy":        [0.102, 0.061, 0.049, 0.038, 0.033, 0.033, 0.033, 0.032],
              "MAB std":             [0.019, 0.017, 0.011, 0.009, 0.007, 0.005, 0.005, 0.005]}
        ax = plt.subplot(111)
        plt.plot(df["GA save time ratio"],
                 df["GA accuracy"],
                 "-o",
                 color="#34a853",
                 linewidth=2,
                 label="GA",
                 markersize=6)
        plt.fill_between(df["GA save time ratio"],
                         [df["GA accuracy"][index] + df["GA std"][index] for index in range(len(df["GA accuracy"]))],
                         [df["GA accuracy"][index] - df["GA std"][index] for index in range(len(df["GA accuracy"]))],
                         color="#34a853",
                         alpha=0.2)
        plt.plot(df["MAB save time ratio"],
                 df["MAB accuracy"],
                 "-D",
                 color="#4285f4",
                 linewidth=2,
                 label="MAB",
                 markersize=6)
        plt.fill_between(df["MAB save time ratio"],
                         [df["MAB accuracy"][index] + df["MAB std"][index] for index in range(len(df["MAB accuracy"]))],
                         [df["MAB accuracy"][index] - df["MAB std"][index] for index in range(len(df["MAB accuracy"]))],
                         color="#4285f4",
                         alpha=0.2)
        plt.xlabel("Average ratio of reduction in time")
        plt.ylabel("Average reduction in accuracy")
        plt.grid(alpha=0.1,
                 color="black")
        plt.xlim(28, 0)
        plt.ylim((0, 0.14))
        plt.yticks([0 + 0.01 * i for i in range(15)])
        plt.xticks([2*i + 1 for i in range(14)])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.legend()
        plt.savefig("time_performance.png")
        plt.close()


if __name__ == '__main__':
    PaperPlots.run()
