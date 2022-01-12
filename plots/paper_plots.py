# library imports
import os
import pandas as pd
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
        PaperPlots.ga_generation_plot()

    @staticmethod
    def ga_generation_plot():
        """

        :return:
        """
        df = pd.read_csv(r"C:\Users\lazeb\Desktop\generations_over_time_and_performance.csv")
        ax = plt.subplot(111)
        plt.plot(df["generation"],
                 df["mean_accuracy"],
                 "-o",
                 color="black",
                 linewidth=2,
                 markersize=6)
        plt.xlabel("GA generations")
        plt.ylabel("Average performance")
        plt.grid(alpha=0.1,
                 color="black")
        plt.ylim((0.55, 0.75))
        plt.xticks([5*i for i in range(12)])
        plt.xlim((-1, 56))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig("generation_performance.png")
        plt.close()


        ax = plt.subplot(111)
        plt.plot(df["generation"],
                 df["mean_time_min"],
                 "-o",
                 color="black",
                 linewidth=2,
                 markersize=6)
        plt.xlabel("GA generations")
        plt.ylabel("Average computation time in minutes")
        plt.grid(alpha=0.1,
                 color="black")
        plt.ylim((0, 35))
        plt.xticks([5*i for i in range(12)])
        plt.xlim((-1, 56))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        plt.savefig("generation_time.png")
        plt.close()


if __name__ == '__main__':
    PaperPlots.run()
