# library imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GenerationGraph:
    """
    Just plot generation plots
    """

    def __init__(self):
        pass

    @staticmethod
    def run_absolute():
        # generation,avg_absolute_time_change_min,avg_relative_acc,avg_absolute_acc
        df = pd.read_csv(r"C:\Users\lazeb\Desktop\stability_feature_selection_dataset_summary\experiments\automl\generation_sensitivity_auto_sklearn.csv")
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Genetic algorithm generation')
        ax1.set_ylabel('Absolute relative time save',
                       color="blue")
        ax1.plot(df["generation"],
                 df["avg_absolute_time_change_min"],
                 "--o",
                 color="blue")
        ax1.tick_params(axis='y',
                        labelcolor="blue")
        ax1.set_yticks([25, 30, 35, 40, 45, 50, 55])

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'orange'
        ax2.set_ylabel('Absolute relative accuracy reduce',
                       color=color)  # we already handled the x-label with ax1
        ax2.plot(df["generation"],
                 df["avg_absolute_acc"],
                 "--o",
                 color=color)
        ax2.tick_params(axis='y',
                        labelcolor=color)
        ax2.set_yticks([0, 0.005, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175])

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig("generation_graph_absolute.png")
        plt.close()

    @staticmethod
    def run_relative():
        # generation,avg_relative_time_change,avg_absolute_time_change_min,avg_relative_acc,avg_absolute_acc
        df = pd.read_csv(r"C:\Users\lazeb\Desktop\stability_feature_selection_dataset_summary\experiments\automl\generation_sensitivity_auto_sklearn.csv")
        fig, ax1 = plt.subplots()

        ax1.set_xlabel('Genetic algorithm generation')
        ax1.set_ylabel('Average relative time save',
                       color="blue")
        ax1.plot(df["generation"],
                 df["avg_relative_time_change"],
                 "--o",
                 color="blue")
        ax1.tick_params(axis='y',
                        labelcolor="blue")
        ax1.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'orange'
        ax2.set_ylabel('Average relative accuracy reduce',
                       color=color)  # we already handled the x-label with ax1
        ax2.plot(df["generation"],
                 df["avg_relative_acc"],
                 "--o",
                 color=color)
        ax2.tick_params(axis='y',
                        labelcolor=color)
        ax2.set_yticks([0, 0.006, 0.05, 0.1, 0.15, 0.2, 0.25])

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig("generation_graph_relative.png")
        plt.close()


if __name__ == '__main__':
    GenerationGraph.run_relative()
    GenerationGraph.run_absolute()
