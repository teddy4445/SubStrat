# library import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# project import
from experiments.genetic_summary_multiple_optimization_functions import GeneticSummaryMultipleOptimizationMetrics


class SummariesConnectionPlots:
    """

    """

    def __init__(self):
        pass

    @staticmethod
    def run():
        answer_df = pd.read_csv(r"C:\Users\lazeb\Desktop\stability_feature_selection_dataset_summary\experiments\multi_metric_optimization\answer.csv")
        ds_names = list(GeneticSummaryMultipleOptimizationMetrics.DATASETS.keys())
        for row_index, row in answer_df.iterrows():
            plt.scatter(row["performance"],
                        row["stability"],
                        marker="o" if "hmean" in row["metric"] else "^" if "mean" in row["metric"] else "P",
                        color=GeneticSummaryMultipleOptimizationMetrics.COLORS[ds_names.index(row["dataset"])])
        plt.xlabel("Performance [1]")
        plt.ylabel("Stability [1]")
        plt.legend(handles=[mlines.Line2D([], [], color='black', marker='o', markersize=8, linewidth=0, label='Harmonic Mean'),
                            mlines.Line2D([], [], color='black', marker='^', markersize=8, linewidth=0, label='Mean'),
                            mlines.Line2D([], [], color='black', marker='P', markersize=8, linewidth=0, label='Only Performance')])
        plt.grid(alpha=0.1)
        plt.xlim((0, 2))
        plt.ylim((0, 20))
        plt.savefig(os.path.join(GeneticSummaryMultipleOptimizationMetrics.RESULT_PATH, "answer_scatter.png"))
        plt.close()

        h_mean_performance = []
        h_mean_stability = []
        mean_performance = []
        mean_stability = []
        regular_performance = []
        regular_stability = []
        for row_index, row in answer_df.iterrows():
            if "hmean" in row["metric"]:
                h_mean_performance.append(row["performance"])
                h_mean_stability.append(row["stability"])
            elif "mean" in row["metric"]:
                mean_performance.append(row["performance"])
                mean_stability.append(row["stability"])
            else:
                regular_performance.append(row["performance"])
                regular_stability.append(row["stability"])
        plt.bar(list(range(3)),
                [np.mean(h_mean_performance), np.mean(mean_performance), np.mean(regular_performance)],
                yerr=[np.std(h_mean_performance), np.std(mean_performance), np.std(regular_performance)],
                width=0.8,
                color="black",
                capsize=3)
        plt.xlabel("Optimization category")
        plt.ylabel("Performance")
        plt.xticks(range(3), ["Harmonic Mean", "Mean", "Performance Only"])
        plt.savefig(os.path.join(GeneticSummaryMultipleOptimizationMetrics.RESULT_PATH, "metric_compare_performance.png"))
        plt.bar(list(range(3)),
                [np.mean(h_mean_stability), np.mean(mean_stability), np.mean(regular_stability)],
                yerr=[np.std(h_mean_stability), np.std(mean_stability), np.std(regular_stability)],
                width=0.8,
                color="black",
                capsize=3)
        plt.xlabel("Optimization category")
        plt.ylabel("Stability")
        plt.xticks(range(3), ["Harmonic Mean", "Mean", "Performance Only"])
        plt.savefig(os.path.join(GeneticSummaryMultipleOptimizationMetrics.RESULT_PATH, "metric_compare_stability.png"))


if __name__ == '__main__':
    SummariesConnectionPlots.run()
