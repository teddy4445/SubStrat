# library imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines


class VisualResultsFromCSV:
    """
    This class visualize the results from a CSV of
    index, metric, dataset, performance, stability, std_performance, std_stability
    """

    # CONSTS #
    MARKERS = ["o", "^", "P", "s", "*", "+", "X", "D", "d"]
    COLORS = ["black", "blue", "red", "green", "yellow", "purple", "orange", "gray", "peru", "aqua", "violet",
              "crimson", "indigo", "lime", "darkolivegreen"]

    # END - CONSTS #

    def __init__(self):
        pass

    @staticmethod
    def run(data_path: str,
            save_path: str):
        df = pd.read_csv(data_path)

        df = df[df["metric"] == "mean_entropy"]

        markers_map = {name: index for index, name in enumerate(set(list(df["metric"])))}
        colors_map = {name: index for index, name in enumerate(set(list(df["dataset"])))}

        legend = {}

        for row_index, row in df.iterrows():
            scatter_obj = plt.errorbar(row["performance"],
                                       row["stability"],
                                       xerr=row["std_performance"],
                                       yerr=row["std_stability"],
                                       elinewidth=0,
                                       capsize=2,
                                       marker=VisualResultsFromCSV.MARKERS[markers_map[row["metric"]]],
                                       color="black")#VisualResultsFromCSV.COLORS[colors_map[row["dataset"]]])
            if row["metric"] not in legend:
                legend[row["metric"]] = scatter_obj

        plt.xlabel("Performance [1]")
        plt.ylabel("Stability [1]")
        plt.xlim((0, 1 + max(list(df["performance"]))))
        plt.ylim((0, 1 + max(list(df["stability"]))))
        plt.legend(tuple(legend.values()), tuple(legend.keys()))
        plt.grid(alpha=0.1, color="black")
        plt.savefig(save_path)


if __name__ == '__main__':
    VisualResultsFromCSV.run(
        data_path=os.path.join(os.path.dirname(__file__), "performance_stabiliy_large.csv"),
        save_path=os.path.join(os.path.dirname(__file__),
                               "performance_stabiliy_large_graph_mean_entropy.png")
    )
