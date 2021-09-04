# library imports
import os
import numpy as np
import matplotlib.pyplot as plt


class AnalysisConvergeProcess:
    """
    This class studies and plot the converge process of a summary algorithm
    """

    def __init__(self):
        pass

    # HELP #

    @staticmethod
    def _iou(set_a: set,
             set_b: set) -> float:
        return 1 - len(set_a.intersection(set_b))/len(set_a.union(set_b))

    # END - HELP #

    @staticmethod
    def iou_greedy_converge(rows_list: list,
                            col_list: list,
                            save_path: str):
        """
        Plot a scatter plot of the changes in the summary rows and cols with the IOU metric over the process
        :param rows_list: list of list of rows indexes from the summary algorithm
        :param col_list: list of list of rows indexes from the summary algorithm
        :return: save plot
        """
        x_values = [AnalysisConvergeProcess._iou(set_a=set(rows_list[row_index]),
                                                 set_b=set(rows_list[row_index+1]))
                    for row_index in range(len(rows_list)-1)]
        y_values = [AnalysisConvergeProcess._iou(set_a=set(col_list[col_index]),
                                                 set_b=set(col_list[col_index+1]))
                    for col_index in range(len(col_list)-1)]
        plt.scatter(x_values,
                    y_values,
                    s=20,
                    marker='o',
                    c="black")
        for i, txt in enumerate(list(range(len(x_values)))):
            plt.annotate("{}->{}".format(i+1, i+2), (x_values[i] + 0.01, y_values[i] + 0.01))
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.xlabel("IOU between any two summary row's indexes [1]")
        plt.ylabel("IOU between any two summary column's indexes [1]")
        plt.savefig(save_path)
        plt.close()


if __name__ == '__main__':
    AnalysisConvergeProcess.iou_greedy_converge(rows_list=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 69, 70, 71, 72, 73, 74],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 57, 58, 59, 97, 98, 99],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 69, 70, 71, 72, 73, 74],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 69, 70, 71, 72, 73, 74],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 69, 70, 71, 72, 73, 74]
    ],
    col_list=[
        [0, 1, 3, 5, 6, 13, 14, 18, 26, 27, 31, 32, 39, 40, 52, 53, 64],
        [0, 1, 2, 3, 4, 26, 27, 29, 30, 32, 34, 41, 42, 117, 121, 130, 131],
        [0, 27, 29, 30, 55, 72, 78, 86, 101, 109, 111, 113, 114, 121, 141, 142, 150],
        [0, 1, 2, 3, 4, 26, 27, 29, 30, 32, 34, 41, 42, 117, 121, 130, 131],
        [0, 1, 2, 3, 4, 26, 27, 29, 30, 32, 34, 41, 42, 117, 121, 130, 131]
    ],
    save_path=os.path.join(os.path.dirname(__file__), "results", "greedy_converge_report.png"))