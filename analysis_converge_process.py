# library imports
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# project imports
from movie_from_images_maker import MovieFromImages


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
                            cols_list: list,
                            save_path: str):
        """
        Plot a scatter plot of the changes in the summary rows and cols with the IOU metric over the process
        :param rows_list: list of list of rows indexes from the summary algorithm
        :param cols_list: list of list of rows indexes from the summary algorithm
        :param save_path: a path to save the plot in
        :return: save plot
        """
        x_values = [AnalysisConvergeProcess._iou(set_a=set(rows_list[row_index]),
                                                 set_b=set(rows_list[row_index+1]))
                    for row_index in range(len(rows_list)-1)]
        y_values = [AnalysisConvergeProcess._iou(set_a=set(cols_list[col_index]),
                                                 set_b=set(cols_list[col_index+1]))
                    for col_index in range(len(cols_list)-1)]
        plt.scatter(x_values,
                    y_values,
                    s=20,
                    marker='o',
                    c="black")
        for i, txt in enumerate(list(range(len(x_values)))):
            plt.annotate("{}->{}".format(i+1, i+2), (x_values[i] + 0.01, y_values[i] + 0.01))
        plt.xlim((-0.05, 1.05))
        plt.ylim((-0.05, 1.05))
        plt.xticks([i/10 for i in range(11)], [i/10 for i in range(11)])
        plt.yticks([i/10 for i in range(11)], [i/10 for i in range(11)])
        plt.grid(alpha=0.3)
        plt.xlabel("IOU between any two summary row's indexes [1]")
        plt.ylabel("IOU between any two summary column's indexes [1]")
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def picking_summary_video(rows_list: list,
                              cols_list: list,
                              original_data_set_shape: tuple,
                              save_path_folder: str,
                              fps: int = 1):
        """
        Plot a video of the steps
        :param rows_list: list of list of rows indexes from the summary algorithm
        :param cols_list: list of list of rows indexes from the summary algorithm
        :param original_data_set_shape: the original data of the
        :param save_path_folder: a path to save the plot in
        :param fps: frames per second in the video
        :return: save plot
        """
        # if needed, make a dedicated folder
        try:
            os.makedirs(save_path_folder)
        except:
            pass

        # for each step, make an image
        for step in range(len(rows_list)):
            base = np.zeros(original_data_set_shape)
            for row_index in rows_list[step]:
                base[row_index, :] = [1 for i in range(original_data_set_shape[1])]
            for col_index in cols_list[step]:
                base[:, col_index] = [1 for i in range(original_data_set_shape[0])]
            ax = sns.heatmap(base,
                             cbar=False,
                             cmap='binary',
                             vmax=1,
                             vmin=0)
            plt.xticks()
            plt.yticks()
            plt.xlabel("")
            plt.ylabel("")
            plt.savefig(os.path.join(save_path_folder, "image_{}.png".format(step+1)))
            plt.close()
        # make video from the photos
        MovieFromImages.create(source_folder=save_path_folder,
                               output_path=os.path.join(save_path_folder, "movie.avi"),
                               fps=fps)