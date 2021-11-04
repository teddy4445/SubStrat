# library imports
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# project imports


class StableUnstableSummaryHeatmap:
    """
    Generate a single plot of heatmap for the stable and unstable stability compare heatmap
    """

    def __init__(self):
        pass

    @staticmethod
    def generate(json_file_path: str):
        """

        """
        # load data
        with open(json_file_path, "r") as data_json_file:
            data = json.load(data_json_file)
        # create the data
        df_data = []
        columns = []
        datasets = []
        for dataset_key in data.keys():
            datasets.append(dataset_key)
            df_data_row = []
            for metric in data[dataset_key]:
                if metric not in columns:
                    columns.append(metric)
                df_data_row.append(abs(max(data[dataset_key][metric]["y_stable"]) - max(data[dataset_key][metric]["y_unstable"])))
            df_data.append(df_data_row)

        # convert to pandas

        df = pd.DataFrame(data=df_data,
                          index=datasets,
                          columns=list(columns))
        df.drop(['entropy'], axis=1, inplace=True)
        sns.heatmap(data=df,
                    vmin=0,
                    cmap="coolwarm",
                    annot=True,
                    fmt=".5f",
                    linewidths=1,
                    linecolor="white",
                    square=False)
        plt.tight_layout()
        plt.savefig("Summary_stable_unstable_compare_heatmap.png")
        plt.close()


if __name__ == '__main__':
    StableUnstableSummaryHeatmap.generate(json_file_path=r"C:\Users\lazeb\Desktop\stability_feature_selection_dataset_summary\experiments\stability_important\dist\overall_answer.json")
