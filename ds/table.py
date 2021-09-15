# library imports
import pandas as pd


class Table:
    """
    A simple table class, allowing to populate pandas' dataframe object in an easy way for this programmer
    """

    def __init__(self,
                 columns: list,
                 rows_ids: list):
        self.columns = columns
        self.rows_ids = rows_ids
        self.data = [[0
                      for col_index in range(len(columns))]
                     for row_index in range(len(rows_ids))]

    def add(self,
            column: str,
            row_id: str,
            data_point) -> None:
        """
        Add a data point to the right place in the table
        :param column: the column name
        :param row_id: the row index name
        :param data_point: the value we want to add
        :return: None
        """
        self.data[self.rows_ids.index(row_id)][self.columns.index(column)] = data_point

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert the 'Table' instance to pd.DataFrame instance
        """
        return pd.DataFrame(data=self.data, columns=self.columns, index=self.rows_ids)

    def to_csv(self,
               save_path: str):
        """
        Convert the 'Table' instance to CSV file and save it
        """
        self.to_dataframe().to_csv(save_path)

    def __repr__(self):
        return "<Table: {}X{}>".format(len(self.rows_ids), len(self.columns))

    def __str__(self):
        return "Table: {}X{}\n--------------\n{}".format(len(self.rows_ids), len(self.columns), self.data)
