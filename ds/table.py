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

    def add_row(self,
                row_id: str,
                data) -> None:
        """
        Add a row of data to the right place in the table
        :param row_id: the row index name
        :param data: the values we want to add to this row
        :return: None
        """
        # make sure the length is legit
        if len(data) != len(self.columns):
            raise Exception("Error is Table.add_row: the number of values and columns do not match")

        if isinstance(data, list):
            [self.add(column=col_name,
                      row_id=row_id,
                      data_point=data[index]) for index, col_name in enumerate(self.columns)]
        elif isinstance(data, dict):
            [self.add(column=key,
                      row_id=row_id,
                      data_point=val) for key, val in data.items()]
        else:
            raise Exception("Error is Table.add_row: data type must be list of dict")

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
