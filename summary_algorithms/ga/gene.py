# library imports
from random import random


class SummaryGene:
    """
    A data class for summary gene
    """

    def __init__(self,
                 rows: list = None,
                 cols: list = None):
        self._rows = rows if isinstance(rows, list) else []
        self._cols = cols if isinstance(cols, list) else []

    # getters #

    def get_summary(self,
                    dataset):
        return dataset.iloc[self._rows, self._cols]

    def get_rows(self):
        return self._rows

    def get_columns(self):
        return self._cols

    def get_row_count(self):
        return len(self._rows)

    def get_col_count(self):
        return len(self._cols)

    # end - getters #

    # logic #

    def mutation(self,
                 max_row_index: int,
                 max_col_index: int,
                 mutation_rate: float):
        mutation_count = round((len(self._rows) + len(self._cols)) * mutation_rate)
        for mutations_index in range(mutation_count):
            # pick row or column
            is_row = random()
            # if we have all columns, we can only change rows
            if len(self._cols) == max_col_index + 1:
                is_row = 0
            # if row
            if is_row < 0.5:
                # pick index we want to change
                change_index = round(random() * (len(self._rows)-1))
                # pick new value
                new_val = round(random() * max_row_index)
                while new_val in self._rows:
                    new_val = round(random() * max_row_index)
                self._rows[change_index] = new_val
            else:  # else column
                # pick index we want to change
                change_index = round(random() * (len(self._cols)-1))
                # pick new value
                new_val = round(random() * max_col_index)
                while new_val in self._cols:
                    new_val = round(random() * max_col_index)
                self._cols[change_index] = new_val

    # end - logic #

    def __repr__(self):
        return "<SummaryGene ({}X{})>".format(len(self._rows), len(self._cols))

    def __str__(self):
        return "SummaryGene:\nrows = {}\ncols = {}>".format(self._rows, self._cols)
