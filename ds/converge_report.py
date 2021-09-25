class ConvergeReport:
    """
    A class responsible to store and provide analysis on a summary's algorithm converge process
    """

    def __init__(self,
                 rows: list = None,
                 cols: list = None,
                 rows_score: list = None,
                 cols_score: list = None,
                 rows_calc_time: list = None,
                 cols_calc_time: list = None,
                 total_score: list = None):
        self.rows = rows if isinstance(rows, list) else []
        self.cols = cols if isinstance(cols, list) else []
        self.rows_score = rows_score if isinstance(rows_score, list) else []
        self.cols_score = cols_score if isinstance(cols_score, list) else []
        self.rows_calc_time = rows_calc_time if isinstance(rows_calc_time, list) else []
        self.cols_calc_time = cols_calc_time if isinstance(cols_calc_time, list) else []
        self.total_score = total_score if isinstance(total_score, list) else []

    # data modification function #

    def clear(self):
        self.rows = []
        self.cols = []
        self.rows_score = []
        self.cols_score = []
        self.rows_calc_time = []
        self.cols_calc_time = []
        self.total_score = []

    def add_step(self,
                 row: list,
                 col: list,
                 row_score: float,
                 col_score: float,
                 row_calc_time: float,
                 col_calc_time: float,
                 total_score: float):
        self.rows.append(row)
        self.cols.append(col)
        self.rows_score.append(row_score)
        self.cols_score.append(col_score)
        self.rows_calc_time.append(row_calc_time)
        self.cols_calc_time.append(col_calc_time)
        self.total_score.append(total_score)

    # end - data modification function #

    # simple get functions #

    def to_dict(self):
        return {
            "_rows": self.rows,
            "_cols": self.cols,
            "rows_score": self.rows_score,
            "cols_score": self.cols_score,
            "rows_calc_time": self.rows_calc_time,
            "cols_calc_time": self.cols_calc_time,
            "total_score": self.total_score
        }

    def total_time(self,
                   index: int):
        return self.rows_calc_time[index] + self.cols_calc_time[index]

    def step_get(self,
                 key: str,
                 index: int):
        return self.__getitem__(key=key)[index]

    def __getitem__(self,
                    key: str):
        if key == "_rows":
            return self.rows
        elif key == "_cols":
            return self.cols
        elif key == "rows_score":
            return self.rows_score
        elif key == "cols_score":
            return self.cols_score
        elif key == "rows_calc_time":
            return self.rows_calc_time
        elif key == "cols_calc_time":
            return self.cols_calc_time
        elif key == "total_score":
            return self.total_score

    # end - simple get functions #

    # smart get functions #

    def final_total_score(self):
        return self.total_score[-1]

    def steps_count(self):
        return len(self.rows)

    def compute_time(self):
        return sum(self.cols_calc_time) + sum(self.rows_calc_time)

    # end - smart get functions #

    def __repr__(self):
        return "<ConvergeReport: {} steps>".format(len(self.rows))

    def __str__(self):
        return self.to_dict().__str__()
