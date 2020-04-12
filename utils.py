# import numpy as np
import pandas as pd
import time
from config import data_dir


class LOG:
    def __init__(self, logname):
        self.logname = logname
        self.list_lines = []
        self.save_path = "{}/{}.csv".format(data_dir, self.logname)#.encode('utf-8',errors = 'strict')

    def add(self, log):
        self.list_lines.append(log)

    def save(self):
        set_cols = set()
        for ks in self.list_lines:
            set_cols.update(ks.keys())

        tab = {k: [] for k in set_cols}
        for log in self.list_lines:
            for k in tab:
                if k in log:
                    tab[k].append(log[k])
                else:
                    tab[k].append("")
        pd.DataFrame(tab).to_csv(self.save_path)
