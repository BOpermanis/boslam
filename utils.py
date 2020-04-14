import numpy as np
import pandas as pd
import time
from config import data_dir
import cv2


def match_indices(inds1, inds2):
    yinds = []
    for i1, a in enumerate(inds1):
        for i2, b in enumerate(inds2):
            if a == b:
                yinds.append((i1, i2))
    return yinds


def id2color(i):
    np.random.seed(i)
    # return tuple(np.random.randint(0, 256, (3,), dtype=np.uint8))
    return tuple(map(int, np.random.randint(0, 256, (3,))))


class LOG:
    def __init__(self, logname):
        self.logname = logname
        self.list_lines = []
        self.save_path = "{}/{}.csv".format(data_dir, self.logname)  # .encode('utf-8',errors = 'strict')

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


class Visualizer:
    def __init__(self):
        self.pts = None
        self.identities = None

    def _visualize_pts(self, img, inds, pts):
        for i, (x, y) in zip(inds, pts):
            cv2.circle(img, (x, y), 3, id2color(i), thickness=5)

    def initialize(self, frame_ob):
        self.pts = frame_ob.kp_arr
        self.identities = np.arange(self.pts.shape[0], dtype=int)
        self._visualize_pts(frame_ob.rgb_frame, self.identities, self.pts)
        return frame_ob.rgb_frame

    def show(self, frame_ob, inds_prev, inds):
        # print("np.average(inds_prev == inds)", np.average(inds_prev == inds))
        # inds_prev, inds = np.arange(len(self.identities), dtype=int), np.arange(len(self.identities), dtype=int)
        # print((np.max(self.identities[inds_prev]) - np.min(self.identities[inds_prev])) / len(inds_prev))
        self.pts = frame_ob.kp_arr
        new_identities = -np.ones((self.pts.shape[0],), dtype=int)
        new_identities[inds] = self.identities[inds_prev]
        num_new_pts = self.pts.shape[0] - len(inds_prev)
        first_new_id = np.max(new_identities) + 1
        new_identities[new_identities == -1] = np.arange(first_new_id, first_new_id + num_new_pts)

        self.identities = new_identities
        self._visualize_pts(frame_ob.rgb_frame, self.identities, self.pts)
        return frame_ob.rgb_frame
