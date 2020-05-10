import unittest
from slam.covisibility_graph import CovisibilityGraph
from slam.nodes import MapPoint, KeyFrame
from slam.bow_db import Dbow
from slam.tracking import Tracker
from camera import Frame
import config
import numpy as np
from copy import deepcopy
from pprint import pprint
import pickle
from itertools import combinations
from utils import key_common_mps
from queue import Queue


# class TrackerMethods(unittest.TestCase):
class TrackerMethods(unittest.TestCase):

    def test_tracking(self):
        with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
            frames = pickle.load(conn)

        MapPoint.cnt = 0
        KeyFrame.cnt = 0
        dbow = Dbow()
        cg = CovisibilityGraph(dbow, config)
        tracker = Tracker(cg, dbow, config)
        kf_queue = Queue()

        for frame in frames:
            tracker.update(frame, kf_queue)

        all_kf_ids = set(cg.kfs.keys())
        for i1, i2 in combinations(all_kf_ids, 2):
            self.assertTrue(cg.kf2kf_num_common_mps[key_common_mps(i1, i2)] > 0)

        self.assertTrue(min(cg.kf2kf_num_common_mps.values()) >= 15)

        for i in all_kf_ids:
            self.assertTrue(cg.edges_kf2kfs[i] == all_kf_ids.difference({i}))
            self.assertTrue(cg.get_local_map(i) == all_kf_ids)
            self.assertTrue(cg.get_local_map(i, flag_with_input_kf=False) == all_kf_ids.difference({i}))

if __name__ == "__main__":
    unittest.main()