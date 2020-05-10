import unittest
from slam.covisibility_graph import CovisibilityGraph
from slam.nodes import MapPoint, KeyFrame
from slam.bow_db import Dbow
from slam.tracking import Tracker
from slam.local_mapping import LocalMapManager
from camera import Frame
import config
import numpy as np
from copy import deepcopy
from pprint import pprint
import pickle
from itertools import combinations
from utils import key_common_mps
from queue import Queue


class LocalMapManagerMethods(unittest.TestCase):

    def test_tracking(self):
        with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
            frames = pickle.load(conn)

        MapPoint.cnt = 0
        KeyFrame.cnt = 0
        dbow = Dbow()
        cg = CovisibilityGraph(dbow, config)
        tracker = Tracker(cg, dbow, config)
        local_map_manager = LocalMapManager(cg, dbow, config)
        kf_queue = Queue()
        bow_queue = Queue()

        for frame in frames:
            tracker.update(frame, kf_queue)

        for _ in range(100):
            local_map_manager.update(kf_queue, bow_queue)


if __name__ == "__main__":
    unittest.main()