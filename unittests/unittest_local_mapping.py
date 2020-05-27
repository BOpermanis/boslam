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

#
# class LocalMapManagerMethods(unittest.TestCase):
#
#     def test_tracking(self):
#         with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
#             frames = pickle.load(conn)
#
#         MapPoint.cnt = 0
#         KeyFrame.cnt = 0
#         dbow = Dbow()
#         cg = CovisibilityGraph(dbow, config)
#         tracker = Tracker(cg, dbow, config)
#         local_map_manager = LocalMapManager(cg, dbow, config)
#         kf_queue = Queue()
#         bow_queue = Queue()
#
#         kf_ids = set()
#
#         for frame in frames: #[:20]:
#             tracker.update(frame, kf_queue)
#             kf_ids.update(cg.kfs)
#
#             while kf_queue.qsize() > 0:
#                 local_map_manager.update(kf_queue, bow_queue)
#         self.assertTrue(len(cg.kfs) < 5)
#         n_obs = [k.n_obs for k in cg.mps.values()]
#         l = [len(cg.edges_mp2kfs[k]) for k in cg.edges_mp2kfs]
#         print("np.min(n_obs), np.max(n_obs)", np.min(n_obs), np.max(n_obs))
#         print(np.min(l), np.max(l))
#         print(len(kf_ids), MapPoint.cnt)
#         print("len(cg.kfs), len(cg.mps)", len(cg.kfs), len(cg.mps))
#         for kf in cg.kfs.values():
#             self.assertTrue(np.linalg.norm(kf.R - np.eye(3)) < 0.01)
#             self.assertTrue(np.linalg.norm(kf.t - np.zeros((3,))) < 0.01)


if __name__ == "__main__":
    # unittest.main()
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

    kf_ids = set()

    for frame in frames:  # [:20]:
        frame.kp_arr += np.random.normal(0, 1, size=frame.kp_arr.shape).astype(np.int64)
        # frame.cloud_kp += np.random.normal(0, 0, size=frame.cloud_kp.shape)
        print("--------------------------")
        tracker.update(frame, kf_queue)
        kf_ids.update(cg.kfs)

        prev_vals = {}
        for i, kf_id in enumerate(sorted(list(cg.kfs.keys()))):
            kf = cg.kfs[kf_id]
            # print("{} ) {}, {}".format(i, np.linalg.norm(kf.R - np.eye(3)), np.linalg.norm(kf.t - np.zeros((3,)))))
            prev_vals[i] = kf.R, kf.t
            # assert np.linalg.norm(kf.R - np.eye(3)) < 0.01
            # assert np.linalg.norm(kf.t - np.zeros((3,))) < 0.02

        for mp in cg.mps.values():
            mp.pt3df()

        while kf_queue.qsize() > 0:
            local_map_manager.update(kf_queue, bow_queue)

        diffs = []
        for i, kf_id in enumerate(sorted(list(cg.kfs.keys()))):
            kf = cg.kfs[kf_id]
            diffs.append(np.linalg.norm(kf.R - prev_vals[i][0]))
            diffs.append(np.linalg.norm(kf.t - prev_vals[i][1]))
            diffs.append(np.linalg.norm(kf.t - np.zeros((3,))))
            # print("{} ) {}, {}, {}".format(i, np.linalg.norm(kf.R - np.eye(3)), np.linalg.norm(kf.t - np.zeros((3,))), np.linalg.norm(kf.R - prev_vals[i])))
            # assert np.linalg.norm(kf.R - np.eye(3)) < 0.01
            # assert np.linalg.norm(kf.t - np.zeros((3,))) < 0.01
        print(np.max(diffs))

    exit()
    print("len(cg.kfs) < 5", len(cg.kfs) < 5)
    n_obs = [k.n_obs for k in cg.mps.values()]
    l = [len(cg.edges_mp2kfs[k]) for k in cg.edges_mp2kfs]
    print("np.min(n_obs), np.max(n_obs)", np.min(n_obs), np.max(n_obs))
    print(np.min(l), np.max(l))
    print(len(kf_ids), MapPoint.cnt)
    print("len(cg.kfs), len(cg.mps)", len(cg.kfs), len(cg.mps))
    for kf in cg.kfs.values():
        assert np.linalg.norm(kf.R - np.eye(3)) < 0.01
        assert np.linalg.norm(kf.t - np.zeros((3,))) < 0.01
