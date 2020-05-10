import unittest
from slam.covisibility_graph import CovisibilityGraph
from slam.nodes import MapPoint, KeyFrame
from slam.bow_db import Dbow
from camera import Frame
import config
import numpy as np
from copy import deepcopy
from pprint import pprint
import pickle
from itertools import combinations
from utils import key_common_mps


class CovisibilityGraphMethods(unittest.TestCase):

    def test_add_kf(self):
        with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
            frames = pickle.load(conn)

        MapPoint.cnt = 0
        KeyFrame.cnt = 0

        dbow = Dbow()

        cg = CovisibilityGraph(dbow, config)
        frame = frames[0]
        frame1 = frames[1]
        R, t = np.eye(3), np.zeros(3)
        frame.setPose(R, t)
        kf0 = cg.add_kf(frame)

        self.assertEqual(len(cg.kfs), 1)
        self.assertEqual(len(cg.mps), frame.kp_arr.shape[0])
        self.assertEqual(len(cg.edges_kf2mps[frame.id]), frame.kp_arr.shape[0])

        cg.add_kf(frame1)
        self.assertEqual(len(cg.kfs), 2)
        self.assertTrue(frame.kp_arr.shape[0] <= len(cg.mps) <= frame.kp_arr.shape[0] + frame1.kp_arr.shape[0])
        self.assertEqual(len(cg.kf2kf_num_common_mps), 1)
        self.assertTrue(cg.edges_kf2kfs[frame.id] == {*[frame1.id]})
        self.assertTrue(cg.edges_kf2kfs[frame1.id] == {*[frame.id]})

        n_common1 = list(cg.kf2kf_num_common_mps.values())[0]
        n_common2 = len(cg.edges_kf2mps[frame.id].intersection(cg.edges_kf2mps[frame1.id]))
        self.assertEqual(n_common1, n_common2)

    def test_get_local_map(self):
        with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
            frames = pickle.load(conn)
        dbow = Dbow()
        MapPoint.cnt = 0
        KeyFrame.cnt = 0
        cg = CovisibilityGraph(dbow, config)

        # R, t = np.eye(3), np.zeros(3)
        # frames[0].setPose(R, t)

        kf0 = cg.add_kf(frames[0])
        kf1 = cg.add_kf(frames[1])
        kf2 = cg.add_kf(frames[2])
        kf3 = cg.add_kf(frames[3])

        self.assertTrue(cg.get_local_map(kf0) == {*[kf0.id, kf1.id, kf2.id, kf3.id]})
        self.assertTrue(cg.get_local_map(kf0, flag_with_input_kf=False) == {*[kf1.id, kf2.id, kf3.id]})
        # self.assertTrue(cg.get_local_map(kf0) == {*[kf0.id, kf1.id]})
        for i1, i2 in combinations(cg.kfs.keys(), 2):
            n_common1 = len(cg.edges_kf2mps[i1].intersection(cg.edges_kf2mps[i2]))
            n_common2 = cg.kf2kf_num_common_mps[key_common_mps(i1, i2)]
            self.assertTrue(n_common2, n_common1)

    def test_erase_kf(self):
        with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
            frames = pickle.load(conn)
        dbow = Dbow()
        MapPoint.cnt = 0
        KeyFrame.cnt = 0
        cg = CovisibilityGraph(dbow, config)

        kf0 = cg.add_kf(frames[0])
        kf1 = cg.add_kf(frames[1])
        kf2 = cg.add_kf(frames[2])
        kf3 = cg.add_kf(frames[3])

        cg.erase_kf(kf0)
        cg.erase_kf(kf2)
        self.assertTrue(len(cg.kfs), 2)
        for i1, i2 in combinations(cg.kfs.keys(), 2):
            n_common1 = len(cg.edges_kf2mps[i1].intersection(cg.edges_kf2mps[i2]))
            n_common2 = cg.kf2kf_num_common_mps[key_common_mps(i1, i2)]
            self.assertTrue(n_common2, n_common1)

    def test_erase_mp(self):
        with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
            frames = pickle.load(conn)
        dbow = Dbow()
        MapPoint.cnt = 0
        KeyFrame.cnt = 0
        cg = CovisibilityGraph(dbow, config)

        kf0 = cg.add_kf(frames[0])
        kf1 = cg.add_kf(frames[1])
        kf2 = cg.add_kf(frames[2])
        kf3 = cg.add_kf(frames[3])

        id_mp0 = next(iter(cg.edges_kf2mps[kf0.id]))
        id_mp3 = next(iter(cg.edges_kf2mps[kf3.id]))
        if id_mp0 == id_mp3:
            print(11111)
        else:
            n_mps = len(cg.mps)
            n_common = cg.kf2kf_num_common_mps[key_common_mps(kf0.id, kf3.id)]
            cg.erase_mp(id_mp0)
            cg.erase_mp(id_mp3)
            n_common_after = cg.kf2kf_num_common_mps[key_common_mps(kf0.id, kf3.id)]
            self.assertTrue(n_mps - 2, len(cg.mps))
            self.assertTrue(n_mps - 2, len(cg.edges_mp2kfs))
            self.assertTrue(n_common - 2, n_common_after)

            for i in cg.kfs:
                self.assertFalse(id_mp0 in cg.edges_kf2mps[i])
                self.assertFalse(id_mp3 in cg.edges_kf2mps[i])

if __name__ == '__main__':
    # CovisibilityGraphMethods().test_get_local_map()
    # CovisibilityGraphMethods().test_get_local_map()
    # CovisibilityGraphMethods().test_add_kf()
    unittest.main()
