# import unittest
from slam.covisibility_graph import CovisibilityGraph
from slam.nodes import MapPoint, KeyFrame
from slam.bow_db import Dbow
from camera import Frame
import config
import numpy as np
from copy import deepcopy
from pprint import pprint
import pickle

# with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
#     frames = pickle.load(conn)
# dbow = Dbow()


# class CovisibilityGraphMethods(unittest.TestCase):
class CovisibilityGraphMethods:

    def test_add_kf(self):
        return
        with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
            frames = pickle.load(conn)
        # print(np.sum(frames[0].rgb_frame))
        # print(np.sum(frames[1].rgb_frame))
        # exit()

        dbow = Dbow()

        cg = CovisibilityGraph(dbow, config)
        frame = frames[0]
        # frame1 = deepcopy(frame)
        # frame1.id += 1
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
        # print(n_common1, n_common2)
        self.assertEqual(n_common1, n_common2)
        # print(cg.get_local_map(kf0))
        del cg

    def test_get_local_map(self):
        with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
            frames = pickle.load(conn)
        dbow = Dbow()

        cg = CovisibilityGraph(dbow, config)

        print("len(cg.mps)", len(cg.mps))
        R, t = np.eye(3), np.zeros(3)
        frames[0].setPose(R, t)

        kf0 = cg.add_kf(frames[0])
        kf1 = cg.add_kf(frames[1])
        kf2 = cg.add_kf(frames[2])
        kf3 = cg.add_kf(frames[3])

        # print(cg.get_local_map(kf0))
        # print(cg.edges_kf2mps)
        print("cg.kfs.keys()", cg.kfs.keys())
        print(cg.edges_kf2kfs)
        # pprint(cg.kf2kf_num_common_mps)
        # self.assertTrue(cg.get_local_map(kf0) == {*[kf0.id, kf1.id, kf2.id]})
        # self.assertTrue(cg.get_local_map(kf0) == {*[kf0.id, kf1.id]})

if __name__ == '__main__':
    CovisibilityGraphMethods().test_get_local_map()
    CovisibilityGraphMethods().test_get_local_map()
    # CovisibilityGraphMethods().test_add_kf()
    # unittest.main()
