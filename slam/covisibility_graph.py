import numpy as np
import cv2
import g2o
from typing import Dict, Set, Tuple, List
from collections import defaultdict
from itertools import combinations

from multiprocessing import Lock
from config import d_hamming_max
from camera import Frame
from utils import key_common_mps

from slam.nodes import KeyFrame, MapPoint
cos60 = np.cos(np.pi / 3)
dmin = 0.1
dmax = 20


class CovisibilityGraph(g2o.SparseOptimizer):
    def __init__(self, camera):

        # g2o related
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        self.cam = g2o.CameraParameters(camera.f, camera.principal_point, camera.baseline)
        self.cam.set_id(0)
        super().add_parameter(self.cam)

        # params
        self.width = camera.width
        self.height = camera.height
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        # data and counters
        self.kfs = Dict[int, KeyFrame]
        self.mps = Dict[int, MapPoint]
        self.kf2kf_num_common_mps = defaultdict(0)
        self.edges_kf2kfs = defaultdict(set) #Dict[int, List[int, int]] # list of (#common mps, id kf)
        self.edges_kf2mps = defaultdict(set) #Dict[int, List[int]]
        self.edges_mp2kfs = defaultdict(set) #Dict[int, List[int]]

    def get_local_map(self, kf):
        if isinstance(kf, int):
            kf = self.kfs[kf]
        set_ids = {kf.id}
        for id_kf1 in self.edges_kf2kfs[kf.id]:
            if self.kf2kf_num_common_mps[key_common_mps(id_kf1, kf.id)] >= 15:
                set_ids.add(id_kf1)
                for id_kf2 in self.edges_kf2kfs[id_kf1]:
                    if self.kf2kf_num_common_mps[key_common_mps(id_kf1, kf.id)] >= 15:
                        set_ids.add(id_kf2)
        return set_ids

    def add_edge_to_cg(self, kf, mp, num_common=None):
        if isinstance(mp, KeyFrame):
            assert kf.id in self.kfs and mp.id in self.kfs
            if num_common is None:
                num_common = len(self.edges_kf2mps[kf.id].intersection(self.edges_kf2mps[mp.id]))
            self.edges_kf2kfs[kf.id].add(mp.id)
            self.edges_kf2kfs[mp.id].add(kf.id)
            self.kf2kf_num_common_mps[key_common_mps(kf.id, mp.id)] = num_common
        else:
            assert kf.id in self.kfs and mp.id in self.mps
            self.edges_kf2mps[kf.id].add(mp.id)
            self.edges_mp2kfs[mp.id].add(kf.id)

    def get_loop_candidates(self):
        pass

    def add_kf(self, frame: Frame):
        self.dbow.add(frame)
        kf = KeyFrame(frame, self.dbow, self)
        self.dict_kfs[frame.id] = kf

        for i_feat, (feat, d2, d3, id_mp) in enumerate(zip(kf.des, kf.kp, kf.cloud_kp, kf.des2mp)):
            n = frame.t - d3
            norm = np.linalg.norm(n)
            if norm > 0.0:
                if id_mp == -1 or id_mp not in self.mps:
                    mp = MapPoint(feat, d3, n / norm, kf.id, self.dbow, self)
                    self.mps[mp.id] = mp
                    self.kfs[frame.id].des2mp[i_feat] = mp.id
                    id_mp = mp.id
                else:
                    self.mps[id_mp].add_observation(feat, n, kf.id)
                self.add_edge_to_cg(kf, self.cg.mps[id_mp])

    def erase_kf(self, kf):
        for id_mp in self.edges_kf2mps[kf.id]:
            self.edges_mp2kfs[id_mp].remove(kf.id)

        for id_kf in self.edges_kf2kfs[kf.id]:
            del self.kf2kf_num_common_mps[key_common_mps(kf.id, id_kf)]
            self.edges_kf2kfs[id_kf].remove(kf.id)

        del self.edges_kf2kfs[kf.id]
        del self.edges_kf2mps[kf.id]
        del self.kfs[kf.id]

    def erase_mp(self, mp):
        id_kfs = []
        for id_kf in self.edges_mp2kfs[mp.id]:
            self.edges_kf2mps[id_kf].remove(mp.id)
            id_kfs.append(id_kf)

        for i1, i2 in combinations(id_kfs, 2):
            self.kf2kf_num_common_mps[key_common_mps(i2, i1)] -= 1

        del self.mps[mp.id]
        del self.edges_mp2kfs[mp.id]

