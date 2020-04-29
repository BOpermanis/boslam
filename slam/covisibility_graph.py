import numpy as np
import cv2
import g2o
from typing import Dict, Set, Tuple, List

from multiprocessing import Lock
from config import d_hamming_max
from camera import Frame

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
        int: self.width = camera.width
        int: self.height = camera.height
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        # data and counters
        self.kfs = Dict[int, KeyFrame]
        self.mps = Dict[int, MapPoint]
        self.edges_kf2kfs = Dict[int, List[Tuple[int, int]]] # list of (#common mps, id kf)
        self.edges_kf2mps = Dict[int, List[int]]

        int: self.num_kfs = 0
        int: self.num_mps = 0

        # localmap
        self.local_map = Set[int]
        self.kf_ref = None

        # TODO locks

    def get_loop_candidates(self):
        pass

    def get_reloction_candidates(self):
        pass

    # def add_kf(self, kf: KeyFrame):
    #     self.dict_kfs[self.num_kfs] = kf
    #     self.num_kfs += 1

    def erase_kf(self):
        pass

    # def add_mp(self, mp: MapPoint):
    #     self.dict_mps[self.num_mps] = mp
    #     self.num_mps += 1

    def erase_mp(self):
        pass

    def maintain(self):
        pass

