import numpy as np
import cv2
from typing import Dict, Set, Tuple, List
import g2o
from multiprocessing import Queue

from slam.covisibility_graph import CovisibilityGraph
from slam.nodes import KeyFrame
from config import d_hamming_max
from camera import Frame


class LocalMapManager:
    def __init__(self, cg: CovisibilityGraph, camera):
        self.cg = cg
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        self.width = camera.width
        self.height = camera.height
        self.cam = g2o.CameraParameters(camera.f, camera.principal_point, camera.baseline)
        self.cam_mat, self.dist_coefs = camera.cam_mat, camera.distCoeffs

    def _add_new_kf(self, kf: KeyFrame):
        # TODO adds to covisibility graph?
        pass

    def _recent_mps_culling(self):
        # TODO cull recent mappoints
        pass

    def _new_points_creation(self):
        # TODO checks all new mappoint correspondences, if mapppoint survives all criterion, then it is added to covisibility graph, ready for BA
        pass

    def _local_ba(self):
        # TODO runs BA on local map
        pass

    def _local_kf_culling(self):
        # TODO kf culling
        pass

    def update(self, kf_queue: Queue, bow_queue: Queue):
        """
        :param kf_queue: input frames processed to be included into localmapping
        sidefects:
        mutates covisibility graph structure
        adds bow of the last processed kf to a queue for loop closing
        """

        kf = kf_queue.get()

        self._add_new_kf(kf)

        bow_queue.put(kf)

        self._recent_mps_culling()

        self._new_points_creation()

        self._local_ba()

        self._local_kf_culling()
