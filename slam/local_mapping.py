import numpy as np
import cv2
from typing import Dict, Set, Tuple, List
import g2o
from multiprocessing import Queue
from itertools import combinations
from collections import defaultdict

from slam.covisibility_graph import CovisibilityGraph
from slam.nodes import KeyFrame
from config import d_hamming_max
from camera import Frame
from slam.bow_db import Dbow
from utils import key_common_mps


class LocalMapManager:
    def __init__(self, cg: CovisibilityGraph, dbow: Dbow, camera):
        self.cg = cg
        self.dbow = dbow
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        self.width = camera.width
        self.height = camera.height
        self.cam = g2o.CameraParameters(camera.f, camera.principal_point, camera.baseline)
        self.cam_mat, self.dist_coefs = camera.cam_mat, camera.distCoeffs

    def _insert_new_kf(self, id_kf):
        # TODO adds to covisibility graph? (already done in tracking thread ?)
        return True

    def _recent_mps_culling(self, id_kf):
        mps_to_delete = []
        with self.cg.lock_mps:
            for mp in self.cg.mps.values():
                if not (mp.get_found_ratio() >= 0.25 and (mp.n_obs >= 3 or mp.first_kf == id_kf)):
                    mps_to_delete.append(mp)
        for mp in mps_to_delete:
            self.cg.erase_mp(mp)

    def _new_points_creation(self):
        # TODO checks all new mappoint correspondences, if mapppoint survives all criterion, then it is added to covisibility graph, ready for BA
        # this is done in tracking thread by this implementation
        pass

    def _local_ba(self):
        # TODO runs BA on local map
        pass

    def _local_kf_culling(self, id_kf):
        # !!! orbslam2 kodā ir cikls cauri kfs, katrai fičai iet cauri un skatās vai tas ir redundnt observation,
        # tad saskaita cik tādu daudz ir un skatās vai prop > 0.9, ja ir tad vnk izmet

        voter = defaultdict(list)
        with self.cg.lock_edges_kf2mps and self.cg.lock_kf2kf_num_common_mps:
            ids = self.cg.get_local_map(id_kf, flag_with_input_kf=False)

            for i1, i2 in combinations(ids, 2):
                num_common = self.cg.kf2kf_num_common_mps[key_common_mps(i2, i1)]
                p1 = len(self.cg.edges_kf2mps[i1]) / num_common
                p2 = len(self.cg.edges_kf2mps[i2]) / num_common
                if p1 > 0.9:
                    voter[i1].append(i2)
                if p2 > 0.9:
                    voter[i2].append(i1)

        triples = sorted([(id_kf, l, len(l)) for id_kf, l in voter if len(l) > 2], key=lambda x: -x[1])
        set_close = {*[_[0] for _ in triples]}
        kfs_to_cull = {*[]}

        with self.cg.lock_edges_kf2mps:
            while len(triples) > 0:
                id_kf, l, n = triples.pop(0)
                if id_kf in kfs_to_cull:
                    continue
                for id_kf1 in l:
                    if id_kf1 in set_close:
                        #  tests kuru izmest + izmeshana no set_close
                        if len(voter[id_kf]) > len(voter[id_kf1]):
                            kfs_to_cull.add(id_kf)
                            set_close.remove(id_kf)
                            break
                        elif len(voter[id_kf]) < len(voter[id_kf1]):
                            kfs_to_cull.add(id_kf1)
                            set_close.remove(id_kf1)
                        else:
                            if len(self.cg.edges_kf2mps[id_kf]) < len(self.cg.edges_kf2mps[id_kf1]):
                                kfs_to_cull.add(id_kf)
                                set_close.remove(id_kf)
                                break
                            else:
                                kfs_to_cull.add(id_kf1)
                                set_close.remove(id_kf1)

        for id_kf in kfs_to_cull:
            self.cg.erase_kf(self.cg.kfs[id_kf])

    def update(self, kf_queue: Queue, bow_queue: Queue):
        """
        :param kf_queue: input frames processed to be included into localmapping
        sidefects:
        mutates covisibility graph structure
        adds bow of the last processed kf to a queue for loop closing
        """

        id_kf = kf_queue.get()

        self._insert_new_kf(id_kf)

        self.dbow.add(self.cg.kfs[id_kf])

        bow_queue.put(id_kf)

        self._recent_mps_culling(id_kf)

        self._new_points_creation()

        self._local_ba()

        self._local_kf_culling(id_kf)
