import numpy as np
import cv2
from typing import Dict, Set, Tuple, List
import g2o
from multiprocessing import Queue
from itertools import combinations
from collections import defaultdict

from slam.covisibility_graph import CovisibilityGraph
from slam.nodes import KeyFrame
from config import d_hamming_max, min_common_ratio, min_found_ratio, min_n_obs
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
                # if not (mp.get_found_ratio() >= min_found_ratio and (mp.n_obs >= min_n_obs or mp.first_kf == id_kf)):
                if not (mp.get_found_ratio() >= min_found_ratio): # and mp.n_obs >= min_n_obs):
                    mps_to_delete.append(mp)
        for mp in mps_to_delete:
            self.cg.erase_mp(mp)

    def _new_points_creation(self):
        # TODO checks all new mappoint correspondences, if mapppoint survives all criterion, then it is added to covisibility graph, ready for BA
        # this is done in tracking thread by this implementation
        pass

    def _local_ba(self, kf):
        self.cg.optimize_local(kf)

    def _local_kf_culling(self, id_kf):
        # !!! orbslam2 kodā ir cikls cauri kfs, katrai fičai iet cauri un skatās vai tas ir redundnt observation,
        # tad saskaita cik tādu daudz ir un skatās vai prop > 0.9, ja ir tad vnk izmet
        kfs_to_cull = []
        ids = self.cg.get_local_map(id_kf, flag_with_input_kf=False)

        with self.cg.lock_edges_kf2mps:
            for id_kf1 in self.cg.edges_kf2mps:
                if len(self.cg.edges_kf2mps[id_kf1]) < 15:
                    kfs_to_cull.append(id_kf1)

        with self.cg.lock_kfs:
            for id_kf1 in kfs_to_cull:
                if id_kf1 in self.cg.kfs:
                    self.cg.erase_kf(self.cg.kfs[id_kf1])
                if id_kf1 in ids:
                    ids.remove(id_kf1)

        voter = defaultdict(list)
        with self.cg.lock_edges_kf2mps and self.cg.lock_kf2kf_num_common_mps:
            for i1, i2 in combinations(ids, 2):
                num_common = self.cg.kf2kf_num_common_mps[key_common_mps(i2, i1)]
                if num_common >= 15:
                    p1 = num_common / len(self.cg.edges_kf2mps[i1])
                    p2 = num_common / len(self.cg.edges_kf2mps[i2])
                    if p1 > min_common_ratio:
                        voter[i1].append(i2)
                    if p2 > min_common_ratio:
                        voter[i2].append(i1)

        l1 = []
        for id_kf, l in voter.items():
            if len(l) > 2:
                l1.append((id_kf, l, len(l)))

        triples = sorted(l1, key=lambda x: -x[2])
        kfs_to_cull = {*[]}

        with self.cg.lock_edges_kf2mps:
            while len(triples) > 0:
                id_kf, l, n = triples.pop(0)
                if id_kf in kfs_to_cull:
                    continue
                for id_kf1 in l:
                    if id_kf1 not in kfs_to_cull:
                        #  tests kuru izmest + izmeshana no set_close
                        if len(voter[id_kf]) > len(voter[id_kf1]):
                            kfs_to_cull.add(id_kf)
                            break
                        elif len(voter[id_kf]) < len(voter[id_kf1]):
                            kfs_to_cull.add(id_kf1)
                        else:
                            if len(self.cg.edges_kf2mps[id_kf]) < len(self.cg.edges_kf2mps[id_kf1]):
                                kfs_to_cull.add(id_kf)
                                break
                            else:
                                kfs_to_cull.add(id_kf1)

        with self.cg.lock_edges_mp2kfs:
            lens = []
            for v in self.cg.edges_mp2kfs.values():
                lens.append(len(v))
        num_obs = []
        with self.cg.lock_mps:
            # max_mp_id = np.max(list(self.cg.mps.keys()))
            for mp in self.cg.mps.values():
                num_obs.append(len(mp.obs))

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

        self._local_ba(self.cg.kfs[id_kf])

        self._local_kf_culling(id_kf)
