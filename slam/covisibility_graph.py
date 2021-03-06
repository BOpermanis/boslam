import numpy as np
import cv2
import g2o
# from typing import Dict, Set, Tuple, List
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations, chain
from pprint import pprint
from scipy.spatial import distance_matrix

# from multiprocessing import Lock
# from threading import Lock
from utils import Lock
from config import d_hamming_max, chi2_sig_value
from camera import Frame
from utils import key_common_mps

from slam.nodes import KeyFrame, MapPoint
from slam.optimizer import BundleAdjustment

cos60 = np.cos(np.pi / 3)
dmin = 0.1
dmax = 20

class CovisibilityGraph():
    def __init__(self, dbow, camera):

        self.dbow = dbow
        self.cam = g2o.CameraParameters(camera.f, camera.principal_point, camera.baseline)
        self.cam.set_id(0)

        # params
        self.width = camera.width
        self.height = camera.height
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        # data and counters
        self.kfs = {} #Dict[int, KeyFrame]
        self.mps = {} #Dict[int, MapPoint]
        self.kf2kf_num_common_mps = defaultdict(lambda: 0)
        self.edges_kf2kfs = defaultdict(set) #Dict[int, List[int, int]] # list of (#common mps, id kf)
        self.edges_kf2mps = defaultdict(set) #Dict[int, List[int]]
        self.edges_mp2kfs = defaultdict(set) #Dict[int, List[int]]

        self.lock_kfs = Lock()
        self.lock_mps = Lock()
        self.lock_kf2kf_num_common_mps = Lock()
        self.lock_edges_kf2kfs = Lock()
        self.lock_edges_kf2mps = Lock()
        self.lock_edges_mp2kfs = Lock()

    def get_local_map(self, kf, flag_with_input_kf=True, flag_inner_outer=False):
        if isinstance(kf, int):
            with self.lock_kfs:
                kf = self.kfs[kf]

        with kf.lock:
            with self.lock_kfs:
                if isinstance(kf, int):
                    kf = self.kfs[kf]

            set_ids_inner = {kf.id}
            set_ids_outter = set()
            with self.lock_edges_kf2kfs and self.lock_kf2kf_num_common_mps:
                for id_kf1 in self.edges_kf2kfs[kf.id]:
                    if self.kf2kf_num_common_mps[key_common_mps(id_kf1, kf.id)] >= 15:
                        set_ids_inner.add(id_kf1)
                        for id_kf2 in self.edges_kf2kfs[id_kf1]:
                            if id_kf2 != kf.id:
                                if self.kf2kf_num_common_mps[key_common_mps(id_kf2, kf.id)] >= 15:
                                    set_ids_outter.add(id_kf2)

            if not flag_with_input_kf:
                set_ids_inner.remove(kf.id)

            if flag_inner_outer:
                return set_ids_inner, set_ids_outter.difference(set_ids_inner)

            return set_ids_inner.union(set_ids_outter)

    def add_edge_to_cg(self, kf, mp, num_common=None):
        if isinstance(mp, KeyFrame):
            assert kf.id in self.kfs and mp.id in self.kfs

            with self.lock_edges_kf2mps:
                if num_common is None:
                    num_common = len(self.edges_kf2mps[kf.id].intersection(self.edges_kf2mps[mp.id]))

            with self.lock_edges_kf2kfs:
                self.edges_kf2kfs[kf.id].add(mp.id)
                self.edges_kf2kfs[mp.id].add(kf.id)

            with self.lock_kf2kf_num_common_mps:
                self.kf2kf_num_common_mps[key_common_mps(kf.id, mp.id)] = num_common
        else:
            assert kf.id in self.kfs and mp.id in self.mps
            with self.lock_edges_kf2mps:
                self.edges_kf2mps[kf.id].add(mp.id)

            with self.lock_edges_kf2kfs and self.lock_kf2kf_num_common_mps:
                for id_kf1 in self.edges_mp2kfs[mp.id]:
                    self.edges_kf2kfs[id_kf1].add(kf.id)
                    self.edges_kf2kfs[kf.id].add(id_kf1)
                    self.kf2kf_num_common_mps[key_common_mps(kf.id, id_kf1)] += 1

            with self.lock_edges_mp2kfs:
                self.edges_mp2kfs[mp.id].add(kf.id)

    def get_loop_candidates(self):
        pass

    def num_kfs(self):
        with self.lock_kfs:
            return len(self.kfs)

    def num_mps(self):
        with self.lock_mps:
            return len(self.mps)

    def add_kf(self, frame: Frame):
        self.dbow.add(frame)
        kf = KeyFrame(frame, self.dbow, self)
        with self.lock_kfs and kf.lock and self.lock_mps:
            self.kfs[kf.id] = kf

            for i_feat, (feat, d3, id_mp) in enumerate(zip(kf.des, kf.cloud_kp, kf.des2mp)):
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
                    self.add_edge_to_cg(kf, self.mps[id_mp])
        kf.make_mp2kp()
        return kf

    def erase_kf(self, kf):
        self.dbow.erase(kf.id)

        with self.lock_edges_kf2mps and self.lock_edges_mp2kfs and self.lock_edges_kf2kfs and self.lock_kf2kf_num_common_mps and self.lock_kfs:
            for id_mp in self.edges_kf2mps[kf.id]:
                self.edges_mp2kfs[id_mp].remove(kf.id)

            for id_kf in self.edges_kf2kfs[kf.id]:
                del self.kf2kf_num_common_mps[key_common_mps(kf.id, id_kf)]
                self.edges_kf2kfs[id_kf].remove(kf.id)

            del self.edges_kf2kfs[kf.id]
            del self.edges_kf2mps[kf.id]
            del self.kfs[kf.id]
            del kf

    def erase_mp(self, mp):
        if isinstance(mp, int):
            with self.lock_mps:
                mp = self.mps[mp]
        id_kfs = []
        with self.lock_mps and self.lock_edges_mp2kfs and self.lock_edges_kf2mps and self.lock_kf2kf_num_common_mps:
            for id_kf in self.edges_mp2kfs[mp.id]:
                self.edges_kf2mps[id_kf].remove(mp.id)
                id_kfs.append(id_kf)

            for i1, i2 in combinations(id_kfs, 2):
                self.kf2kf_num_common_mps[key_common_mps(i2, i1)] -= 1

            del self.mps[mp.id]
            del self.edges_mp2kfs[mp.id]
            del mp

    def erase_edge(self, id_kf, id_mp):
        with self.lock_edges_mp2kfs and self.lock_kf2kf_num_common_mps and self.lock_edges_kf2kfs and self.lock_edges_kf2mps:
            for i1 in self.edges_mp2kfs[id_mp]:
                if i1 != id_kf:
                    key = key_common_mps(i1, id_kf)
                    self.kf2kf_num_common_mps[key] -= 1
                    if self.kf2kf_num_common_mps[key] == 0:
                        self.edges_kf2kfs[i1].remove(id_kf)
                        self.edges_kf2kfs[id_kf].remove(i1)

            self.edges_mp2kfs[id_mp].remove(id_kf)
            self.edges_kf2mps[id_kf].remove(id_mp)

    def optimize_local(self, kf: KeyFrame):
        optimizer = BundleAdjustment(self.cam)
        with self.lock_kfs and self.lock_mps:
            inner, outter = self.get_local_map(kf, flag_inner_outer=True, flag_with_input_kf=False)
            if len(outter) == 0:
                if len(inner) == 0:
                    print("nothing to optimize")
                    return
                optimizer.add_pose(kf, self.cam, fixed=True)
            else:
                optimizer.add_pose(kf, self.cam, fixed=False)
                for i in outter:
                    optimizer.add_pose(self.kfs[i], self.cam, fixed=True)

            for i in inner:
                optimizer.add_pose(self.kfs[i], self.cam, fixed=False)
            # mps with more than 1 occ between pts
            pairs = []
            counter = Counter()
            for i in chain([kf.id], inner, outter):
                counter.update(self.edges_kf2mps[i])
                pairs.extend(zip([i] * len(self.edges_kf2mps[i]), self.edges_kf2mps[i]))

            s = set()
            for kf_id, mp_id in pairs:
                if counter[mp_id] > 1:
                    if mp_id not in s:
                        s.add(mp_id)
                        optimizer.add_point(self.mps[mp_id])
                    pixels = self.kfs[kf_id].mp2kp[mp_id]
                    optimizer.add_edge(self.mps[mp_id], self.kfs[kf_id], pixels)
                else:
                    del counter[mp_id]

            optimizer.optimize(7)

            for i in chain([kf.id], inner, outter):
                R, t = optimizer.get_pose(i)
                self.kfs[i].Rf(R)
                self.kfs[i].tf(t)

            for i in counter:
                t = optimizer.get_point(i)
                self.mps[i].pt3df(t)

        bad_edges = optimizer.outlier_edges()
        for id_kf, id_mp in bad_edges:
            self.erase_edge(id_kf, id_mp)

        del optimizer

    def get_stats(self, flag_unique_mp_feats=False):

        stats = OrderedDict()

        with self.lock_kfs:
            l = list(self.kfs.keys())
            stats["kfs_len"] = len(self.kfs)
            stats["kfs_min_id"] = np.min(l) if len(l) > 0 else None
            stats["kfs_max_id"] = np.max(l) if len(l) > 0 else None

        with self.lock_mps:
            l = list(self.mps.keys())
            stats["mps_len"] = len(l)
            stats["mps_min_id"] = np.min(l) if len(l) > 0 else None
            stats["mps_max_id"] = np.max(l) if len(l) > 0 else None
            pts = []
            if len(l) > 0:
                for id_mp in self.mps:
                    pts.append(self.mps[id_mp].pt3df())
                stats["mps_avg_vec"] = np.average(np.stack(pts), axis=0)
            else:
                stats["mps_avg_vec"] = None

            if flag_unique_mp_feats:
                feats = []
                ids = []
                ts = []
                for ii, mp in self.mps.items():
                    x = mp.featf()
                    x = x.astype(np.uint64)
                    x = x * np.arange(len(x)).astype(np.int64)
                    feats.append(np.sum(x))
                    ids.append(ii)
                    ts.append(mp.pt3df())

                m0 = np.max(distance_matrix(ts, ts))

                pairs = defaultdict(list)
                for ii, feat in zip(ids, feats):
                    pairs[feat].append(ii)

                m = 0
                for feat, iis in pairs.items():
                    if len(iis) > 1:
                        ts = [self.mps[ii].pt3df() for ii in iis]
                        m = max(m, np.max(distance_matrix(ts, ts)))
                print(2222222222222222222, m / m0)
                # print(111111111, Counter(feats).most_common(10))
                stats["mps_unique_mp_feats"] = len(set(feats))

        with self.lock_edges_kf2mps:
            l = [len(_) for _ in self.edges_kf2mps.values()]
            if len(l) > 0:
                stats["kf2mps_min"] = np.min(l)
                stats["kf2mps_max"] = np.max(l)

        with self.lock_edges_kf2kfs:
            l = [len(_) for _ in self.edges_kf2kfs.values()]
            if len(l) > 0:
                stats["kf2kfs_min"] = np.min(l)
                stats["kf2kfs_max"] = np.max(l)

        with self.lock_edges_mp2kfs:
            l = [len(_) for _ in self.edges_mp2kfs.values()]
            if len(l) > 0:
                stats["mp2kfs_min"] = np.min(l)
                stats["mp2kfs_max"] = np.max(l)

        with self.lock_kf2kf_num_common_mps and self.lock_edges_kf2mps:
            if len(self.kf2kf_num_common_mps) > 0:
                l = list(self.kf2kf_num_common_mps.values())
                stats["num_common_mps_min"] = np.min(l)
                stats["num_common_mps_max"] = np.max(l)
                stats["num_common_mps_len"] = len(l)

                a = []
                for (i1, i2), n in self.kf2kf_num_common_mps.items():
                    if len(self.edges_kf2mps[i1]) > 0:
                        a.append(n / len(self.edges_kf2mps[i1]))
                    if len(self.edges_kf2mps[i2]) > 0:
                        a.append(n / len(self.edges_kf2mps[i2]))

                stats["p_min"] = np.min(a) if len(a) > 0 else None
                stats["p_max"] = np.max(a) if len(a) > 0 else None
                stats["p_avg"] = np.average(a) if len(a) > 0 else None

        return stats



