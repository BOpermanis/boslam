import numpy as np
import cv2
import g2o
# from typing import Dict, Set, Tuple, List
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations, chain

# from multiprocessing import Lock
# from threading import Lock
from utils import Lock
from config import d_hamming_max
from camera import Frame
from utils import key_common_mps

from slam.nodes import KeyFrame, MapPoint

cos60 = np.cos(np.pi / 3)
dmin = 0.1
dmax = 20


class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, cam):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)
        super().verbose()
        self.cam = cam

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().set_verbose(True)
        super().optimize(max_iterations)

    def add_pose(self, kf: KeyFrame, cam, fixed=False):
        pose_id, R, t = kf.idf(), kf.Rf(), kf.tf()
        if len(t.shape) == 2:
            t = t[:, 0]
        pose_estimate = g2o.SE3Quat(R, t)
        sbacam = g2o.SBACam(pose_estimate.orientation(), pose_estimate.position())
        if hasattr(cam, 'fx'):
            sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)
        else:
            cx, cy = cam.principal_point
            fx, fy = cam.focal_length, cam.focal_length
            sbacam.set_cam(fx, fy, cx, cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_point(self, mp: MapPoint, fixed=False, marginalized=True):
        point_id, point = mp.idf(), mp.pt3df()
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, mp, kf, measurement, information=np.identity(2), robust_kernel=g2o.RobustKernelHuber(np.sqrt(7.815))):   # 95% CI
        edge = g2o.EdgeProjectP2MC()
        point_id, pose_id = mp.idf(), kf.idf()
        pose = g2o.SE3Quat(kf.Rf(), kf.tf())#[:, 0])
        pixel = self.cam.cam_map(pose * mp.pt3d)
        if 0 <= pixel[0] < 640 and 0 <= pixel[1] < 480:
            edge.set_vertex(0, self.vertex(point_id * 2 + 1))
            edge.set_vertex(1, self.vertex(pose_id * 2))
            edge.set_measurement(measurement)   # projection
            edge.set_information(information)

            if robust_kernel is not None:
                edge.set_robust_kernel(robust_kernel)
            super().add_edge(edge)

    def get_pose(self, pose_id):
        m = self.vertex(pose_id * 2).estimate().matrix()
        return m[:3, :3], m[:3, 3]

    def get_point(self, point_id):
        m = self.vertex(point_id * 2 + 1).estimate()
        return m.T


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

        with self.lock_edges_kf2mps and self.lock_edges_mp2kfs:
            for id_mp in self.edges_kf2mps[kf.id]:
                self.edges_mp2kfs[id_mp].remove(kf.id)

        with self.lock_edges_kf2kfs and self.lock_kf2kf_num_common_mps:
            for id_kf in self.edges_kf2kfs[kf.id]:
                del self.kf2kf_num_common_mps[key_common_mps(kf.id, id_kf)]
                self.edges_kf2kfs[id_kf].remove(kf.id)
        with self.lock_kfs and self.lock_edges_kf2kfs and self.lock_edges_kf2mps:
            del self.edges_kf2kfs[kf.id]
            del self.edges_kf2mps[kf.id]
            del self.kfs[kf.id]
            del kf

    def erase_mp(self, mp):
        if isinstance(mp, int):
            with self.lock_mps:
                mp = self.mps[mp]
        id_kfs = []
        with self.lock_edges_mp2kfs and self.lock_edges_kf2mps:
            for id_kf in self.edges_mp2kfs[mp.id]:
                self.edges_kf2mps[id_kf].remove(mp.id)
                id_kfs.append(id_kf)

        with self.lock_kf2kf_num_common_mps:
            for i1, i2 in combinations(id_kfs, 2):
                self.kf2kf_num_common_mps[key_common_mps(i2, i1)] -= 1

        with self.lock_mps and self.lock_edges_mp2kfs:
            del self.mps[mp.id]
            del self.edges_mp2kfs[mp.id]
            del mp

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

            optimizer.optimize(5)

            for i in chain([kf.id], inner, outter):
                R, t = optimizer.get_pose(i)
                self.kfs[i].Rf(R)
                self.kfs[i].tf(t)

            for i in counter:
                t = optimizer.get_point(i)
                self.mps[i].pt3df(t)
        del optimizer
        # TODO izmest outlaijerus

    def maintain(self):
        with self.lock_mps:
            pass
        with self.lock_kf2kf_num_common_mps:
            pass

    def get_stats(self):

        stats = OrderedDict()

        with self.lock_kfs:
            stats["kfs_len"] = len(self.kfs)

        with self.lock_mps:
            stats["mps_len"] = len(self.mps)

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

                stats["p_min"] = np.min(a)
                stats["p_max"] = np.max(a)
                stats["p_avg"] = np.average(a)


        return stats



