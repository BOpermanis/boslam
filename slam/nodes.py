import numpy as np
from utils import Lock, normalize_t_shape, Rt2se3
# from copy import deepcopy
# import g2o
# from camera import Frame
# from slam.covisibility_graph import CovisibilityGraph
# from slam.bow_db import Dbow


class KeyFrame:

    def __init__(self, frame, dbow, cg):
        self.lock = Lock()

        self.R = frame.R
        self.t = normalize_t_shape(frame.t)
        self.cg = cg
        self.dbow = dbow

        self.id = frame.id

        self.cloud = frame.cloud
        # self.kp = frame.kp
        self.des = frame.des
        self.cloud_kp = frame.cloud_kp
        self.kp_arr = frame.kp_arr
        self.is_kf = False

        if frame.des2mp is None:
            self.des2mp = -np.ones((frame.kp_arr.shape[0]), dtype=int)
        else:
            self.des2mp = frame.des2mp

        self.mp2kp = None

    def make_mp2kp(self):
        self.mp2kp = {}
        for i, mp_id in enumerate(self.des2mp):
            self.mp2kp[mp_id] = self.kp_arr[i, :]

    def is_kf_ref(self, flag=None):
        with self.lock:
            if flag is None:
                return self.is_kf
            self.is_kf = flag

    def idf(self):
        with self.lock:
            return self.id

    def Rf(self, R=None):
        with self.lock:
            if R is None:
                return self.R
            self.R = R

    def pose(self):
        with self.lock:
            return Rt2se3(self.R, self.t)

    def tf(self, t=None):
        with self.lock:
            if t is None:
                return self.t
            self.t = normalize_t_shape(t)

    def cloudf(self, cloud=None):
        with self.lock:
            if cloud is None:
                return self.cloud
            self.cloud = cloud

    def kpf(self, kp=None):
        with self.lock:
            if kp is None:
                return self.kp
            self.kp = kp

    def desf(self, des=None):
        with self.lock:
            if des is None:
                return self.des
            self.des = des

    def cloud_kpf(self, cloud_kp=None):
        with self.lock:
            if cloud_kp is None:
                return self.cloud_kp
            self.cloud_kp = cloud_kp

    def kp_arrf(self, kp_arr=None):
        with self.lock:
            if kp_arr is None:
                return self.kp_arr
            self.kp_arr = kp_arr

    def des2mpf(self, des2mp=None):
        with self.lock:
            if des2mp is None:
                return self.des2mp
            self.des2mp = des2mp


class MapPoint:
    cnt = 0

    def __init__(self, feat, t, n, id_kf, dbow, cg):
        self.id = MapPoint.cnt
        MapPoint.cnt += 1

        self.lock = Lock()

        self.first_kf = id_kf # tikai prieksh mp culling

        self.n = n / np.linalg.norm(n)
        self.pt3d = normalize_t_shape(t)
        self.feat = feat
        self.n_obs = 1
        self.obs = [(id_kf, feat)]

        self.num_frames_found = 1
        self.num_frames_visible = 1

        self.cg = cg
        self.dbow = dbow

    def get_word(self):
        with self.lock:
            return self.dbow.voc.feat_id(self.feat)

    def get_found_ratio(self):
        with self.lock:
            return self.num_frames_found / self.num_frames_visible

    def add_observation(self, feat, n, id_kf):
        with self.lock:
            n /= np.linalg.norm(n)
            self.n_obs += 1
            if len(self.obs) < 10:
                self.obs.append((id_kf, feat))
                dists = np.zeros((len(self.obs), len(self.obs)), dtype=np.uint8)
                for i1, (_, f1) in enumerate(self.obs):
                    for i2, (_, f2) in enumerate(self.obs):
                        if i1 < i2:
                            dists[i2, i1] = dists[i1, i2] = self.dbow.distance(f1, f2)
                self.feat = self.obs[np.argmin(np.median(dists, axis=0))][1]
            self.n = (n + (self.n_obs - 1) * self.n) / self.n_obs

    def erase_observation(self, id_kf):
        with self.lock:
            flag_deleted = False
            for i, (id_kf1, _) in enumerate(self.obs):
                if id_kf == id_kf1:
                    flag_deleted = True
                    break
            if flag_deleted:
                self.obs.pop(i)
                self.n_obs -= 1

    def featf(self):
        with self.lock:
            return self.feat

    def num_frames_found_increment(self):
        with self.lock:
            self.num_frames_found += 1

    def num_frames_visible_increment(self):
        with self.lock:
            self.num_frames_visible += 1

    def pt3df(self, pt3d=None):
        with self.lock:
            if pt3d is None:
                return self.pt3d
            self.pt3d = normalize_t_shape(self.pt3d)

    def nf(self, n=None):
        with self.lock:
            if n is None:
                return self.n
            self.n = n

    def idf(self):
        with self.lock:
            return self.id
