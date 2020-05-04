import numpy as np
import g2o
from camera import Frame
from slam.covisibility_graph import CovisibilityGraph
from slam.bow_db import Dbow

class KeyFrame:

    def __init__(self, frame: Frame, dbow: Dbow, cg: CovisibilityGraph):
        self.R = frame.R
        self.t = frame.t
        self.cg = cg
        self.dbow = dbow

        self.id = frame.id

        self.cloud = frame.cloud
        self.kp = frame.kp
        self.des = frame.des
        self.cloud_kp = frame.cloud_kp
        self.kp_arr = frame.kp_arr

        if frame.des2mp is None:
            self.des2mp = -np.ones((len(frame.kp)), dtype=int)
        else:
            self.des2mp = frame.des2mp


class MapPoint:
    cnt = 0

    def __init__(self, feat, t, n, id_kf, dbow: Dbow, cg: CovisibilityGraph):
        self.id = MapPoint.cnt
        MapPoint.cnt += 1

        self.first_kf = id_kf # tikai prieksh mp culling

        self.n = n / np.linalg.norm(n)
        self.pt3d = t
        self.feat = feat
        self.n_obs = 1
        self.obs = [(id_kf, feat)]

        self.num_frames_found = 1
        self.num_frames_visible = 1

        self.cg = cg
        self.dbow = dbow

    def get_word(self):
        return self.dbow.voc.feat_id(self.feat)

    def get_found_ratio(self):
        return self.num_frames_found / self.num_frames_visible

    def add_observation(self, feat, n, id_kf):
        n /= np.linalg.norm(n)
        self.n_obs += 1
        if len(self.obs) < 10:
            self.obs.append((id_kf, feat))
            dists = np.zeros((len(self.obs), len(self.obs)), dtype=np.uint8)
            for i1, (_, f1) in enumerate(self.obs):
                for i2, (_, f2) in enumerate(self.obs):
                    if i1 < i2:
                        dists[i2, i1] = dists[i1, i2] = self.dbow.distance(f1, f2)
            self.feat = self.obs[np.argmin(np.median(dists, axis=0))]
        self.n = (n + (self.n_obs - 1) * self.n) / self.n_obs

    def erase_observation(self, id_kf):
        flag_deleted = False
        for i, (id_kf1, _) in enumerate(self.obs):
            if id_kf == id_kf1:
                flag_deleted = True
                break
        if flag_deleted:
            self.obs.pop(i)
            self.n_obs -= 1

