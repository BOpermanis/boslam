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

        # self.frame_ob = frame
        self.des = frame.des
        self.set_mps = set() # set of mappoint ids
        self.id = frame.id

        self.cloud = frame.cloud
        self.kp = frame.kp
        self.des = frame.des
        self.cloud_kp = frame.cloud_kp
        self.kp_arr = frame.kp_arr

        self.des2mp = -np.ones((len(self.kp)), dtype=int)


class MapPoint:
    cnt = 0

    def __init__(self, feat, t, n, dbow: Dbow, cg: CovisibilityGraph):
        self.id = MapPoint.cnt
        MapPoint.cnt += 1

        self.n = n
        self.pt3d = t
        self.feat = feat
        self.n_obs = 1
        self.obs = [feat]

        self.cg = cg
        self.dbow = dbow

    def get_word(self):
        return self.dbow.voc.feat_id(self.feat)

    def add_observation(self, feat, n):
        if len(self.obs) > 5:
            self.obs.pop(0)
        self.obs.append(feat)


    def get_viewing_direction(self):
        pass

