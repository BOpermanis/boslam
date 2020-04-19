import numpy as np
import g2o
from camera import Frame


class KeyFrame:
    def __init__(self, frame: Frame, R: np.ndarray, t: np.ndarray):
        self.R = R
        self.t = t
        # self.frame_ob = frame
        self.des = frame.des
        self.set_mps = set() # set of mappoint ids
        self.id = None


class MapPoint:
    def __init__(self, feat, t):
        self.pt3d = t
        self.feat = feat
        self.n_obs = 1
        self.obs = [feat]
        self.id = None

    def get_viewing_direction(self):
        pass

