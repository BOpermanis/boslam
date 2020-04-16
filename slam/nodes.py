class KeyFrame:
    def __init__(self, frame_ob, R, t):
        self.R = R
        self.t = t
        self.frame_ob = frame_ob
        self.set_mps = set() # set of mappoint ids


class MapPoint:
    def __init__(self, feat):
        self.feat = feat
        self.n_obs = 1

    def get_viewing_direction(self):
        pass

