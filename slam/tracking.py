import numpy as np
import cv2
from pprint import pprint
from multiprocessing import Queue

from typing import Dict, Set, Tuple, List
# pprint([a for a in dir(cv2) if "pnp" in a.lower()])
# print(cv2.SOLVEPNP_EPNP)
# exit()
from pyDBoW3 import Database
import g2o

from config import d_hamming_max, min_matches_cg, dbow_tresh
from slam.covisibility_graph import CovisibilityGraph
from camera import Frame, RsCamera
from slam.nodes import KeyFrame, MapPoint

cos60 = np.cos(np.pi / 3)
dmin = 0.1
dmax = 20


# tracker states
state_map_init = 0
state_ok = 1
state_lost = 2


class Tracker:

    def __init__(self, cg: CovisibilityGraph, dbow: Database, cam: RsCamera):
        self.state = state_map_init
        self.cam_mat, self.dist_coefs = cam.cam_mat, cam.distCoeffs
        self.t_from_last_kf = 0

        self.cg = cg # covisibility graph
        self.dbow = dbow # bag of words database
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        self.last_frame = None

    def _track_with_motion(self, frame):
        if self.last_frame.relt is None:
            return False
        # TODO tracking with motion
        pass

    def _track_wrt_refkf(self, frame):

        # TODO
        pass

    def _track_local_map(self, frame):
        pass

    def match_to_kf(self, kf: KeyFrame, frame: Frame):
        matches = self.matcher.match(kf.des, frame.des)
        inds_ref, inds = zip(*((_.queryIdx, _.trainIdx) for _ in matches if _.distance <= d_hamming_max))

        if len(inds) < min_matches_cg:
            self.state = state_lost
            return None, None, 0.0

        is_ok, R, t, inliers = cv2.solvePnPRansac(kf.cloud_kp[inds_ref, :],
                                                  frame.kp_arr[inds, :],
                                                  self.cam_mat, self.dist_coefs,
                                                  flags=cv2.SOLVEPNP_EPNP)
        if not is_ok:
            return None, None, 0.0

        if len(inliers) < min_matches_cg:
            return None, None, 0.0

        R = cv2.Rodrigues(R)[0]
        return R, t, (len(inds) - len(inliers)) / len(inds_ref)

    def _project_local_map(self, frame: Frame):
        ids_matching = List[Tuple[int, int]]
        feats = List[np.ndarray]
        pts3d = []
        for kf in self.cg.local_map:
            for mp in kf:
                pose = g2o.SE3Quat(frame.R, frame.t)
                pixel = self.cam.cam_map(pose * mp.pt3d)
                if 0 <= pixel[0] < self.width and 0 <= pixel[1] < self.height:
                    if np.dot(frame.see_vector, mp.viewing_direction) < cos60:
                        if dmin < np.linalg.norm(mp.t - frame.t) < dmax:
                            # TODO check if scales matches
                            ids_matching.append((kf.id, mp.id))
                            feats.append(mp.feat)
                            pts3d.append(mp.t)

        matches = self.matcher.match(frame.des, feats)
        inds_frame, inds = zip(*((_.queryIdx, _.trainIdx) for _ in matches if _.distance <= d_hamming_max))

        # is_ok, R, t, inliers = cv2.solvePnPRansac(pts3d[inds_frame, :],
        #                                           frame.kp_arr[inds, :],
        #                                           self.cam_mat, self.dist_coefs,
        #                                           flags=cv2.SOLVEPNP_EPNP)

        # TODO ko darīt šeit, publikaacijaa rakstiits ka sheit pozu nosaka ar movement only BA
        if is_ok:
            return R, t
        return None, None

    def _ok_as_new_keyframe(self, frame):
        pass

    def update(self, frame):
        self.t_from_last_kf += 1
        R, t = None, None

        if self.state == state_map_init:
            if len(frame.des) >= 50:
                R, t = np.eye(3), np.zeros(3)
                self.cg.add_kf(KeyFrame(frame, R, t))
                self.t_from_last_kf = 0
                self.last_frame = frame
                self.state = state_ok
                return R, t

        if self.state == state_ok:
            is_ok = self._track_with_motion(frame)

            if not is_ok:
                is_ok = self._track_wrt_refkf(frame)
                if not is_ok:
                    self.state = state_lost

            if is_ok:
                is_ok = self._track_local_map(frame)
                if not is_ok:
                    self.state = state_lost

            if is_ok:
                self._ok_as_new_keyframe(frame)

        if self.state == state_lost:
            # TODO check if this implementation is ok
            q = self.dbow.query(frame.des, [], 1, -1)[0]
            if q.Score >= dbow_tresh:
                R, t = self.match_to_kf(self.cg.dict_kfs[q.Id], frame)
                if R is not None:
                    self.t_from_last_kf = 0
                    self.state = state_ok

        if self.state == state_ok:
            self.last_frame = frame
            
        return R, t

