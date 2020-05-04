import numpy as np
import cv2
from pprint import pprint
from collections import Counter

from typing import Dict, Set, Tuple, List
# pprint([a for a in dir(cv2) if "pnp" in a.lower()])
# print(cv2.SOLVEPNP_EPNP)
# exit()
from slam.bow_db import Dbow
import g2o

from config import d_hamming_max, min_matches_cg, dbow_tresh, num_frames_from_last_kf, num_frames_from_last_relocation
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
state_relocated = 3


class Tracker:

    def __init__(self, cg: CovisibilityGraph, dbow: Dbow, cam: RsCamera):
        self.state = state_map_init
        self.width = cam.width
        self.height = cam.height
        self.cam = g2o.CameraParameters(cam.f, cam.principal_point, cam.baseline)
        self.cam_mat, self.dist_coefs = cam.cam_mat, cam.distCoeffs

        self.t_from_last_kf = 0
        self.t_from_last_relocation = 0

        self.cg = cg # covisibility graph
        self.dbow = dbow # bag of words database
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        self.last_frame = None
        self.kf_ref = None

    def _track_with_motion(self, frame):
        if self.last_frame.relt is None:
            return False
        # TODO tracking with motion
        pass

    def _track_wrt_refkf(self, frame):

        matches = self.matcher.match(frame.des, self.kf_ref.des)
        inds_f, inds_kf = zip(*((_.queryIdx, _.trainIdx) for _ in matches if _.distance < d_hamming_max))

        if len(inds_f) < min_matches_cg:
            self.state = state_lost
            return False, 0.0

        is_ok, R, t, inliers = cv2.solvePnPRansac(frame.cloud_kp[inds_f, :],
                                                  self.kf_ref.kp_arr[inds_kf, :],
                                                  self.cam_mat, self.dist_coefs,
                                                  flags=cv2.SOLVEPNP_EPNP)
        R = cv2.Rodrigues(R)[0]

        if len(inliers) < 15:
            return False, 0.0

        frame.kf_ref = self.kf_ref.id
        frame.rel_to_kf = R, t
        frame.setPose(R, t)
        return True, len(inliers) / len(self.kf_ref.des.shape[0])

    def _track_local_map(self, frame):
        ids_matching_kfs = []
        ids_matching_mps = []
        feats = List[np.ndarray]
        pts3d = []
        for id_kf in self.cg.get_local_map(self.kf_ref):
            for id_mp in self.cg.edges_kf2mps[id_kf]:
                mp = self.cg.mps[id_mp]
                pose = g2o.SE3Quat(frame.R, frame.t)
                pixel = self.cam.cam_map(pose * mp.pt3d)
                if 0 <= pixel[0] < self.width and 0 <= pixel[1] < self.height:
                    if np.dot(frame.see_vector, mp.n) < cos60:
                        if dmin < np.linalg.norm(mp.t - frame.t) < dmax:
                            # TODO check if scales matches
                            ids_matching_kfs.append(id_kf)
                            ids_matching_mps.append(id_mp)
                            feats.append(mp.feat)
                            pts3d.append(mp.t)
                            self.cg.mps[id_mp].num_frames_visible += 1

        matches = self.matcher.match(frame.des, feats)
        inds_frame, inds = zip(*((_.queryIdx, _.trainIdx) for _ in matches if _.distance <= d_hamming_max))

        is_ok, R, t, inliers = cv2.solvePnPRansac(pts3d[inds_frame, :],
                                                  frame.kp_arr[inds, :],
                                                  self.cam_mat, self.dist_coefs,
                                                  flags=cv2.SOLVEPNP_EPNP)

        if not is_ok or len(inliers) < 15:
            return False

        frame.des2mp = -np.ones((len(frame.kp)), dtype=int)
        for i_feat, id_mp in zip(inds_frame[inliers], ids_matching_mps[inds[inliers]]):
            frame.des2mp[i_feat] = id_mp
            self.cg.mps[id_mp].num_frame_found += 1

        # matched frame features and matching mappoints
        id_new_kf_ref = Counter(ids_matching_kfs[inds[inliers]]).most_common(1)[0]
        self.kf_ref = self.cg.kfs[id_new_kf_ref]

        R = cv2.Rodrigues(R)[0]
        frame.setPose(R, t)

        return True

    def _ok_as_new_keyframe(self, frame, r):
        if self.t_from_last_kf > num_frames_from_last_kf and r < 0.9 and \
                        frame.des.shape[0] >= 50 and self.t_from_last_relocation > num_frames_from_last_relocation:
            return True
        return False

    def update(self, frame, kf_queue):
        self.t_from_last_kf += 1
        self.t_from_last_relocation += 1

        if self.state == state_map_init:
            if len(frame.des) >= 500:
                R, t = np.eye(3), np.zeros(3)
                frame.setPose(R, t)
                kf = self.cg.add_kf(frame)
                self.t_from_last_kf = 0
                self.last_frame = frame
                self.kf_ref = kf
                self.state = state_ok
                return R, t

        if self.state == state_lost:
            q = self.dbow.query(frame)
            if q.Score >= dbow_tresh:
                self.kf_ref = self.cg.kfs[q.Id]
                self.state = state_relocated

        if self.state in (state_ok, state_relocated):
            r = 0.0
            is_ok = False
            if self.state == state_ok:
                is_ok = self._track_with_motion(frame)

            if not is_ok:
                is_ok, r = self._track_wrt_refkf(frame)
                if not is_ok:
                    self.state = state_lost

            if is_ok:
                is_ok = self._track_local_map(frame)
                if not is_ok:
                    self.state = state_lost

            if is_ok:
                if self._ok_as_new_keyframe(frame, r):
                    kf = self.cg.add_kf(frame)
                    self.kf_ref = kf
                    kf_queue.put(kf.id)

        if self.state == state_ok:
            self.last_frame = frame
        else:
            self.last_frame = None

        return frame.R, frame.t

