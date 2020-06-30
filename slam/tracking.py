import numpy as np
import cv2
from pprint import pprint
from collections import Counter

# from typing import Dict, Set, Tuple, List
# pprint([a for a in dir(cv2) if "pnp" in a.lower()])
# print(cv2.SOLVEPNP_EPNP)
# exit()
from slam.bow_db import Dbow
import g2o

from config import d_hamming_max, min_matches_cg, dbow_tresh, num_frames_from_last_kf, num_frames_from_last_relocation, min_kp_in_frame
from slam.covisibility_graph import CovisibilityGraph
# from camera import RsCamera
# from slam.nodes import KeyFrame, MapPoint
from slam.optimizer import CamOnlyBA

cos60 = np.cos(np.pi / 3)
dmin = 0.1
dmax = 20

# tracker states
state_map_init = 0
state_ok = 1
state_lost = 2
state_relocated = 3


class Tracker:

    def __init__(self, cg: CovisibilityGraph, dbow: Dbow, cam):
        self.state = state_map_init
        self.width = cam.width
        self.height = cam.height
        self.cam = g2o.CameraParameters(cam.f, cam.principal_point, cam.baseline)
        self.cam_mat, self.dist_coefs = cam.cam_mat, cam.distCoeffs

        self.t_from_last_kf = 0
        self.t_from_last_relocation = 0
        self.t_absolute = 0

        self.cg = cg # covisibility graph
        self.dbow = dbow # bag of words database
        self.matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

        self.last_frame = None
        self.kf_ref = None
        self.mps_visible = set()

    def _track_with_motion(self, frame):
        return False

    def _track_wrt_refkf(self, frame, flag_use_pnp=False):

        matches = self.matcher.match(frame.des, self.kf_ref.desf())
        matches = [_ for _ in matches if _.distance < d_hamming_max]
        if len(matches) == 0:
            return False, 0.0
        inds_f, inds_kf = zip(*((_.queryIdx, _.trainIdx) for _ in matches))

        if len(inds_f) < min_matches_cg:
            self.kf_ref.is_kf_ref(False)
            self.state = state_lost
            return False, 0.0

        if flag_use_pnp:
            is_ok, R, t, inliers = cv2.solvePnPRansac(frame.cloud_kp[inds_f, :],
                                                      self.kf_ref.kp_arrf()[inds_kf, :].astype(np.float64),
                                                      self.cam_mat, self.dist_coefs,
                                                      flags=cv2.SOLVEPNP_EPNP)
        else:
            is_ok, R, t, inliers = CamOnlyBA(frame.kp_arr[inds_f, :], self.kf_ref.cloud_kpf()[inds_kf, :], self.cam, frame.R, frame.t)

        if inliers is None:
            return False, 0.0

        # R = cv2.Rodrigues(R)[0]
        # stats = self.cg.get_stats(flag_unique_mp_feats=True)
        # print("mps_unique_mp_feats", stats['mps_unique_mp_feats'])
        # print("ratio inliers", len(inliers) / len(inds_kf), stats['mps_unique_mp_feats'] / stats["mps_len"])
        print(11111111111111111111111111111111111, len(inliers))
        if len(inliers) < 15:
            return False, 0.0

        frame.kf_ref = self.kf_ref.idf()
        frame.rel_to_kf = R, t
        frame.setPose(R, t)
        return True, len(inliers) / self.kf_ref.desf().shape[0]

    def _track_local_map(self, frame):
        ids_matching_kfs = []
        ids_matching_mps = []
        feats = []
        pts3d = []
        with self.cg.lock_mps and self.cg.lock_edges_kf2mps:
            for id_kf in self.cg.get_local_map(self.kf_ref):
                for id_mp in list(self.cg.edges_kf2mps[id_kf]):
                    mp = self.cg.mps[id_mp]
                    # print(frame.R.shape, frame.t.shape)
                    pose = g2o.SE3Quat(frame.R, frame.t)
                    pixel = self.cam.cam_map(pose * mp.pt3d)
                    if 0 <= pixel[0] < self.width and 0 <= pixel[1] < self.height:
                        if np.dot(frame.see_vector, mp.nf()) < cos60:
                            if True: #dmin < np.linalg.norm(mp.pt3df() - frame.t) < dmax:
                                # TODO check if scales matches
                                ids_matching_kfs.append(id_kf)
                                ids_matching_mps.append(id_mp)
                                feats.append(mp.featf())
                                pts3d.append(mp.pt3df())
                                self.cg.mps[id_mp].num_frames_visible_increment()

            if len(feats) > 0:
                print(444, len(set(ids_matching_mps)) / len(self.cg.mps))

        if len(feats) == 0:
            return False

        feats = np.stack(feats)
        pts3d = np.stack(pts3d)
        matches = [m for m in self.matcher.match(frame.des, feats) if m.distance <= d_hamming_max]

        if len(matches) < 15:
            return False

        inds_frame, inds = zip(*((_.queryIdx, _.trainIdx) for _ in matches))

        is_ok, R, t, inliers = CamOnlyBA(frame.kp_arr[inds_frame, :], pts3d[inds, :], self.cam,
                                         frame.R, frame.t)

        # is_ok, R, t, inliers = cv2.solvePnPRansac(pts3d[inds, :],
        #                                           frame.kp_arr[inds_frame, :].astype(np.float64),
        #                                           self.cam_mat, self.dist_coefs,
        #                                           flags=cv2.SOLVEPNP_EPNP)

        if not is_ok or len(inliers) < 15:
            return False

        inliers = inliers.flatten()
        inds_frame = np.asarray(inds_frame)
        ids_matching_mps = np.asarray(ids_matching_mps)
        inds = np.asarray(inds)
        ids_matching_kfs = np.asarray(ids_matching_kfs)
        self.mps_visible.update(ids_matching_mps[inds[inliers]])

        frame.des2mp = -np.ones((frame.kp_arr.shape[0]), dtype=int)
        with self.cg.lock_mps:
            for i_feat, id_mp in zip(inds_frame[inliers], ids_matching_mps[inds[inliers]]):
                frame.des2mp[i_feat] = id_mp
                self.cg.mps[id_mp].num_frames_found_increment()

        # matched frame features and matching mappoints
        with self.cg.lock_kfs:
            for id_new_kf_ref, _ in Counter(ids_matching_kfs[inds[inliers]]).most_common(100):
                if id_new_kf_ref in self.cg.kfs:
                    self.kf_ref = self.cg.kfs[id_new_kf_ref]
                    self.kf_ref.is_kf_ref(True)

        # R = cv2.Rodrigues(R)[0]
        frame.setPose(R, t)

        return True

    def _ok_as_new_keyframe(self, frame, r):
        if self.t_from_last_kf > num_frames_from_last_kf and r < 0.9 and frame.des.shape[0] >= min_kp_in_frame and \
                (self.t_from_last_relocation > num_frames_from_last_relocation or self.t_absolute <= num_frames_from_last_relocation):
            self.t_from_last_kf = 0
            return True
        return False

    def update(self, frame, kf_queue):
        self.mps_visible.clear()
        self.t_absolute += 1
        self.t_from_last_kf += 1
        self.t_from_last_relocation += 1

        if self.state == state_map_init:
            if len(frame.des) >= min_kp_in_frame:
                R, t = np.eye(3), np.zeros(3)
                frame.setPose(R, t)
                kf = self.cg.add_kf(frame)
                self.t_from_last_kf = 0
                self.last_frame = frame
                self.kf_ref = kf
                self.state = state_ok
                return R, t

        if self.state == state_lost:
            id_kf, score = self.dbow.query(frame)
            if score is not None:
                print("score", score)
                if score >= dbow_tresh:
                    self.kf_ref = self.cg.kfs[id_kf]
                    self.kf_ref.is_kf_ref(True)
                    self.state = state_ok

        if self.state in (state_ok, state_relocated):
            r = 0.0
            is_ok = False
            if self.state == state_ok:
                is_ok = self._track_with_motion(frame)

            if not is_ok:
                is_ok, r = self._track_wrt_refkf(frame, flag_use_pnp=state_relocated == self.state)
                if not is_ok:
                    self.kf_ref.is_kf_ref(False)
                    self.state = state_lost

            if is_ok:
                is_ok = self._track_local_map(frame)
                if not is_ok:
                    self.kf_ref.is_kf_ref(False)
                    self.state = state_lost

            if is_ok:
                if self._ok_as_new_keyframe(frame, r):
                    kf = self.cg.add_kf(frame)
                    kf.is_kf_ref(True)
                    self.kf_ref = kf
                    kf_queue.put(kf.id)
                elif self.t_from_last_kf > num_frames_from_last_kf:
                    kf_queue.put(None)

        if self.state == state_ok:
            self.last_frame = frame
        else:
            self.last_frame = None

        if len(self.cg.kfs) == 0 or len(self.cg.mps) == 0:
            self.state = state_map_init

        return frame.R, frame.t

