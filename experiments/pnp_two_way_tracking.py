import cv2
import numpy as np
from camera import RsCamera
from pprint import pprint
from sklearn.linear_model import RANSACRegressor
from filterpy.kalman.kalman_filter import KalmanFilter
from utils import LOG

camera = RsCamera(flag_return_with_features=True)

bf_matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)


prev_frame_ob = None

"""
state coding:
0 - try initialize
1 - track
"""
state = 0

logs = LOG("pnp_two_way_tracking")
i_frame = -1
num_lost = -1
num_frames_in_this_track = -1
while True:
    i_frame += 1
    num_frames_in_this_track += 1
    frame_ob = camera.get()
    log = {}

    if state == 1:
        if prev_frame_ob.des is None or frame_ob.des is None:
            state = 0
            continue

        matches = bf_matcher.match(prev_frame_ob.des, frame_ob.des)
        if len(matches) < 10:
            state = 0
            continue

        log['num_matches'] = len(matches)
        inds_prev, inds = zip(*((_.queryIdx, _.trainIdx) for _ in matches))
        a = np.ascontiguousarray(prev_frame_ob.cloud_kp[inds_prev, :])
        b = np.ascontiguousarray(frame_ob.kp_arr[inds, :].astype(np.float64))

        a1 = np.ascontiguousarray(prev_frame_ob.kp_arr[inds_prev, :].astype(np.float64))
        b1 = np.ascontiguousarray(frame_ob.cloud_kp[inds, :])

        is_ok_ab, rvec_ab, tvec_ab, inliers_ab = cv2.solvePnPRansac(a, b, camera.cam_mat, camera.distCoeffs)

        is_ok_ba, rvec_ba, tvec_ba, inliers_ba = cv2.solvePnPRansac(b1, a1, camera.cam_mat, camera.distCoeffs)

        if not is_ok_ab or not is_ok_ba:
            state = 0
            continue

        rvec_ab = cv2.Rodrigues(rvec_ab)[0]
        rvec_ba = cv2.Rodrigues(rvec_ba)[0]

        rvec = (rvec_ab + np.linalg.inv(rvec_ba)) / 2.0
        tvec = (tvec_ab - tvec_ba) / 2.0
        # rvec, tvec = cv2.Rodrigues(rvec_ab)[0], tvec_ab

        inliers = np.asarray(list(set(inliers_ab.flatten()).union(inliers_ba.flatten())))

        log['num_inliers'] = len(inliers_ab)
        log['norm_tvec'] = np.linalg.norm(tvec)

        inds_prev = np.asarray(inds_prev)
        inds = np.asarray(inds)

        prev_cloud_kp = prev_frame_ob.cloud_kp[inds_prev[inliers], :]
        log['old_tvec_norm'] = np.average(np.linalg.norm(prev_cloud_kp, axis=1))
        new_inds_for_old = inds[inliers]
        # frame_ob.cloud_kp[inds] = prev_frame_ob.cloud_kp[inds_prev]
        frame_ob.transform2global(rvec, tvec, prev_cloud_kp, new_inds_for_old, log=log)

    if state == 0:
        num_lost += 1
        num_frames_in_this_track = -1
        logs = LOG("pnp_two_way_tracking")
        rvec, tvec = np.eye(3), np.zeros(3)
        log['norm_tvec'] = np.linalg.norm(tvec)
        state = 1

    logs.add(log)

    prev_frame_ob = frame_ob

    print("{}) tvec norm {}, num_lost {}".format(num_frames_in_this_track, np.linalg.norm(tvec.flatten()), num_lost))
    if num_frames_in_this_track > 200:
        logs.save()
        exit()
    # cv2.imshow('my webcam', frame)
    # if cv2.waitKey(1) == 27:
    #     break  # esc to qui