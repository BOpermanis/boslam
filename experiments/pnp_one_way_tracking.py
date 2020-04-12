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

logs = LOG("temp")
i_frame = -1
while True:
    i_frame += 1
    frame_ob = camera.get()
    log = {}
    if state == 1:
        if prev_frame_ob.des is not None and frame_ob.des is not None:
            matches = bf_matcher.match(prev_frame_ob.des, frame_ob.des)
            log['num_matches'] = len(matches)
            inds_prev, inds = zip(*((_.queryIdx, _.trainIdx) for _ in matches))
            a = np.ascontiguousarray(prev_frame_ob.cloud_kp[inds_prev, :])
            b = np.ascontiguousarray(frame_ob.kp_arr[inds, :].astype(np.float64))

            if a.shape[0] < 10 or b.shape[0] < 0:
                print("lost")
                log['is_lost'] = True
                state = 0
                continue
            is_ok, rvec, tvec, inliers = cv2.solvePnPRansac(a, b, camera.cam_mat, camera.distCoeffs)
            if not is_ok:
                print("lost")
                log['is_lost'] = True
                state = 0
                continue
            log['num_inliers'] = len(inliers)
            log['norm_tvec'] = np.linalg.norm(tvec)
            inds_prev = np.asarray(inds_prev)
            inds = np.asarray(inds)

            prev_cloud_kp = prev_frame_ob.cloud_kp[inds_prev[inliers], :]
            log['old_tvec_norm'] = np.average(np.linalg.norm(prev_cloud_kp, axis=1))
            new_inds_for_old = inds[inliers]
            rvec = cv2.Rodrigues(rvec)[0]
            # frame_ob.cloud_kp[inds] = prev_frame_ob.cloud_kp[inds_prev]
            frame_ob.transform2global(rvec, tvec, prev_cloud_kp, new_inds_for_old, log=log)

        else:
            print("lost")
            log['is_lost'] = True
            state = 0
            continue

    else:
        # _, rvec, tvec = cv2.solvePnP(np.ascontiguousarray(frame_ob.cloud_kp),
        #                              np.ascontiguousarray(frame_ob.kp_arr.astype(np.float64)),
        #                              camera.cam_mat, camera.distCoeffs)
        rvec, tvec = np.eye(3), np.zeros(3)
        log['norm_tvec'] = np.linalg.norm(tvec)
        state = 1
    logs.add(log)

    prev_frame_ob = frame_ob

    print("tvec norm", np.linalg.norm(tvec.flatten()))
    if i_frame > 50:
        logs.save()
        exit()
    # cv2.imshow('my webcam', frame)
    # if cv2.waitKey(1) == 27:
    #     break  # esc to qui