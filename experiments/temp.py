import cv2
import numpy as np
from camera import RsCamera
from pprint import pprint
from sklearn.linear_model import RANSACRegressor
from filterpy.kalman.kalman_filter import KalmanFilter

camera = RsCamera(flag_return_with_features=True)

bf_matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)

kf = KalmanFilter(3, 3)

prev_frame_ob = None
camera_coords = []

while True:

    frame_ob = camera.get()

    if prev_frame_ob is not None:
        # kf.predict()
        if prev_frame_ob.des is not None and frame_ob.des is not None:
            matches = bf_matcher.match(prev_frame_ob.des, frame_ob.des)

            inds_prev, inds = zip(*((_.queryIdx, _.trainIdx) for _ in matches))
            a = np.ascontiguousarray(prev_frame_ob.cloud_kp[inds_prev, :])
            b = np.ascontiguousarray(frame_ob.kp_arr[inds, :].astype(np.float64))

            _, rvec, tvec, inliers = cv2.solvePnPRansac(a, b, camera.cam_mat, camera.distCoeffs)
            # print("len(rvec)", len(rvec))
            # for a in cv2.Rodrigues(rvec):
            #     print(a.shape)
            # exit()
            rvec = cv2.Rodrigues(rvec)[0]
            # print(type(rvec))
            # print(np.asarray(rvec).shape)
            # exit()
            frame_ob.transform2global(rvec, tvec)

        else:
            print("lost")
            camera_coords.clear()
            kf.update(None)
            continue

    else:
        _, rvec, tvec = cv2.solvePnP(np.ascontiguousarray(frame_ob.cloud_kp),
                                     np.ascontiguousarray(frame_ob.kp_arr.astype(np.float64)),
                                     camera.cam_mat, camera.distCoeffs)

    kf.update(tvec)
    camera_coords.append((cv2.Rodrigues(rvec), tvec))

    prev_frame_ob = frame_ob
    a = np.linalg.norm(kf.z.flatten())
    b = np.linalg.norm(tvec.flatten())
    c = np.linalg.norm(kf.x.flatten())
    print(a, b, c)

    # cv2.imshow('my webcam', frame)
    # if cv2.waitKey(1) == 27:
    #     break  # esc to qui