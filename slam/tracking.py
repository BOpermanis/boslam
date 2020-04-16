class Tracker:

    def __init__(self):
        pass

    def update(self, frame_ob, cam):
        state = 0
    n_pose = 0
    n_point = 0

    i_frame = -1
    dict_featid2nodeid = {}

    while True:
        i_frame += 1
        frame_ob = cam.get()
        if frame_ob.kp_arr is None:
            print('lost')
            continue

        if state == 1:
            if prev_frame_ob.des is not None and frame_ob.des is not None:
                # matches = bf_matcher.match(prev_frame_ob.des, frame_ob.des)
                matches = match_indices(prev_frame_ob.des, frame_ob.des)

                if len(matches) < 10:
                    state = 0
                    continue

                # inds_prev, inds = zip(*((_.queryIdx, _.trainIdx) for _ in matches))
                inds_prev, inds = zip(*matches)

                is_ok, rvec, t, inliers = cv2.solvePnPRansac(
                    prev_frame_ob.cloud_kp[inds_prev, :],
                    frame_ob.kp_arr[inds, :].astype(np.float64),
                    cam.cam_mat, cam.distCoeffs)

                if not is_ok:
                    state = 0
                    continue

                R = cv2.Rodrigues(rvec)[0]

                inds_prev = np.asarray(inds_prev)
                inds = np.asarray(inds)

                prev_cloud_kp = prev_frame_ob.cloud_kp[inds_prev[inliers], :]
                new_inds_for_old = inds[inliers]

                frame_ob.transform2global(R, t, prev_cloud_kp, new_inds_for_old)

                set_points_not_matched = set(list(range(frame_ob.kp_arr.shape[0]))).difference(inds)

                ba.add_pose(n_pose, R, t[:, 0], cam, fixed=False)

                dict_featid2nodeid_new = {}
                for i_prev, i in zip(inds_prev, inds):
                    ba.add_edge(dict_featid2nodeid[i_prev], n_pose, frame_ob.kp_arr[i, :])
                    dict_featid2nodeid_new[i] = dict_featid2nodeid[i_prev]

                for i in set_points_not_matched:
                    dict_featid2nodeid_new[i] = n_point
                    ba.add_point(n_point, frame_ob.cloud_kp[i, :])
                    n_point += 1

                dict_featid2nodeid = dict_featid2nodeid_new
                n_pose += 1
            else:
                state = 0
                continue

        else:
            R, t = np.eye(3), np.zeros(3)
            ba.add_pose(n_pose, R, t, cam, fixed=True)
            for i in range(frame_ob.cloud_kp.shape[0]):
                ba.add_point(n_point, frame_ob.cloud_kp[i, :])
                dict_featid2nodeid[i] = n_point
                ba.add_edge(n_point, n_pose, frame_ob.kp_arr[i, :])
                n_point += 1

            n_pose += 1
            state = 1

        prev_frame_ob = frame_ob

        # cv2.imshow('my webcam', frame_ob.rgb_frame)
        # if cv2.waitKey(1) == 27:
        #     break  # esc to qui
        time.sleep(0.05)
        if i_frame > 100:
            break

