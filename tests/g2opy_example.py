import numpy as np
import g2o

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().set_verbose(True)
        super().optimize(max_iterations)

    def add_pose(self, pose_id, R, t, cam, fixed=False):
        pose_estimate = g2o.SE3Quat(R, t)
        sbacam = g2o.SBACam(pose_estimate.orientation(), pose_estimate.position())
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, measurement, information=np.identity(2), robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()


if __name__ == "__main__":
    from camera import RsCamera
    import cv2
    from PIL import Image
    import time
    from utils import match_indices
    from pprint import pprint

    ba = BundleAdjustment()
    cam = RsCamera(flag_return_with_features=2)

    # bf_matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)

    prev_frame_ob = None

    state = 0
    n_pose = 0
    n_point = 0

    i_frame = -1
    dict_featid2nodeid = {}

    while True:
        i_frame += 1
        frame_ob = cam.get()

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

    # cv2.destroyAllWindows()
    ba.optimize()
    ts = []
    for i_pose in range(n_pose):
        pose = ba.get_pose(i_pose)
        ts.append(pose.position())
        # print(pose.rotation().R, pose.position())
        # pprint(dir(pose.rotation()))
        # exit()

    ts = np.asarray(ts)
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    pca = PCA(n_components=2)
    print(ts.shape)
    ts = pca.fit_transform(ts)
    print(ts.shape)

    plt.plot(ts[:, 0], ts[:, 1])
    plt.show()
    # pose = ba.get_pose(n_pose - 1)
    # print(pose.rotation().R, pose.position())