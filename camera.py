import cv2
import numpy as np
import pyrealsense2 as rs


class Frame:
    def __init__(self, rgb_frame, cloud, kp=None, des=None, cloud_kp=None, kp_arr=None):
        self.rgb_frame = rgb_frame
        self.cloud = cloud
        self.kp = kp
        self.des = des
        self.cloud_kp = cloud_kp
        self.kp_arr = kp_arr

    def transform2global(self, R, t, prev_cloud_kp, new_inds_for_old, log=None):

        # R, t = np.eye(3), np.zeros((3, 1))
        # print(R.shape, R.T.shape)
        # a1 = (np.matmul(R.T, self.cloud_kp.T).T + t[:, 0])[new_inds_for_old]
        # a2 = (np.matmul(R, self.cloud_kp.T).T + t[:, 0])[new_inds_for_old]
        # a3 = (np.matmul(self.cloud_kp, R) + t[:, 0])[new_inds_for_old]
        # a4 = (np.matmul(self.cloud_kp, R.T) + t[:, 0])[new_inds_for_old]
        # b = prev_cloud_kp
        # c1 = np.average(np.linalg.norm(a1 - b, axis=1))
        # c2 = np.average(np.linalg.norm(a2 - b, axis=1))
        # c3 = np.average(np.linalg.norm(a3 - b, axis=1))
        # c4 = np.average(np.linalg.norm(a4 - b, axis=1))
        # print("reproj err 3d", c1, c2, c3, c4)
        # exit()
        self.cloud_kp = np.matmul(R.T, self.cloud_kp.T).T + t[:, 0]
        if log is not None:
            log['3d_point_diff'] = np.average(np.linalg.norm(self.cloud_kp[new_inds_for_old] - prev_cloud_kp, axis=1))
        self.cloud_kp[new_inds_for_old] = prev_cloud_kp


class RsCamera:

    def __init__(self, flag_return_with_features=False):

        self.flag_return_with_features = flag_return_with_features
        self.width = 640
        self.height = 480

        if self.flag_return_with_features:
            self.feature_extractor = cv2.ORB_create()
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)
        cfg = self.pipeline.start(config)
        profile = cfg.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
        self.intr = profile.as_video_stream_profile().get_intrinsics()
        self.u = None
        self.v = None
        self.cam_mat = np.asarray([
            [self.intr.fx, 0, self.intr.ppx],
            [0, self.intr.fy, self.intr.ppy],
            [0, 0, 1]
        ])
        self.distCoeffs = np.zeros((8, 1), dtype=np.float32)

    def convert_depth_frame_to_pointcloud(self, depth_image, kp_arr=None):

        if self.u is None:
            height, width = depth_image.shape
            self.u, self.v = np.meshgrid(
                np.linspace(0, width - 1, width, dtype=np.int16),
                np.linspace(0, height - 1, height, dtype=np.int16))
            self.u = self.u.flatten()
            self.v = self.v.flatten()

            self.x = (self.u - self.intr.ppx) / self.intr.fx
            self.y = (self.v - self.intr.ppy) / self.intr.fy

        z = depth_image.flatten() / 1000
        x = np.multiply(self.x, z)
        y = np.multiply(self.y, z)
        mask = np.nonzero(z)

        points3d_all = np.stack([x, y, z], axis=1)

        if kp_arr is not None:
            if len(kp_arr) == 0:
                return points3d_all[mask], []
            # set_kp_arr = {*[tuple(_) for _ in kp_arr]}
            # inds0 = np.where(np.asarray([tuple(a) in set_kp_arr for a in zip(self.u, self.v)]))[0]
            # print(inds0.shape, len(set_kp_arr))
            # inds1 = kp_arr[:, 0] * self.width + kp_arr[:, 1]
            # inds2 = kp_arr[:, 0] * self.height + kp_arr[:, 1]
            inds_kp = kp_arr[:, 1] * self.width + kp_arr[:, 0]
            # inds4 = kp_arr[:, 1] * self.height + kp_arr[:, 0]
            #
            # print(set(inds0) == set(inds1))
            # print(set(inds0) == set(inds2))
            # print(set(inds0) == set(inds_kp))
            # print(set(inds0) == set(inds4))
            # exit()
            # mask_kps = None
            return points3d_all[mask], points3d_all[inds_kp]

        return points3d_all[mask, :]

    def get(self):

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())

        kp_arr = None
        if self.flag_return_with_features:
            kp, des = self.feature_extractor.detectAndCompute(frame, mask=(depth_image > 0).astype(np.uint8))
            kp_arr = np.asarray([tuple(map(int, k.pt)) for k in kp])

        cloud = self.convert_depth_frame_to_pointcloud(depth_image, kp_arr)

        if self.flag_return_with_features:
            return Frame(frame, cloud[0], kp, des, cloud[1], kp_arr)

        return Frame(frame, cloud)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cap = RsCamera()
    while True:
        frame, cloud = cap.get()
        print(frame.shape, cloud.shape)
        plt.scatter(cloud[:, 2], cloud[:, 4])
        plt.show()
        exit()
        cv2.imshow('my webcam', frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

