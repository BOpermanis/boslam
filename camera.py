import cv2
import numpy as np
import pyrealsense2 as rs
from utils import normalize_t_shape, int2orb

class Frame:
    cnt = 0
    def __init__(self, rgb_frame, cloud, kp=None, des=None, cloud_kp=None, kp_arr=None):
        self.rgb_frame = rgb_frame
        self.cloud = cloud
        # self.kp = kp
        self.des = des
        self.cloud_kp = cloud_kp
        self.kp_arr = kp_arr

        self.id = Frame.cnt
        Frame.cnt += 1

        # global transformation
        self.R = None
        self.t = None
        self.des2mp = None
        self.flag_global_set = False
        self.see_vector = np.asarray((0.0, 0.0, 1.0))

    def setPose(self, R, t):
        self.R, self.t = R, normalize_t_shape(t)
        self.see_vector = np.matmul(self.R, self.see_vector)
        self.see_vector /= np.linalg.norm(self.see_vector)
        self.transform2global(R, t)

    def transform2global(self, R, t, prev_cloud_kp=None, new_inds_for_old=None, log=None):
        # assert not self.flag_global_set
        t = normalize_t_shape(t)
        self.cloud_kp = np.matmul(self.cloud_kp, R) + t
        self.flag_global_set = True
        # self.setPose(R, t)

        if log is not None:
            log['3d_point_diff'] = np.average(np.linalg.norm(self.cloud_kp[new_inds_for_old] - prev_cloud_kp, axis=1))

        if prev_cloud_kp is not None and new_inds_for_old is not None:
            self.cloud_kp[new_inds_for_old] = prev_cloud_kp


class RsCamera:
    def __init__(self, flag_return_with_features=0):

        self.orb_params = dict(
            nfeatures=600,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            patchSize=31,
            fastThreshold=20)
        self.flag_return_with_features = flag_return_with_features
        self.width = 640
        self.height = 480

        if self.flag_return_with_features == 1:
            self.feature_extractor = cv2.ORB_create(**self.orb_params)
        if self.flag_return_with_features == 2:
            self.ncol = 6
            self.nrow = 8
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

        # params for BA
        self.fx, self.fy, self.cx, self.cy = self.intr.fx, self.intr.fy, self.intr.ppx, self.intr.ppy
        self.baseline = 0.06
        self.f = (self.fx + self.fy) / 2
        self.principal_point = self.cx, self.cy

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

        des, kp_arr, kp = None, None, None
        if self.flag_return_with_features == 1:
            kp, des = self.feature_extractor.detectAndCompute(frame, mask=(depth_image > 0).astype(np.uint8) * 255)
            kp_arr = np.asarray([tuple(map(int, k.pt)) for k in kp])

        if self.flag_return_with_features == 2:
            _, corners = cv2.findChessboardCorners(frame, (self.ncol, self.nrow), None)
            is_ok = False
            if corners is not None:
                if corners.shape[0] == self.ncol * self.nrow:
                    is_ok = True
                    corners = corners.astype(int)
                    kp_arr = np.asarray(
                        [(i, x, y) for i, (x, y) in enumerate(corners[:, 0, :]) if depth_image[y, x] > 0])
                    des = np.stack([int2orb(_) for _ in kp_arr[:, 0]])
                    kp_arr = kp_arr[:, 1:]

            if not is_ok:
                kp_arr = np.empty((0, 2))
                des = np.empty((0, ))

        cloud = self.convert_depth_frame_to_pointcloud(depth_image, kp_arr)

        if self.flag_return_with_features != 0 and isinstance(cloud, tuple):
            return Frame(frame, cloud[0], kp, des, cloud[1], kp_arr)

        return Frame(frame, cloud)


if __name__ == "__main__":
    # import matplotlib.pyplot as plt

    cap = RsCamera(flag_return_with_features=2)
    i_frame = 0
    while True:
        i_frame += 1
        frame = cap.get()
        # if i_frame > 100:
        #     break
        # plt.scatter(cloud[:, 2], cloud[:, 4])
        # plt.show()
        # exit()
        # print(frame.rgb_frame.shape, frame.rgb_frame.dtype, np.min(frame.rgb_frame), np.max(frame.rgb_frame))
        print(frame.kp_arr.shape)
        # print(frame.des.shape, frame.des.dtype)
        for kp in frame.kp_arr:
            cv2.circle(frame.rgb_frame, tuple(kp), 3, (0, 255, 0))
        cv2.imshow('my webcam', frame.rgb_frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit
