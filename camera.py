import cv2
import numpy as np
import pyrealsense2 as rs


class RsCamera:

    def __init__(self, flag_return_with_features=False):

        self.flag_return_with_features = flag_return_with_features
        if self.flag_return_with_features:
            self.feature_extractor = cv2.ORB_create()
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg = self.pipeline.start(config)
        profile = cfg.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
        self.intr = profile.as_video_stream_profile().get_intrinsics()
        self.u = None
        self.v = None
        self.cam_mat = np.asarray([
            self.intr.fx, 0, self.intr.ppx,
            0, self.intr.fy, self.intr.ppy,
            0, 0, 1
        ])

    def convert_depth_frame_to_pointcloud(self, depth_image):

        if self.u is None:
            height, width = depth_image.shape
            self.u, self.v = np.meshgrid(
                np.linspace(0, width - 1, width),
                np.linspace(0, height - 1, height))
            self.u = self.u.flatten()
            self.v = self.v.flatten()

            self.u = (self.u - self.intr.ppx) / self.intr.fx
            self.v = (self.v - self.intr.ppy) / self.intr.fy

        z = depth_image.flatten() / 1000
        x = np.multiply(self.u, z)
        y = np.multiply(self.v, z)
        mask = np.nonzero(z)

        return np.stack([self.u[mask], self.v[mask], x[mask], y[mask], z[mask]], axis=1)

    def get(self):

        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())

        if self.flag_return_with_features:
            kp, des = self.feature_extractor.detectAndCompute(frame, mask=(depth_image > 0).astype(np.uint8))

        cloud = self.convert_depth_frame_to_pointcloud(depth_image)

        if self.flag_return_with_features:
            return frame, cloud, kp, des

        return frame, cloud

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

