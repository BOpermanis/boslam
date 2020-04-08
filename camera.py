import cv2
import numpy as np
import pyrealsense2 as rs
from utils import depth_to_3d, convert_depth_frame_to_pointcloud

class RsCamera:

    def __init__(self):
        # Camera calibration and distortion parameters (OpenCV)
        self.fx = 611.528
        self.fy = 611.503
        self.cx = 320.503
        self.cy = 237.288

        self.width = 640
        self.height = 480
        self.scale = 1.0

        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg = self.pipeline.start(config)
        profile = cfg.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
        self.intr = profile.as_video_stream_profile().get_intrinsics()

    def get(self):
        # try:
        #     frames = self.pipeline.wait_for_frames()
        #     depth_frame = frames.get_depth_frame()
        #     color_frame = frames.get_color_frame()
        #     depth_image = np.asanyarray(depth_frame.get_data())
        #     frame = np.asanyarray(color_frame.get_data())
        #
        #     # depth_to_3d(depth_image, )
        #     cloud = convert_depth_frame_to_pointcloud(depth_image, rs.intrinsics)
        #     return True, frame, cloud
        # except Exception as e:
        #     print(e)
        #     return False, None, None
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_image = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())

        # depth_to_3d(depth_image, )
        cloud = convert_depth_frame_to_pointcloud(depth_image, self.intr)
        return True, frame, cloud

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    cap = RsCamera()
    while True:
        is_ok, frame, cloud = cap.get()
        print(frame.shape, cloud.shape)
        plt.scatter(cloud[:, 0], cloud[:, 1])
        plt.show()
        exit()
        cv2.imshow('my webcam', frame)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

