import cv2
import numpy as np
import pyrealsense2 as rs
from pprint import pprint

# pprint(dir(rs))
# print(rs.intrinsics)
# exit()
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    frame = np.asanyarray(color_frame.get_data())

    depth_image = (np.log(1 + depth_image) * 30).astype(np.uint8)

    depth_image = np.expand_dims(depth_image, axis=2)
    # depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2RGB)
    depth_image = np.concatenate([depth_image] * 3, axis=2)
    depth_frame = np.reshape(depth_image, (480, 640, 3))
    frame = np.concatenate([
        frame, depth_frame
    ], axis=1)

    cv2.imshow('my webcam', frame)
    if cv2.waitKey(1) == 27:
        break  # esc to quit
