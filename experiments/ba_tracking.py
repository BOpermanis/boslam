import cv2
import numpy as np
from camera import RsCamera
from pprint import pprint
from sklearn.linear_model import RANSACRegressor
from filterpy.kalman.kalman_filter import KalmanFilter
from utils import LOG, Visualizer
import g2o

pprint(dir(g2o))

exit()
camera = RsCamera(flag_return_with_features=1)

bf_matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)

prev_frame_ob = None

"""
state coding:
0 - try initialize
1 - track
"""
state = 0

logs = LOG("pnp_two_way_tracking")
vis = Visualizer()

i_frame = -1
num_lost = -1
num_frames_in_this_track = -1

camera_state = np.eye(3), np.zeros((3, 1))

while True:
    i_frame += 1
    frame_ob = camera.get()
