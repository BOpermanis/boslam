import cv2
import numpy as np
from camera import RsCamera
from pprint import pprint
from sklearn.linear_model import RANSACRegressor
from filterpy.kalman.kalman_filter import KalmanFilter
from utils import LOG, Visualizer
import g2o

pprint(dir(g2o))

camera = RsCamera(flag_return_with_features=2)

bf_matcher = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING, crossCheck=True)

prev_frame_ob = None

i_frame = -1

camera_state = np.eye(3), np.zeros((3, 1))

while True:
    i_frame += 1
    frame_ob = camera.get()
