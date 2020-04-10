import cv2
import numpy as np
from camera import RsCamera
from pprint import pprint

camera = RsCamera(flag_return_with_features=True)

while True:

    frame_ob = camera.get()


    (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, camera.cam_mat, camera.distCoeffs)

    cv2.imshow('my webcam', frame)
    if cv2.waitKey(1) == 27:
        break  # esc to qui