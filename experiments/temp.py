import cv2
import numpy as np
from camera import RsCamera

camera = RsCamera(flag_return_with_features=True)

while True:

    frame, cloud, kp, des = camera.get()

    (_, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coefs)

    cv2.imshow('my webcam', frame)
    if cv2.waitKey(1) == 27:
        break  # esc to qui