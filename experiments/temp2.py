import numpy as np
import cv2
World = np.array([[-0.5, -0.5,  3. ],
               [ 0.5, -0.5,  3. ],
               [ 0.5,  0.5,  3. ],
               [-0.5,  0. ,  3. ]])
keyPoints = np.array([[ 279.03286469,  139.80463604,    1.        ],
                     [ 465.40665724,  136.70519839,    1.        ],
                     [ 465.40665724,  325.1505936 ,    1.        ],
                     [ 279.03286469,  230.927896  ,    1.        ]])

objectPoints = World
imagePoints = keyPoints[:,:2] # <--- THIS SLICE IS A PROBLEM CAUSER!!!
# cv2.solvePnP(objectPoints, imagePoints, np.eye(3), np.zeros(5))


# imagePoints = np.ascontiguousarray(keyPoints[:,:2]).reshape((4,1,2)) # Now OpenCV is HAPPY!
print(objectPoints.dtype)
print(imagePoints.dtype)
objectPoints = objectPoints.astype(np.int32)
retval, rvec, tvec = cv2.solvePnP(np.ascontiguousarray(objectPoints), np.ascontiguousarray(imagePoints), np.eye(3), np.zeros(5))