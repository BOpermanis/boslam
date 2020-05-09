import cv2
import numpy as np
# from multiprocessing import Process, Queue
from queue import Queue
from threading import Thread

from camera import RsCamera
from slam.tracking import Tracker
from slam.local_mapping import LocalMapManager
from slam.covisibility_graph import CovisibilityGraph
from slam.bow_db import Dbow

camera = RsCamera(flag_return_with_features=1)
dbow = Dbow()
cg = CovisibilityGraph(dbow, camera)

tracker = Tracker(cg, dbow, camera)
local_map_manager = LocalMapManager(cg, dbow, camera)

kf_queue = Queue()
bow_queue = Queue()

def worker_local_mapper():
    while True:
        local_map_manager.update(kf_queue, bow_queue)

threads = [
    Thread(target=worker_local_mapper)
]

for thread in threads:
    thread.daemon = True
    thread.start()

i_frame = 0
frames = []
while True:
    i_frame += 1
    frame = camera.get()
    print("{}) tracker state = {}, num_ks = {}, num_mps = {}".format(i_frame, tracker.state, tracker.cg.num_kfs(), tracker.cg.num_mps()))
    tracker.update(frame, kf_queue)
    # frames.append(frame)
    cv2.imshow('my webcam', frame.rgb_frame)
    if cv2.waitKey(1) == 27 or len(frames) == 100:
        break  # esc to quit

# import pickle
# with open("/home/slam_data/data_sets/realsense_frames.pickle", "wb") as conn:
#     pickle.dump(frames, conn)