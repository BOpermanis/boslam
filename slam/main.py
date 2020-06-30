import cv2
import numpy as np
# from multiprocessing import Process, Queue
from queue import Queue
from threading import Thread
from pprint import pprint
from time import sleep
from copy import deepcopy

from camera import RsCamera
from slam.tracking import Tracker
from slam.local_mapping import LocalMapManager
from slam.covisibility_graph import CovisibilityGraph
from slam.bow_db import Dbow
from visualization.slam_viewer import MapViewer
import config
from utils import Plotter, R2angles


def main(flag_use_camera=True, flag_visualize=False, flag_plot=False):
    if flag_use_camera:
        camera = RsCamera(flag_return_with_features=1)
    else:
        camera = config
    dbow = Dbow()
    cg = CovisibilityGraph(dbow, camera)

    tracker = Tracker(cg, dbow, camera)
    local_map_manager = LocalMapManager(cg, dbow, camera)

    if flag_visualize:
        map_viewer = MapViewer(tracker, cg)

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
    if flag_use_camera:
        frames = []
    else:
        import pickle
        with open("/home/slam_data/data_sets/realsense_frames.pickle", "rb") as conn:
            frames = pickle.load(conn)

    if flag_plot:
        plotter = Plotter()

    while True:
        i_frame += 1
        if flag_use_camera:
            frame = camera.get()
        else:
            if len(frames) == 0:
                break
            frame = frames.pop()
        print("{}) tracker state = {}, num_ks = {}, num_mps = {}".format(i_frame, tracker.state, tracker.cg.num_kfs(),
                                                                         tracker.cg.num_mps()))
        # frames.append(deepcopy(frame))

        R, t = tracker.update(frame, kf_queue)

        # if R is not None:
        #     print(np.linalg.norm(t))
        # stats = cg.get_stats()
        # print(stats['mps_avg_vec'])
        # print(stats['mps_min_id'], stats['mps_max_id'])
        # print(stats['kfs_min_id'], stats['kfs_max_id'])
        # sleep(4)
        if flag_visualize:
            map_viewer.update(frame, R, t)
        plot = None
        if flag_plot:
            if t is not None:
                angles = R2angles(R)
                plot = plotter.update(t[0], flag_map_reset=tracker.state==0)
                # plot = plotter.update(angles[0])

        for kp in frame.kp_arr:
            cv2.circle(frame.rgb_frame, tuple(kp), 3, (0, 255, 0))
        frame_arr = frame.rgb_frame
        if flag_plot:
            if plot is not None:
                frame_arr = np.concatenate([frame_arr, plot], axis=1)
        cv2.imshow('my webcam', frame_arr)
        if cv2.waitKey(1) == 27: # or len(frames) == 100:
            # import pickle
            # with open("/home/slam_data/data_sets/realsense_frames_chessboard.pickle", "wb") as conn:
            #     pickle.dump(frames, conn)
            break  # esc to quit
        # sleep(0.1)


if __name__ == "__main__":
    main(flag_use_camera=True, flag_visualize=False, flag_plot=True)
