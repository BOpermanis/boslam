import numpy as np
from multiprocessing import Queue
from pyDBoW3 import Database

from config import dbow_tresh
from slam.covisibility_graph import CovisibilityGraph

class GlobalMapManager:
    def __init__(self, cg: CovisibilityGraph, dbow: Database):
        self.dbow = dbow
        self.cg = cg

    def _loop_candidate_detection(self, kf):
        # TODO
        pass

    def _similiarity_transformation(self) -> (np.ndarray, np.ndarray):
        # TODO return R, t that transofms start of the loop with the end
        pass

    def _loop_fusion(self, R, t):
        """
        finds duplicated mappoints and fses them together
        :return:
        """
        pass

    def _essential_graph_optimization(self):
        pass

    def update(self, bow_queue: Queue):
        kf = bow_queue.get()

        self._loop_candidate_detection(kf)
        R, t = self._similiarity_transformation()

        self._loop_fusion(R, t)

        self._essential_graph_optimization()


