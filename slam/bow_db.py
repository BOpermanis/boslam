import numpy as np
import cv2
from pprint import pprint

from camera import Frame
from slam.nodes import KeyFrame

from typing import Dict, Set, Tuple, List
# pprint([a for a in dir(cv2) if "pnp" in a.lower()])
# print(cv2.SOLVEPNP_EPNP)
# exit()
import pyDBoW3 as bow


class Dbow():
    def __init__(self, *args, **kwargs):
        self.db = bow.Database()
        self.voc = bow.Vocabulary()
        self.voc.load("/DBow3/orbvoc.dbow3")
        self.db.setVocabulary(self.voc, True, 2)
        self.entry2frame = Dict[int, int]
        self.frame2entry = Dict[int, int]

    def add(self, frame: Frame):
        if frame.id not in self.frame2entry:
            i = self.db.add(frame.des)
            self.entry2frame[i] = frame.id
            self.frame2entry[frame.id] = i

    def query(self, frame: Frame, subset_inds=[]):
        return db.query(frame.des, subset_inds, 1, -1)[0]

    def compareFrames(self, frame1: Frame, frame2: Frame):
        if frame1.id not in self.frame2entry:
            self.add(frame1)

        if frame2.id not in self.frame2entry:
            self.add(frame2)

        i1 = self.frame2entry[frame1.id]
        i2 = self.frame2entry[frame2.id]
        return self.db.compare_bows(i1, i2)




if __name__ == "__main__":
    db = Dbow()
