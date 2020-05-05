import numpy as np
import cv2
from pprint import pprint
# from multiprocessing import Lock
# from threading import Lock
from utils import Lock
# from typing import Dict

from camera import Frame
import pyDBoW3 as bow


class Dbow():
    def __init__(self, *args, **kwargs):
        self.db = bow.Database()
        self.voc = bow.Vocabulary()
        self.voc.load("/DBow3/orbvoc.dbow3")
        self.db.setVocabulary(self.voc, True, 2)
        self.entry2frame = {} #Dict[int, int]
        self.frame2entry = {} #Dict[int, int]
        self.lock = Lock()

    def add(self, frame):
        if frame.id not in self.frame2entry:
            with self.lock:
                i = self.db.add(frame.des)
                self.entry2frame[i] = frame.id
                self.frame2entry[frame.id] = i

    def erase(self, i):
        with self.lock:
            j = self.frame2entry[i]
            self.db.erase(j)
            del self.frame2entry[i]
            del self.entry2frame[j]

    def query(self, frame: Frame, subset_inds=[]):
        with self.lock:
            qs = self.db.query(frame.des, subset_inds, 1, -1)
            if len(qs) > 0:
                return self.entry2frame[qs[0].Id], qs[0].Score
            return None, None

    def distance(self, f1, f2):
        return self.voc.distance(f1, f2)

    def compareFrames(self, frame1: Frame, frame2: Frame):
        with self.lock:
            if frame1.id not in self.frame2entry:
                self.add(frame1)

            if frame2.id not in self.frame2entry:
                self.add(frame2)

            i1 = self.frame2entry[frame1.id]
            i2 = self.frame2entry[frame2.id]
            return self.db.compare_bows(i1, i2)


if __name__ == "__main__":
    db = Dbow()
