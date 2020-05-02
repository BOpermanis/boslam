import pyDBoW3 as bow
from pprint import pprint
import numpy as np
import cv2

voc = bow.Vocabulary()
voc.load("/DBow3/orbvoc.dbow3")
db = bow.Database()
db.setVocabulary(voc, True, 2)

img1 = cv2.imread("/home/slam_data/2-nature.jpg")
img2 = cv2.imread("/home/slam_data/foresttb-l.jpg")
print(img1.shape, img2.shape)

orb = cv2.ORB_create()
kps1, des1 = orb.detectAndCompute(img1, None)
kps2, des2 = orb.detectAndCompute(img2, None)

db.add(des1)
db.add(des2)

print(db.compare_bows(0, 1))
print(db.compare_bows(1, 0))

print(des1.shape)

print(len(db.commonWords(des1, 0)))

exit()
result = db.query(des1, [0, 1], 1, -1)[0]
print("result.Id", result.Id)
print("result.Score", result.Score)
print("result.minScoreInSelection", result.minScoreInSelection)
