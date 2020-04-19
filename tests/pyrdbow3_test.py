import pyDBoW3 as bow
from camera import RsCamera
from pprint import pprint
import numpy as np

voc = bow.Vocabulary()
voc.load("/DBow3/orbvoc.dbow3")
db = bow.Database()
db.setVocabulary(voc, True, 2)
# db.setVocabulary(voc, True, 2)

camera = RsCamera(flag_return_with_features=True)

frames = []
for _ in range(10):
    frames.append(camera.get())

for frame_ob in frames:
    db.add(frame_ob.des)
    ids = [voc.feat_id(np.expand_dims(f, 0)) for f in frame_ob.des]
    wts = [voc.id_weight(i) for i in ids]
    print("------------------------------------")
    print(np.min(ids), np.max(ids))
    print(np.min(wts), np.max(wts))

print("------------------------------------")
print("------------------------------------")
for frame_ob in frames:
    result = db.query(frame_ob.des, 1, -1)[0]
    pprint(dir(result))
    print("result.Id", result.Id)
    print("result.Score", result.Score)
    print("result.nWords", result.nWords)
    print("result.sumCommonWi", result.sumCommonWi)
    exit()
