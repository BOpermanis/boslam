import pyDBoW3 as bow
from camera import RsCamera
from pprint import pprint

voc = bow.Vocabulary()
voc.load("/DBow3/orbvoc.dbow3")
db = bow.Database()
db.setVocabulary(voc, True, 0)

camera = RsCamera(flag_return_with_features=True)

frames = []
for _ in range(10):
    frames.append(camera.get())

for frame_ob in frames:
    db.add(frame_ob.des)
    print(voc.transform(frame_ob.des))

for frame_ob in frames:
    result = db.query(frame_ob.des, 1, -1)[0]
    # pprint(dir(result))
    print(result.Id, result.Score, result.nWords)
    # exit()
