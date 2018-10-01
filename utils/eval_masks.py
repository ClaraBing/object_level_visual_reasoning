import os
from glob import glob
import pikcle

coco_dict = {}

ts2info = pickle.load(open('/vision2/u/bingbin/ORN/data/ts2info.pkl', 'rb'))
fpkls = list(ts2info.keys())

for fpkl in fpkls:
  noun_cls = ts2info[fpkl]['noun_cls']
