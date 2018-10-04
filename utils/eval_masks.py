import os
from glob import glob
import pickle

coco_dict = {}

mask_dir = '/vision2/u/bingbin/ORN/masks/sampled_1perClip_reformat'
ts2info = pickle.load(open('/vision2/u/bingbin/ORN/data/ts2info.pkl', 'rb'))
fpkls = list(ts2info.keys())

frame_match = 0
clip_match = 0
add1_match = 0
sub1_match = 0
for i, fpkl in enumerate(fpkls):
  if i and i%100 == 0:
    print('frame_match: {}/{} [{:.3f}] & clip_match: {}/{} [{:.3f}] & add_one:{} & sub_one:{}'.format(
        frame_match, 4*i, frame_match/(4*i),
        clip_match, i, clip_match/i,
        add1_match / i, sub1_match / i))
  gt_cls = ts2info[fpkl]['noun_cls']
  boxes = pickle.load(open(os.path.join(mask_dir, fpkl), 'rb'))['boxes']
  curr_match = 0
  curr_add1 = 0
  curr_sub1 = 0
  for fid in range(4):
    curr_match += (boxes[fid][gt_cls+1]!=[])
  frame_match += curr_match
  clip_match += curr_match > 0
