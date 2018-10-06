import os
from glob import glob
import pickle

coco_dict = {}

mask_dir = '/vision2/u/bingbin/ORN/masks/sampled_1perClip_reformat'
ts2info = pickle.load(open('/vision2/u/bingbin/ORN/meta/ts2info.pkl', 'rb'))
fpkls = list(ts2info.keys())

# paused: frame_match: 14298/36400 [0.393] & clip_match: 4866/9100 [0.535]

n_past_iter = 9100
frame_match = 14298
clip_match = 4866
for i, fpkl in enumerate(fpkls):
  if i <= n_past_iter:
    continue
  if i and i%100 == 0:
    print('frame_match: {}/{} [{:.3f}] & clip_match: {}/{} [{:.3f}]'.format(
        frame_match, 4*i, frame_match/(4*i),
        clip_match, i, clip_match/i))
  gt_cls = ts2info[fpkl]['noun_cls']
  fpkl_full = os.path.join(mask_dir, fpkl)
  if not os.path.exists(fpkl_full):
    continue
  boxes = pickle.load(open(fpkl_full, 'rb'))['boxes']
  curr_match = 0
  for fid in range(4):
    curr_match += (boxes[fid][gt_cls+1]!=[])
  frame_match += curr_match
  clip_match += curr_match > 0
