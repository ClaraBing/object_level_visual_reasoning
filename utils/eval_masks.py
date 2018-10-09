import os
from glob import glob
import pickle
import numpy as np
from collections import Counter
import pdb

coco_dict = {}

mask_dir = '/vision2/u/bingbin/ORN/masks/sampled_1perClip_reformat'
ts2info = pickle.load(open('/vision2/u/bingbin/ORN/meta/ts2info.pkl', 'rb'))
fpkls = list(ts2info.keys())

def check_gt_obj_appearance():
  """
  Check the percentage of GT obj appearance at frame / clip level
  """
  # update the following values in case the program pauses halfway
  n_past_iter = 0
  frame_match = 0
  clip_match = 0

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


def check_obj_consistency():
  """
  Check whether obj occurrence is consistent across frames.
  Metrics:
  - object IoU
  - avg (# obj in clip) / (# obj in frame)
  - counts & percent of obj occuring in x frames (x = 1,2,3,4)
  """
  ious = []
  clip_over_frame = []
  occ_counter = Counter({1:0, 2:0, 3:0, 4:0})

  for i,fpkl in enumerate(fpkls):
    if i and i%100 == 0:
      print('avg IOU:{:.3f} / avg clip over frame:{:.3f}'.format(np.mean(ious), np.mean(clip_over_frame)))
      n_objs = sum(occ_counter.values())
      print('occ_counter (out of {} objs):'.format(n_objs))
      for n_occ, n_occ_cnt in occ_counter.items():
        print('{}: {} ({:.3f})'.format(n_occ, n_occ_cnt, n_occ_cnt/n_objs))
      print()

    fpkl_full = os.path.join(mask_dir, fpkl)
    boxes = pickle.load(open(fpkl_full, 'rb'))['boxes']

    n_frame_objs = []
    intersect = set()
    union = set()
    obj_occ = []
    for fid in range(4):
      curr_objs = [cid for cid in range(353) if boxes[fid][cid]!=[]] 
      obj_occ += curr_objs
      if len(intersect)==0:
        intersect = set(curr_objs)
      else:
        intersect = intersect.intersection(curr_objs)
      union = union.union(curr_objs)
      n_frame_objs += len(curr_objs),
    ious += len(intersect) / len(union),
    clip_over_frame += [len(union)/each for each in n_frame_objs]
    for k,v in Counter(obj_occ).items():
      occ_counter.update({v:1})

  print('avg IOU:{:.3f} / avg clip over frame:{:.3f}'.format(np.mean(ious), np.mean(clip_over_frame)))

  n_objs = sum(occ_counter.values())
  print('occ_counter (out of {} objs):'.format(n_objs))
  for n_occ, n_occ_cnt in occ_counter.items():
    print('{}: {} ({:.3f})'.format(n_occ, n_occ_cnt, n_occ_cnt/n_objs))


if __name__ == '__main__':
  check_obj_consistency()
