import os
from glob import glob
import pickle
import cv2
import numpy as np

mask_in_dir = '/vision/u/cy3/data/EPIC/sampled/bboxes'
scores_in_dir = '/vision/u/cy3/data/EPIC/sampled/scores'
mask_out_dir = '/vision2/u/bingbin/ORN/masks/sampled_1perClip_reformat_sizeObj'
if not os.path.exists(mask_out_dir):
  os.makedirs(mask_out_dir, exist_ok=True)
# size_file = '/vision2/u/bingbin/ORN/meta/size_dict.pkl'
size_file = '/vision2/u/bingbin/ORN/meta/size_obj2rgb.pkl'

def reformat():
  size_dict = pickle.load(open(size_file, 'rb'))
  pkls = glob(os.path.join(mask_in_dir, '*.pkl'))
  for i,each in enumerate(pkls):
    if i and i%200 == 0:
      print('{} / {}'.format(i, len(pkls)))
    basename = os.path.basename(each)
    pid, vid = basename.split('_')[:2]
    vid = pid+'_'+vid
    mask = pickle.load(open(each, 'rb'))
    segms, boxes = mask['segms'], mask['boxes']
    # add size to segms
    # TODO
    size = size_dict[pid]['obj_hw']
    mask['segms'] = []

    # append scores to boxes
    scores = pickle.load(open(os.path.join(scores_in_dir, basename.replace('.pkl', '_scores.pkl')), 'rb'))
    for fid in range(len(boxes)):
      mask['segms'] += [],
      for cid in range(len(boxes[fid])):
        mask['segms'][fid] += [],
        for bid in range(len(boxes[fid][cid])):
          mask['segms'][fid][cid] += {
            'size':size,
            'counts': boxes[fid][cid][bid],
          },
          boxes[fid][cid][bid] = np.concatenate([boxes[fid][cid][bid], [scores[fid][cid][bid]]])
    with open(os.path.join(mask_out_dir, basename), 'wb') as handle:
      pickle.dump(mask, handle)

def get_file_size():
  """
  one-time calculation -- shouldn't need to call multiple times
  """
  # root = '/vision2/u/bingbin/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train'
  root = '/vision2/u/bingbin/EPIC_KITCHENS_2018/object_detection_images/train'
  size_dict = {}
  for pid_path in glob(os.path.join(root, 'P*')):
    pid = os.path.basename(pid_path)
    for vid in range(10):
      vid_path = os.path.join(pid_path, pid+'_{:02d}'.format(vid))
      print(vid_path)
      frame = os.path.join(vid_path, 'frame_{:010d}.jpg'.format(1))
      if not os.path.exists(frame):
        print(frame)
        continue
      img = cv2.imread(frame)
      size = img.shape[:2]
      size_dict[pid] = size
      break
  with open('/vision2/u/bingbin/ORN/data/size_dict.pkl', 'wb') as handle:
    pickle.dump(size_dict, handle)
  return


def infer_mask_size():
  """
  deprecated
  """
  maxHs, maxWs = [], []
  for i,each in enumerate(glob(os.path.join(mask_in_dir, '*.pkl'))):
    maxH, maxW = 0, 0
    mask = pickle.load(open(each, 'rb'))
    # flatten
    boxes = [box for frame in mask['boxes'] for cls in frame for box in cls]
    for (xmin, ymin, xmax, ymax) in boxes:
      assert(xmin<=xmax), print('xmin:{} / xmax:{}'.format(xmin, xmax))
      assert(ymin<=ymax), print('ymin:{} / ymax:{}'.format(ymin, ymax))
      maxW = max(maxW, xmax)
      maxH = max(maxH, ymax)
    maxHs += maxH,
    maxWs += maxW,

    if i and i%100==0:
      print('iter', i)
      print('#x>224:', len([each for each in maxWs if each>224]))
      print('#y>224:', len([each for each in maxHs if each>224]))
    
      print('#x>256:', len([each for each in maxWs if each>256]))
      print('#y>256:', len([each for each in maxHs if each>256]))
      print()

  # print('#x>1:', len([_ for each in maxWs if each>224]))
  # print('#y>224:', len([_ for each in maxHs if each>224]))


if __name__ == "__main__":
  reformat()
