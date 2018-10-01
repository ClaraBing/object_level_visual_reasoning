import os
from glob import glob
import pickle

mask_in_dir = '/vision/u/cy3/data/EPIC/sampled/bboxes'
scores_in_dir = 'vision/u/cy3/data/EPIC/sampled/scores'
mask_out_dir = '/vision2/u/bingbin/ORN/masks/sampled_1perClip'

def reformat():
  for each in glob(os.path.join(mask_in_dir, '*.pkl')):
    basename = os.path.basename(each)
    mask = pickle.load(open(each, 'rb'))
    segms, boxes = mask['segms'], mask['boxes']
    # add size to segms
    # TODO
  
    # append scores to boxes
    scores = pickle.load(open(os.path.join(scores_in_dir, basename), 'rb'))
    for fid in range(len(boxes)):
      for cid in range(len(boxes[fid])):
        if boxes[fid][cid]:
          for bid in range(len(boxes[fid][cid])):
            boxes[fid][cid][bid] = np.concatenate(boxes[fid][cid][bid], [scores[fid][cid][bid]])
    with open(os.path.join(mask_out_dir, basename), 'wb') as handle:
      pickle.dump(mask, handle)

def infer_mask_size():
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
  infer_mask_size()
