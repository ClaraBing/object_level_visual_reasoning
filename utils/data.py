import os
import csv
import numpy as np
import pickle

def ts2info(fannot, frame_root, mask_root):
  """
  build a map (saved in a pkl file)
  from start/end timestamps to a list of frames
  """
  map_ts2info = {}

  frame_format = 'frame_{:010d}.jpg'
  mask_format = '{:s}_{:s}_{:s}.pkl'
  
  reader = csv.reader(open(fannot, 'r'))
  data = [line for line in reader]
  # header:
  #   0-3: uid,participant_id,video_id,narration,
  #   4-7: start_timestamp,stop_timestamp,start_frame,stop_frame,
  #  8-13: verb,verb_class,noun,noun_class,all_nouns,all_noun_classes
  header = data[0]
  annots = data[1:]

  for annot in annots:
    pid, uid = annot[1], annot[2]
    ts_start, ts_end, frame_start, frame_end = annot[4:8]
    verb_cls = int(annot[9])
    noun_cls = int(annot[11])
    all_noun_classes = annot[13].replace('[', '').replace(']', '').replace(' ', '')
    all_noun_classes = [int(each) for each in all_noun_classes.split(',')]
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    frames = []
    for i in range(frame_start, frame_end+1):
      frames += os.path.join(frame_root, pid, uid, frame_format.format(i)),
    mask_name = mask_format.format(uid, ts_start, ts_end)
    mask_path = os.path.join(mask_root, mask_name) 
    if os.path.exists(mask_path):
      map_ts2info[mask_name] = {
        'frames':frames,
        'verb_cls':verb_cls,
        'len':len(frames),
        'noun_cls':noun_cls,
        'all_noun_classes':all_noun_classes}

  return map_ts2info

def ts2info_wrapper():
  fannot = '/sailhome/bingbin/VOG/dataset/EPIC/annotations/EPIC_train_action_labels.csv'
  frame_root = '/vision2/u/bingbin/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train'
  mask_root = '/vision2/u/cy3/data/mask-AR/masks_v2_for_AR_100x100_50/train'
  fpkl_out = '/vision2/u/bingbin/ORN/meta/ts2info.pkl'
  if os.path.exists(fpkl_out):
    overwrite = input('File "{:s}" already exists.\nOverwrite? (y/N)'.format(fpkl_out))
    if 'n' in overwrite or 'N' in overwrite:
      return

  map_ts2info = ts2info(fannot, frame_root, mask_root)

  with open(fpkl_out, 'wb') as handle:
    pickle.dump(map_ts2info, handle)

# 8,cupboard,P01,P01_01,028681,"[(386, 290, 226, 1070), (668, 856, 346, 894)]"
# 8,cupboard,P01,P01_01,028711,"[(542, 232, 262, 1102)]"
# 8,cupboard,P01,P01_01,028741,[]


def get_box_parser():
  import re
  rep = {'(':'', ')':'', ' ':'', '[':'', ']':''}
  rep = dict((re.escape(k), v) for k,v in rep.items())
  pattern = re.compile('|'.join(rep.keys()))
  return lambda txt: [int(each) for each in pattern.sub(lambda m:rep[re.escape(m.group(0))], txt).split(',')]

def prep_obj_lookup(fannot, box_parser):
  reader = csv.reader(open(fannot, 'r'))
  data = [line for line in reader]
  header = data[0]

  lookup = {}
  for nid, noun, pid, vid, fid, boxes in data[1:]:
    if boxes == '[]':
      boxes_parsed = []
    else:
      boxes_parsed = [box_parser(each) for each in boxes.split('), (')]
    if vid not in lookup:
      lookup[vid] = {}
    lookup[vid][int(fid)] = {
      'noun_cls': int(nid),
      'noun': noun,
      'boxes': boxes_parsed}
  return lookup

def prep_obj_lookup_wrapper():
  fannot = '/vision2/u/bingbin/EPIC_KITCHENS_2018/annotations/EPIC_train_object_labels.csv'
  fsave = '/vision2/u/bingbin/EPIC_KITCHENS_2018/annotations/obj_lookup.pkl'
  if os.path.exists(fsave):
    overwrite = input('File "{:s}" already exists.\nOverwrite? (y/N)'.format(fsave))
    if 'n' in overwrite or 'N' in overwrite:
      return

  box_parser = get_box_parser()
  lookup = prep_obj_lookup(fannot, box_parser)

  with open(fsave, 'wb') as handle:
    pickle.dump(lookup, handle)

def prep_gt_masks(frame_lookup, box_lookup, size_lookup, save_format):
  boxes = []
  segms = []
  for each in frame_lookup:
    pid = each['participant_id']
    vid = each['video_id']
    ts_start, ts_stop = each['start_timestamp'], each['stop_timestamp']

    out = {'segms':[[] for _ in range(353)], 'boxes':[[] for _ in range(353)]}
    # NOTE: each['clips'][0] since only 1 sampled clip per video
    for i,frame in enumerate(each['clips'][0]):
      fid = int(os.path.basename(frame).split('_')[1].split('.')[0])
      nn_obj_fid = 30 * round(fid/30) + 1
      while nn_obj_fid not in box_lookup[vid]:
        # corner case: out of bound: too many verb frames than obj frames
        nn_obj_fid -= 30
      boxes = box_lookup[vid][nn_obj_fid]
      nid = boxes['noun_cls']
      for box in boxes['boxes']:
        out['segms'][nid] += {
          'size': size_lookup[pid],
          'counts':np.array(box),
        },
        out['boxes'][nid] += np.array(box+[1]),

      with open(save_format.format(vid, ts_start, ts_stop), 'wb') as handle:
        pickle.dump(out, handle)

def prep_gt_masks_wrapper():
  size_file = '/vision2/u/bingbin/ORN/meta/size_dict.pkl'
  frame_lookup_file = '/vision2/u/cy3/data/EPIC/sampled/sample_rand_clips_all_dicts_n1.pkl' 
  box_lookup_file = '/vision2/u/bingbin/EPIC_KITCHENS_2018/annotations/obj_lookup.pkl'
  save_dir = '/vision2/u/bingbin/ORN/masks/sampled_1perClip_GT'
  if os.path.exists(save_dir):
    overwrite = input('Folder "{:s}" already exists.\nOverwrite? (y/N)'.format(save_dir))
    if 'n' in overwrite or 'N' in overwrite:
      print('Skipping prep_gt_masks_wrapper.')
      return

  os.makedirs(save_dir, exist_ok=True)
  save_format = os.path.join(save_dir, "{:s}_{:s}_{:s}.pkl")

  with open(size_file, 'rb') as handle:
    size_lookup = pickle.load(handle)
  with open(frame_lookup_file, 'rb') as handle:
    frame_lookup = pickle.load(handle)
  with open(box_lookup_file, 'rb') as handle:
    box_lookup = pickle.load(handle)

  prep_gt_masks(frame_lookup, box_lookup, size_lookup, save_format)


if __name__ == '__main__':
  ts2info_wrapper()
  prep_obj_lookup_wrapper()
  prep_gt_masks_wrapper()
