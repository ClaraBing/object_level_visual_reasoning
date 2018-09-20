import os
import csv
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
    frame_start = int(frame_start)
    frame_end = int(frame_end)
    frames = []
    for i in range(frame_start, frame_end+1):
      frames += os.path.join(frame_root, pid, uid, frame_format.format(i)),
    mask_name = mask_format.format(uid, ts_start, ts_end)
    mask_path = os.path.join(mask_root, mask_name) 
    if os.path.exists(mask_path):
      map_ts2info[mask_name] = {'frames':frames, 'verb_cls':verb_cls, 'len':len(frames)}

  return map_ts2info

def ts2info_wrapper():
  fannot = '/sailhome/bingbin/VOG/dataset/EPIC/annotations/EPIC_train_action_labels.csv'
  frame_root = '/vision2/u/bingbin/EPIC_KITCHENS_2018/frames_rgb_flow/rgb/train'
  mask_root = '/vision2/u/cy3/data/mask-AR/masks_v2_for_AR_100x100_50/train'
  fpkl_out = '/vision2/u/bingbin/ORN/data/ts2info.pkl'

  map_ts2info = ts2info(fannot, frame_root, mask_root)

  with open(fpkl_out, 'wb') as handle:
    pickle.dump(map_ts2info, handle)

if __name__ == '__main__':
  ts2info_wrapper()
