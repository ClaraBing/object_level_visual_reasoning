import os
import numpy as np
import csv
import pickle

annot_root = '/sailhome/bingbin/VOG/dataset/EPIC/annotations'
meta_root = '/vision2/u/bingbin/ORN/meta'
with open('/sailhome/bingbin/tmn/data/wv_map.pkl', 'rb') as handle:
  wv_map = pickle.load(handle)

def get_verbs():
  reader = csv.reader(open(os.path.join(annot_root, 'EPIC_verb_classes.csv'), 'r'))
  # NOTE: the order needs to be preserved
  # since the WVs are gonna be used as classifier weights
  verbs = [line[1] for line in reader][1:]
  return verbs

def prep_weights():
  """
  Set classifier weights using WVs.
  Stand-alone function.
  """
  W = []
  verbs = get_verbs()
  for verb in verbs:
    if verb not in wv_map:
      W += np.zeros([300,]),
    else:
      wv = wv_map[verb]
      W += wv  / np.linalg.norm(wv),

  W = np.array(W)
  np.save(os.path.join(meta_root, 'wv_weights.npy'), W)
  return W

def vis():
  verbs = get_verbs()
  for verb in verbs:
    if verb not in verbs:
      continue
    wv = wv_map[verb]


if __name__ == '__main__':
  prep_weights()
