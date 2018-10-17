import os
import numpy as np
from numpy.linalg import norm
import csv
import pickle
from scipy.spatial.distance import cosine
import pdb

annot_root = '/sailhome/bingbin/VOG/dataset/EPIC/annotations'
meta_root = '/vision2/u/bingbin/ORN/meta'
with open('/sailhome/bingbin/tmn/data/wv_map.pkl', 'rb') as handle:
  wv_map = pickle.load(handle)

def get_verbs():
  reader = csv.reader(open(os.path.join(annot_root, 'EPIC_verb_classes.csv'), 'r'))
  # NOTE: the order needs to be preserved
  # since the WVs are gonna be used as classifier weights / 
  verbs = [line[1] for line in reader][1:]
  return verbs

def get_nouns():
  reader = csv.reader(open(os.path.join(annot_root, 'EPIC_noun_classes.csv'), 'r'))
  nouns = [line[1] for line in reader][1:]
  return nouns

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

def prep_wv_adjMtrx():
  verbs = get_verbs()
  nouns = get_nouns()
  n_verbs = len(verbs)
  n_nouns = len(nouns)
  print('#verbs:', len(verbs))
  print('#nouns:', len(nouns))
  
  adj = np.zeros([n_verbs, n_nouns])
  for vid in range(n_verbs):
    for nid in range(n_nouns):
      v_verb = wv_map[verbs[vid]] if verbs[vid] in wv_map else np.ones([300])/300
      v_noun = wv_map[nouns[vid]] if nouns[vid] in wv_map else np.ones([300])/300
      sim = 1 - cosine(v_verb, v_noun)
      adj[vid, nid] = sim
  adj = np.concatenate([adj, np.ones([125,1])/n_verbs], 1)
  o2v = adj / adj.sum(1).reshape(-1, 1) # norm by columns
  adj = adj.transpose([1,0])
  v2o = adj / adj.sum(1).reshape(-1, 1) # norm by rows

  print('max sum v2o:', v2o.sum(1).max())
  print('max sum o2v:', o2v.sum(1).max())
  print('min sum v2o:', v2o.sum(1).min())
  print('min sum o2v:', o2v.sum(1).min())

  with open(os.path.join(meta_root, 'adjWV.pkl'), 'wb') as handle:
    pickle.dump({'v2o':v2o, 'o2v':o2v}, handle)


if __name__ == '__main__':
  # prep_weights()
  prep_wv_adjMtrx()
