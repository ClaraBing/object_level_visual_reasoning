import os
import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import random
from collections import Counter, OrderedDict

# working directory
default_cwd = '/sailhome/bingbin/VOG/dataset/EPIC/annotations/'
# prepend cwd to file name
to_dir_handle = lambda cwd: lambda fname: os.path.join(cwd, fname)

n_objs = 321
n_verbs = 119

# util funcs
norm_row_sum = lambda mtrx: mtrx / np.clip(mtrx.sum(axis=1)[:, np.newaxis], a_min=1, a_max=None)

def load_csv(fname):
  fname = os.path.join(default_cwd, fname)
  id_map = {}
  with open(fname, 'r') as fin:
    reader = csv.reader(fin)
    header = reader.__next__() # skip header
    for line in reader:
      id_map[line[0]] = line
  return id_map

def key_to_vals(data_noun, data_verb, data_action, mode, fcsv='', fpkl=''):
  # init out_map
  out_map = OrderedDict()
  if mode == 'n2v':
    out_map['background'] = Counter()
    for noun_class in data_noun:
      key = data_noun[noun_class][1]
      out_map[key] = Counter()
  elif mode == 'v2n':
    for verb_class in data_verb:
      key = data_verb[verb_class][1]
      out_map[key] = Counter()

  for uid in data_action:
    line = data_action[uid]
    _, _, _, _, _, _, _, _, verb, verb_class, noun, noun_class, all_nouns, all_noun_classes = line
    if mode == 'n2v':
      key, val = data_noun[noun_class][1], data_verb[verb_class][1]
    elif mode == 'v2n':
      key, val = data_verb[verb_class][1], data_noun[noun_class][1]
    else:
      raise ValueError('Mode can only be "n2v" (noun to verb) or "v2n" (verb to noun)')

    out_map[key][val] += 1
  # for key in out_map:
  #   out_map[key] = sorted(list(out_map[key]))
  
  if fcsv:
    with open(fcsv, 'w') as fout:
      writer = csv.writer(fout, delimiter=':')
      for key in sorted(out_map):
        writer.writerow([key] + [','.join(['{}:{}'.format(k, out_map[key][k]) for k in sorted(out_map[key])])])
    print('Saved to csv file', fcsv)
  if fpkl:
    with open(fpkl, 'wb') as handle:
      pickle.dump(out_map, handle)
    print('Saved to pkl file', fpkl)

  return out_map

def search_nv(data_action, v, n):
  ret = {}
  for uid in data_action:
    line = data_action[uid]
    _, _, _, _, _, _, _, _, verb, verb_class, noun, noun_class, all_nouns, all_noun_classes = line
    if noun == n and verb == v:
      ret[uid] = line
    if noun == n:
      print('verb:', verb, '/ noun:', noun)
      print(line)
  return ret

def plot_bar(kv_map, plot_out):
  counts = {}
  for key in kv_map:
    counts[key] = len(kv_map[key])
  keys = []
  vals = []
  for key in sorted(counts, key=lambda k:counts[k]):
    keys += key,
    vals += counts[key],
  pos = np.arange(len(keys))
  plt.barh(pos, vals, height=0.5, tick_label=keys)
  plt.savefig(plot_out)
  plt.clf()
  print('Plot saved at', plot_out)

def build_graph_from_map(kv_map):
  """
  Build a graph on objects or verbs
  Edges weighted by number of shared objs/verbs.
  """
  keys = list(kv_map.keys())
  adj = np.zeros((len(keys), len(keys)))
  for i in range(len(keys)):
    ki = keys[i]
    vi = set(kv_map[ki])
    for j in range(len(keys)):
      kj = keys[j]
      n_overlap = len(vi.intersection(set(kv_map[kj])))
      adj[i][j] = n_overlap
  for r in range(len(keys)):
    total = sum(adj[r]) - adj[r][r]
    if total == 0:
      adj[r] = 0
      adj[r][r] = 1
    else:
      adj[r] /= total # normalize by total number of shared objects
      adj[r][r] = 0
  return adj, keys


def build_vog(fnoun='', fverb='', faction='', cwd='', fn2v='', fv2n=''):
  if not fn2v:
    fn2v = '/sailhome/bingbin/VOG/dataset/EPIC/annotations/noun_to_verbs.pkl'
  if not fv2n:
    fv2n = '/sailhome/bingbin/VOG/dataset/EPIC/annotations/verb_to_nouns.pkl'

  if os.path.exists(fn2v) and os.path.exists(fv2n):
    """
    load nv_map / vn_map from cache
    """
    print('Loading nv_map from', fn2v)
    print('Loading vn_map from', fv2n)
    with open(fn2v, 'rb') as handle:
      nv_map = pickle.load(handle)
    with open(fv2n, 'rb') as handle:
      vn_map = pickle.load(handle)
  else:
    """
    cache missing; build nv_map / vn_map from annot files
    """

    if not cwd:
      cwd = default_cwd
    assert os.path.exists(cwd), "cwd doesn't exist: "+cwd
    to_dir = to_dir_handle(cwd)
  
    if not fnoun:
      # noun header: ['noun_id', 'class_key', 'nouns']
      fnoun = to_dir('EPIC_noun_classes.csv')
    if not fverb:
      # verb header: ['verb_id', 'class_key', 'verbs']
      fverb = to_dir('EPIC_verb_classes.csv')
    if not faction:
      # action header: ['uid', 'participant_id', 'video_id', 'narration',
      #   'start_timestamp', 'stop_timestamp', 'start_frame', 'stop_frame',
      #   'verb', 'verb_class', 'noun', 'noun_class', 'all_nouns', 'all_noun_classes']
      faction = to_dir('EPIC_train_action_labels.csv')
  
    data_noun = load_csv(fnoun)
    data_verb = load_csv(fverb)
    data_action = load_csv(faction)
  
    # noun to verb
    nv_map = key_to_vals(data_noun, data_verb, data_action, 'n2v',
        fcsv=to_dir('noun_to_verbs.csv'),
        fpkl=to_dir('noun_to_verbs.pkl'))
    # verb to noun
    vn_map = key_to_vals(data_noun, data_verb, data_action, 'v2n',
        fcsv=to_dir('verb_to_nouns.csv'),
        fpkl=to_dir('verb_to_nouns.pkl'))

  # verb-object graph
  return build_vog_from_map(nv_map, vn_map)

def build_vog_from_map(nv_map, vn_map):
  """
  Build a bipartite verb-object graph:
    connectivities are symmetric (v2o vs o2v) but weighted differently
  Edges weighted by occurrence counts of (verb, obj)
  """
  okeys = list(nv_map.keys())
  vkeys = list(vn_map.keys())
  omap = {obj:okeys.index(obj) for obj in okeys}
  vmap = {verb:vkeys.index(verb) for verb in vkeys}

  # o2v: n_verbs x n_objs
  o2v = np.zeros((len(vkeys), len(okeys)))
  for obj in nv_map:
    oid = omap[obj]
    for verb in nv_map[obj]:
      vid = vmap[verb]
      o2v[vid, oid] += nv_map[obj][verb]
  o2v = norm_row_sum(o2v)

  # v2o: n_objs x n_verbs
  v2o = np.zeros((len(okeys), len(vkeys)))
  for verb in vn_map:
    vid = vmap[verb]
    for obj in vn_map[verb]:
      oid = omap[obj]
      v2o[oid, vid] += vn_map[verb][obj]
  v2o = norm_row_sum(v2o)

  # vog: (n_objs+n_verbs) x (n_objs+n_verbs)
  print('#okeys:', len(okeys))
  print('#vkeys:', len(vkeys))
  keys = okeys + vkeys
  vog = np.zeros((len(keys), len(keys)))
  vog[len(okeys):, :len(okeys)] = o2v
  vog[:len(okeys), len(okeys):] = v2o
  return (o2v, okeys), (v2o, vkeys), (vog, keys)

def plot_graph(adj, keys, png_file, n_objs=0, sample_size=20):
  plt.figure(figsize=(10,8))
  G = nx.DiGraph() # directed graph
  sampled = random.sample(range(len(keys)), sample_size)
  sampled = sorted(sampled)
  objs = []
  verbs = []
  n_edges = 0
  for i in sampled:
    ki = keys[i]
    G.add_node(ki)
    if n_objs and i < n_objs:
      objs += ki,
    else:
      verbs += ki,
    for j in sampled:
      kj = keys[j]
      if adj[i][j] > 0:
        G.add_edge(ki, kj, key=('o2v' if i<j else 'v2o'), weight=adj[i][j])
        n_edges += 1
  print('n_edges:', n_edges)

  pos = nx.spring_layout(G, k=0.15)
  label_pos = {}
  for key,(x,y) in pos.items():
    label_pos[key] = (x, y+0.01)
  if n_objs:
    print('#objs:', len(objs))
    print(objs)
    print('#verbs:', len(verbs))
    print(verbs)
    nx.draw_networkx_nodes(G, pos, nodelist=objs, node_color='b', node_size=50, label='objects')
    nx.draw_networkx_nodes(G, pos, nodelist=verbs, node_color='r', node_size=50, label='verbs')
  else:
    nx.draw_networkx_nodes(G, pos)
  edge_widths = [5*d['weight'] for (u,v,d) in G.edges(data=True)]
  nx.draw_networkx_edges(G, pos, width=edge_widths)
  nx.draw_networkx_labels(G, label_pos, font_size=12)
  plt.legend(loc='right')
  plt.savefig(png_file)
  plt.clf()

def plot_graphs_wrapper():
  # noun header: ['noun_id', 'class_key', 'nouns']
  fnoun = 'EPIC_noun_classes.csv'
  # verb header: ['verb_id', 'class_key', 'verbs']
  fverb = 'EPIC_verb_classes.csv'
  # action header: ['uid', 'participant_id', 'video_id', 'narration',
  #   'start_timestamp', 'stop_timestamp', 'start_frame', 'stop_frame',
  #   'verb', 'verb_class', 'noun', 'noun_class', 'all_nouns', 'all_noun_classes']
  faction = 'EPIC_train_action_labels.csv'
  data_noun = load_csv(fnoun)
  data_verb = load_csv(fverb)
  data_action = load_csv(faction)

  # noun to verb
  nv_map = key_to_vals(data_noun, data_verb, data_action, 'n2v',
    fcsv=to_dir('noun_to_verbs.csv'),
    fpkl=to_dir('noun_to_verbs.pkl'))
  plot_bar(nv_map, to_dir('nv_count.png'))
  adj, keys = build_graph_from_map(nv_map)
  plot_graph(adj, keys, to_dir('nv_graph.png'))
  # verb to noun
  vn_map = key_to_vals(data_noun, data_verb, data_action, 'v2n',
    fcsv=to_dir('verb_to_nouns.csv'),
    fpkl=to_dir('verb_to_nouns.pkl'))
  plot_bar(vn_map, to_dir('vn_count.png'))
  adj, keys = build_graph_from_map(vn_map)
  plot_graph(adj, keys, to_dir('vn_graph.png'))
  # verb-object graph
  (o2v, okeys), (v2o, vkeys), (vog, keys) = build_vog(nv_map, vn_map)
  plot_graph(vog, keys, to_dir('vog.png'), n_objs=n_objs, sample_size=50)

if __name__ == "__main__":
  build_vog()
