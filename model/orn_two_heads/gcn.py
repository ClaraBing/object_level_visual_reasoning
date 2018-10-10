import torch
import torch.nn as nn
from utils.graph import build_vog
import pdb

class GCN(nn.Module):
  def __init__(self, options):
    super(GCN, self).__init__()
    self.device = options['device']
    if options['adj_type'] == 'prior':
      (o2v, _), (v2o, _), (vog, _) = build_vog()
      self.A_o2v = torch.Tensor(o2v).to(self.device) # shaoe: (nb_classes, nb_obj_classes)
      self.A_v2o = torch.Tensor(v2o).to(self.device)
    elif options['adj_type'] == 'uniform':
      self.A_o2v = torch.ones((options['nb_classes'], options['nb_obj_classes']))
      self.A_v2o = torch.ones((options['nb_obj_classes'], options['nb_classes']))
      self.A_o2v /= options['nb_classes']
      self.A_v2o /= options['nb_obj_classes']
      self.A_o2v = self.A_o2v.to(self.device)
      self.A_v2o = self.A_v2o.to(self.device)
    elif options['adj_type'] == 'learned':
      raise NotImplementedError('adj_type == "learned" is not implemented yet. Sorry!! > <')
    else:
      raise ValueError("options['adj_type'] should be one of 'prior', 'uniform', 'learned'")
    self.n_layers = options['n_layers']
    self.obj_embed = nn.Linear(options['D_obj'], options['D_obj_embed']).to(self.device)
    self.obj_reverse_embed = nn.Linear(options['D_obj_embed'], options['D_obj']).to(self.device)
    self.verb_embed = nn.Linear(options['D_verb'], options['D_verb_embed']).to(self.device)
    self.verb_reverse_embed = nn.Linear(options['D_verb_embed'], options['D_verb']).to(self.device)
    self.Ws_o2v, self.Ws_v2o = {}, {}
    # self.verb_embed = nn.Linear(options['D_verb'], options['nb_classes']).to(self.device)
    self.verb_expand = nn.Linear(1, options['nb_classes']).to(self.device)
    self.n_top_objs = options['n_top_objs']
    # TODO: complete GCN
    for lid in range(self.n_layers):
      """
      e.g. D_obj = 2048 / D_verb = 2048
      """
      # o2v
      mod_name = 'Ws_o2v_{}'.format(lid)
      mod = nn.Linear(options['D_obj_embed'], options['D_verb_embed']).to(self.device)
      self.add_module(mod_name, mod)
      self.Ws_o2v[mod_name] = mod
      # v2o
      mod_name = 'Ws_v2o_{}'.format(lid)
      mod = nn.Linear(options['D_verb_embed'], options['D_obj_embed']).to(self.device)
      self.add_module(mod_name, mod)
      self.Ws_v2o[mod_name] = mod

  def expand_idx(self, indices):
    """
    indices: (batch_size, n_objs, n_top)
    """
    I,J,K = indices.shape
    ids = []
    for i in range(I):
      for j in range(J):
        for k in range(K):
          ids += (i,j,k),
    ids = list(zip(*ids))
    ids = [torch.LongTensor(each).to(self.device) for each in ids]
    return ids

  def forward(self, fm, mode, scores=[]):
    if mode == 'obj':
      """
      fm: object features: (batch, n_frames, n_objs, 2248)
      scores: (batch, n_frames, n_objs, nb_obj_classes)
      """
      obj_feats = fm[:,:,:,:2048]
      n_top_objs = fm.shape[2]

      frame_feats = (obj_feats.unsqueeze(-2) * scores.unsqueeze(-1)).sum(-3)
      # pdb.set_trace()
      # fm_obj_embed: (batch, n_frame, n_obj_class, D_obj_embed)
      fm_obj_embed = self.obj_embed(frame_feats)

      # # new_fm shape: (353, 2148)
      # fm_obj = torch.matmul(cs.transpose(0,1), torch.cat([us, bs], -1)).to(self.device)
      # # embed features to a lower dimentional space
      # fm_obj_embed = self.obj_embed(fm_obj)

      for l in range(self.n_layers):
        mod_name = 'Ws_o2v_{}'.format(l)
        fm_verb_embed = self.Ws_o2v[mod_name](torch.matmul(self.A_o2v, fm_obj_embed))
        mod_name = 'Ws_v2o_{}'.format(l)
        fm_obj_embed = self.Ws_v2o[mod_name](torch.matmul(self.A_v2o, fm_verb_embed))

      # select the top n_top_objs categories
      # fm_obj_embed.shape: e.g. [8, 4, 353, 128]
      # most_activated_ids_tensor: e.g. shape = [8, 4, 10]
      most_activated_ids_tensor = fm_obj_embed.sum(-1).sort(descending=True)[1][:, :, :n_top_objs]
      # pdb.set_trace()
      ti, tj, tk = most_activated_ids_tensor.shape
      most_activated_ids = self.expand_idx(most_activated_ids_tensor)
      most_activated = fm_obj_embed[most_activated_ids].reshape(ti, tj, tk, -1)
      refined = self.obj_reverse_embed(most_activated)
      # most_activated = most_activated.type(torch.LongTensor).to(self.device)
      return refined, most_activated_ids_tensor
    elif mode == 'verb':
      """
      fm: global context vector: (B, 2048)
      """
      # embed the feature to a lower dimentional space
      fm_verb_embed = self.verb_embed(fm.to(self.device))
      # fm_verb_embed = self.verb_expand(fm_verb_embed.transpose(0,1))
      fm_verb_embed = self.verb_expand(fm_verb_embed.unsqueeze(-1))
      fm_verb_embed = fm_verb_embed.transpose(-1, -2)

      for l in range(self.n_layers):
        mod_name = 'Ws_v2o_{}'.format(l)
        fm_obj_embed = self.Ws_v2o[mod_name](torch.matmul(self.A_v2o, fm_verb_embed))
        mod_name = 'Ws_o2v_{}'.format(l)
        fm_verb_embed = self.Ws_o2v[mod_name](torch.matmul(self.A_o2v, fm_obj_embed))
      fm_verb_embed = fm_verb_embed.max(1, keepdim=False)[0]
      refined = self.verb_reverse_embed(fm_verb_embed)
      return refined
    else:
      raise ValueError("GCN forward mode should be 'obj' or 'verb'. Got {}".format(mode))

