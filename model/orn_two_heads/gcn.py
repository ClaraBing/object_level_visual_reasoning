import torch
import torch.nn as nn

class GCN(nn.Module):
  def __init__(self, options):
    super(GCN, self).__init__()
    self.device = options['device']
    if options['adj_type'] == 'prior':
      self.A_o2v = torch.Tensor(options['A_o2v']).transpose(0,1).to(self.device) # shaoe: (nb_classes, nb_obj_classes)
      self.A_v2o = torch.Tensor(options['A_v2o']).transpose(0,1).to(self.device)
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
    self.verb_embed = nn.Linear(options['D_verb'], options['D_verb_embed']).to(self.device)
    self.Ws_o2v, self.Ws_v2o = {}, {}
    # self.verb_embed = nn.Linear(options['D_verb'], options['nb_classes']).to(self.device)
    self.verb_expand = nn.Linear(1, options['nb_classes']).to(self.device)
    self.n_top_objs = options['n_top_objs']
    # TODO: complete GCN
    for lid in range(self.n_layers):
      """
      e.g. D_obj = 2148 / D_verb = 2048
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

  def forward(self, fm, mode, scores=[]):
    if mode == 'obj':
      """
      fm: object features: (batch, n_frames, n_objs, 2248)
      scores: (batch, n_frames, n_objs, nb_obj_classes)
      """
      obj_feats = fm[:,:,:,:2048]
      obj_masks = fm[:,:,:,2048:2148]
      obj_scores = fm[:,:,:,2148:]

      frame_feats = (obj_feats.unsqueeze(-2) * scores.unsqueeze(-1)).sum(-3)
      print('GCN forward: frame_feats:', frame_feats.shape)
      fm_obj_embed = self.obj_embed(frame_feats)
      print('GCN forward: fm_obj_embed:', fm_obj_embed.shape)

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
      most_activated = fm_obj_embed.sum(1).sort(descending=True)[1][:self.n_top_objs]
      most_activated = most_activated.type(torch.LongTensor).to(self.device)
      return fm_obj_embed[most_activated], most_activated
    elif mode == 'verb':
      """
      fm: global context vector: (1, 2048)
      """
      # embed the feature to a lower dimentional space
      fm_verb_embed = self.verb_embed(fm.to(self.device))
      fm_verb_embed = self.verb_expand(fm_verb_embed.transpose(0,1))
      fm_verb_embed = fm_verb_embed.transpose(0,1)

      for l in range(self.n_layers):
        mod_name = 'Ws_v2o_{}'.format(l)
        fm_obj_embed = self.Ws_v2o[mod_name](torch.matmul(self.A_v2o, fm_verb_embed))
        mod_name = 'Ws_o2v_{}'.format(l)
        fm_verb_embed = self.Ws_o2v[mod_name](torch.matmul(self.A_o2v, fm_obj_embed))
      fm_verb_embed = fm_verb_embed.max(0, keepdim=True)[0]
      return fm_verb_embed
    else:
      raise ValueError("GCN forward mode should be 'obj' or 'verb'. Got {}".format(mode))

