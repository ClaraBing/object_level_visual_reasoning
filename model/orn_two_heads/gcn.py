import torch
import torch.nn as nn

DEBUG = False

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
      obj_masks = fm[:,:,:,2048:2148]
      obj_scores = fm[:,:,:,2148:]

      if DEBUG: print('About to unsqueeze features.')
      if DEBUG: print('obj_feats:', obj_feats.shape)
      if DEBUG: print('scores:', scores.shape)
      frame_feats = (obj_feats.unsqueeze(-2) * scores.unsqueeze(-1)).sum(-3)
      if DEBUG: print('GCN forward: frame_feats:', frame_feats.shape)
      # fm_obj_embed: (batch, n_frame, n_obj_class, D_obj_embed)
      fm_obj_embed = self.obj_embed(frame_feats)
      if DEBUG: print('GCN forward: fm_obj_embed:', fm_obj_embed.shape)

      # # new_fm shape: (353, 2148)
      # fm_obj = torch.matmul(cs.transpose(0,1), torch.cat([us, bs], -1)).to(self.device)
      # # embed features to a lower dimentional space
      # fm_obj_embed = self.obj_embed(fm_obj)

      for l in range(self.n_layers):
        if DEBUG: print('layer', l)
        mod_name = 'Ws_o2v_{}'.format(l)
        fm_verb_embed = self.Ws_o2v[mod_name](torch.matmul(self.A_o2v, fm_obj_embed))
        if DEBUG: print('o2v finished')
        mod_name = 'Ws_v2o_{}'.format(l)
        fm_obj_embed = self.Ws_v2o[mod_name](torch.matmul(self.A_v2o, fm_verb_embed))
        if DEBUG: print('v2o finished')

      # select the top n_top_objs categories
      if DEBUG: print('fm_obj_embed:', fm_obj_embed.shape)
      most_activated_ids_tensor = fm_obj_embed.sum(-1).sort(descending=True)[1][:, :, :self.n_top_objs]
      if DEBUG: print('most_activated_ids (tensor):', most_activated_ids_tensor.shape)
      ti, tj, tk = most_activated_ids_tensor.shape
      most_activated_ids = self.expand_idx(most_activated_ids_tensor)
      if DEBUG: print('most_activated_ids: dim={} / len={}'.format(len(most_activated_ids), most_activated_ids[0].shape))
      most_activated = fm_obj_embed[most_activated_ids].reshape(ti, tj, tk, -1)
      # most_activated = most_activated.type(torch.LongTensor).to(self.device)
      if DEBUG: print('most_activated:', most_activated.shape)
      if DEBUG: print('Exiting GCN (obj)')
      return most_activated, most_activated_ids_tensor
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

