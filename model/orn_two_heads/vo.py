import torch
import torch.nn as nn
from utils.graph import build_vog
import pdb

class VO(nn.Module):
  def __init__(self, options, size_object):
    super(VO, self).__init__()

    self.device = options['device']

    # Adjacency matrix
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

    self.n_obj_cls, self.n_verb_cls = self.A_v2o.shape

    # self.n_layers = options['n_layers']
    self.obj_embed = nn.Linear(size_object, options['D_obj_embed']).to(self.device)
    # self.obj_reverse_embed = nn.Linear(options['D_obj_embed'], options['D_obj']).to(self.device)
    # self.verb_embed = nn.Linear(options['D_verb'], options['D_verb_embed']).to(self.device)
    # self.verb_reverse_embed = nn.Linear(options['D_verb_embed'], options['D_verb']).to(self.device)


  def forward(self, objects_features, fm_context, obj_id):
    """
    Input shapes:
    fm_context: (B, T, D_verb)
    objects_features: (B, T, n, D_obj)
    obj_id: (B, T, n, 353)
    """
    B, T, N, _ = obj_id.shape
    obj_id_clone = obj_id.clone()
    for bid in range(B):
      for tid in range(T):
        for nid in range(N):
          if obj_id[bid, tid, nid].sum() == 0:
            obj_id_clone[bid, tid, nid, 0] = 1
          if obj_id_clone[bid,tid,nid].sum()>1:
            last_cls = int(obj_id_clone[bid, tid, nid].nonzero()[-1])
            obj_id_clone[bid,tid,nid] = 0
            obj_id_clone[bid,tid,nid,last_cls] = 1
    try:
      assert(obj_id_clone.sum() == B*T*N)
    except:
      pdb.set_trace()

    obj_embedded = self.obj_embed(objects_features)
    
    # prepare masked adj matrix
    pdb.set_trace()
    expand_to = (B, T, N, self.n_verb_cls, self.n_obj_cls)
    eid = obj_id_clone.unsqueeze(-2).expand(*expand_to).type(torch.ByteTensor)
    eA = self.A_o2v.expand(*expand_to)
    try:
      subA = eA[eid].view(B, T, N, self.n_verb_cls)
    except:
      pdb.set_trace()

    afforded_actions = torch.matmul(subA.transpose(2,3), obj_embedded) # shape: (B, T, 125, D_obj_embed)
    expanded_context = fm_context.unsqueeze(1).unsqueeze(2).expand(B, T, self.n_verb_cls, fm_context.shape[-1])
    vo_actions = torch.cat([afforded_actions, expanded_context], -1)
    
    global_vo_actions = vo_actions.mean(1)
    return global_vo_actions

