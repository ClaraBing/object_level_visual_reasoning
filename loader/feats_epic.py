from __future__ import print_function
import torch
import torch.utils.data as data
from torchvision import transforms
import os
import random
from PIL import Image
import numpy as np
import pickle
from pycocotools import mask as maskUtils
import lintel
import time
from torch.utils.data.dataloader import default_collate
from random import shuffle
from loader.videodataset import VideoDataset


class FeatsEPIC(VideoDataset):
    """
    Loader for the EPIC dataset; load features to save compute.
    """

    def __init__(self, options, **kwargs):
        super().__init__(options, **kwargs)

        # extract
        self.feats_obj_dir = options['feats_obj_dir'] # fm_objects: ndarray (2048, 4, 14, 14)
        self.feats_ctxt_dir = options['feats_ctxt_dir'] # fm_context: ndarray (2048, 4, 7, 7)
        self.feats_obj_fn_format = os.path.join(self.feats_obj_dir, '{}')
        self.feats_ctxt_fn_format = os.path.join(self.feats_ctxt_dir, '{}')
        self.ts2info_fn = '/vision2/u/bingbin/ORN/meta/ts2info.pkl'
        with open(self.ts2info_fn, 'rb') as handle:
          self.ts2info = pickle.load(handle)

        # Videos paths
        self.list_video, self.dict_video_length, self.dict_video_label = self.get_videos()
        print('FeatsEPIC dataset: size =', len(self.list_video))

    def get_videos(self):
        list_video = sorted(self.ts2info.keys())
        # list_video = [each for each in sorted(self.ts2info.keys()) if os.path.exists(os.path.join(self.mask_dir_full, each))]
        dict_video_length = {key:self.ts2info[key]['len'] for key in list_video}
        dict_video_label = {key:self.ts2info[key]['verb_cls'] for key in list_video}

        return list_video, dict_video_length, dict_video_label

    def starting_point(self, id):
        # TODO: does EPIC has random starting points?
        # this seems fine for now
        return 0

    def get_mask_file(self, id):
        """
        id: mask file name e.g. 'P01_P01_02_00:01:49.17_00:01:51.29.pkl'
        """
        # Get the approriate masks
        mask_fn = os.path.join(self.mask_dir_full, id)

        return mask_fn

    def get_video_fn(self, id):
        return self.ts2info[id]['frames']

    def get_gt_obj(self, id):
        return self.ts2info[id]['noun_cls']

    def get_length(self, id):
        return self.dict_video_length[id]

    def get_target(self, id):
        # verb_cls: 0 - 124
        verb_cls = self.dict_video_label[id]
        # return verb_cls
        label = np.zeros([self.nb_classes])
        label[verb_cls] = 1
        return torch.FloatTensor(label)

    
    def __getitem__(self, index):
       ret = super().__getitem__(index)
       id = self.list_video[index]
       with open(self.feats_obj_fn_format.format(id), 'rb') as handle:
         np_obj = pickle.load(handle) 
       with open(self.feats_ctxt_fn_format.format(id), 'rb') as handle:
         np_ctxt = pickle.load(handle)

       ret['fm_obj'] = torch.from_numpy(np_obj)
       ret['fm_context'] = torch.from_numpy(np_ctxt)
       return ret
