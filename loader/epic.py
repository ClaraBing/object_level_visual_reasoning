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


class EPIC(VideoDataset):
    """
    Loader for the EPIC dataset
    """

    def __init__(self, options, **kwargs):
        super().__init__(options, **kwargs)

        self.ts2info_fn = '/vision2/u/bingbin/ORN/data/ts2info.pkl'
        with open(self.ts2info_fn, 'rb') as handle:
          self.ts2info = pickle.load(handle)

        # Videos paths
        self.list_video, self.dict_video_length, self.dict_video_label = self.get_videos()
        print('EPIC dataset: size =', len(self.list_video))

    def get_videos(self):
        list_video = sorted(self.ts2info.keys())
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

    def get_length(self, id):
        return self.dict_video_length[id]

    def get_target(self, id):
        # verb_cls: 0 - 124
        verb_cls = self.dict_video_label[id]
        label = np.zeros([self.nb_classes])
        label[verb_cls] = 1
        return torch.FloatTensor(label)
