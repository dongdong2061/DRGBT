import os
import os.path
import torch,cv2
import numpy as np
import pandas
import csv
import random
from collections import OrderedDict

from lib.train.dataset.depth_utils import get_x_frame
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader,opencv_loader
from lib.train.admin import env_settings

class DRGBT603_trainingSet(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, data_fraction=None, attr=None):
        self.root = env_settings().demt_dir if root is None else root
        super().__init__('DRGBT603_trainingSet', root, image_loader)

        # video_name for each sequence
        with open('evaluate_DRGBT603/train_set.txt','r') as f:
            list =  [line.strip() for line in f]
        # video_name for each sequence
        self.sequence_list = list



        
    def get_name(self):
        return 'DRGBT603_trainingSet'

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, 'init.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False,
                             low_memory=False).values
        return torch.tensor(gt)

    def _read_modal_anno(self, seq_path,direction):
        # print(seq_path)
        bb_anno_file = os.path.join(seq_path, f'modal_{direction}.txt')
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True, na_values=['', 'NaN'],
                             low_memory=False).values
        # print(type(gt))
        gt = gt[0]

        return torch.tensor(gt)



    def get_sequence_info(self, seq_id):
        #print('seq_id', seq_id)
        seq_name = self.sequence_list[seq_id]
        #print('seq_name', seq_name)
        seq_path = os.path.join(self.root, seq_name)
        bbox = self._read_bb_anno(seq_path)

        modal_up = self._read_modal_anno(seq_path, "up")
        modal_down = self._read_modal_anno(seq_path, "down")

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        dict_ = {'bbox': bbox, 'valid': valid, 'visible': visible,'modality_up':modal_up,'modality_down':modal_down}
        # return {'bbox': bbox, 'valid': valid, 'visible': visible},seq_name
        return dict_

    def _get_frame_v(self, seq_path, frame_id):
        frame_path_v = os.path.join(seq_path, 'up', sorted([p for p in os.listdir(os.path.join(seq_path, 'up')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_v)
        
    def _get_frame_i(self, seq_path, frame_id):
        frame_path_i = os.path.join(seq_path, 'down', sorted([p for p in os.listdir(os.path.join(seq_path, 'down')) if os.path.splitext(p)[1] in ['.jpg','.png','.bmp']])[frame_id])
        return self.image_loader(frame_path_i)


    def _get_frame_path(self, seq_path, frame_id):
        vis_frame_names = sorted(os.listdir(os.path.join(seq_path, 'up')))
        inf_frame_names = sorted(os.listdir(os.path.join(seq_path, 'down')))
        return os.path.join(seq_path, 'up', vis_frame_names[frame_id]), os.path.join(seq_path, 'down', inf_frame_names[frame_id])   



    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, ir_frame_path = self._get_frame_path(seq_path, frame_id)
        img = get_x_frame(rgb_frame_path, ir_frame_path, dtype='rgbrgb')
        return img  # (h,w,6)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_name = self.sequence_list[seq_id]
        seq_path = os.path.join(self.root, seq_name)
        frame_list = [self._get_frame(seq_path, f) for f in frame_ids]

        if seq_name not in self.sequence_list:
            print('warning!!!'*100)
        if anno is None:
            anno = self.get_sequence_info(seq_path)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        #return frame_list_v, frame_list_i, anno_frames, object_meta
        return frame_list, anno_frames, object_meta

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
