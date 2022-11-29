from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from ..utils import common_utils
from .augmentor.data_augmentor import DataAugmentor
from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.training = training
        self.class_names = class_names
        self.logger = logger
        self.root_path = root_path if root_path is not None else Path(self.dataset_cfg.DATA_PATH)
        self.logger = logger
        if self.dataset_cfg is None or class_names is None:
            return

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = DataAugmentor(
            self.root_path, self.dataset_cfg.DATA_AUGMENTOR, self.class_names, logger=self.logger
        ) if self.training else None
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR, point_cloud_range=self.point_cloud_range, training=self.training
        )

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        To support a custom dataset, implement this function to receive the predicted results from the model, and then
        transform the unified normative coordinate to your required coordinate, and optionally save them to disk.

        Args:
            batch_dict: dict of original data from the dataloader
            pred_dicts: dict of predicted results from the model
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path: if it is not None, save the results to this path
        Returns:

        """

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
        To support a custom dataset, implement this function to load the raw data (and labels), then transform them to
        the unified normative coordinate and call the function self.prepare_data() to process the data and send them
        to the model.

        Args:
            index:

        Returns:

        """
        raise NotImplementedError
    
    def distance(self, pc):
        return np.sqrt(pc[:,0]**2 + pc[:,1]**2 + pc[:, 2]**2)
        #return np.sqrt(pc[:,0]**2 + pc[:,1]**2)
    
    def distance_box(self, box):
        return np.sqrt(box[:,0]**2 + box[:,1]**2 + box[:, 2]**2)
        #return np.sqrt(box[:,0]**2 + box[:,1]**2)
    
    def get_3d_box(self, center, heading_angle, box_size):
        ''' Calculate 3D bounding box corners from its parameterization.

        Input:
            box_size: tuple of (l,w,h)
            heading_angle: rad scalar, clockwise from pos x axis
            center: tuple of (x,y,z)
        Output:
            corners_3d: numpy array of shape (8,3) for 3D box cornders
        '''
        def roty(t):
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c,  0,  s],
                            [0,  1,  0],
                            [-s, 0,  c]])
            # return np.array([[1,  0,  0],
            #                  [0,  1,  0],
            #                  [0, 0,  1]])
        def rotz(t):
            c = np.cos(t)
            s = np.sin(t)
            return np.array([
                            [np.cos(t), -np.sin(t), 0.0],
                            [np.sin(t), np.cos(t), 0.0],
                            [0.0, 0.0, 1.0]])
            # return np.array([[1,  0,  0],
            #                  [0,  1,  0],
            #                  [0, 0,  1]])

        #R = roty(heading_angle)
        R = rotz(heading_angle)
        #l,w,h = box_size
        l,h,w = box_size
        #w,h,l = box_size
        #w,l,h = box_size
        x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
        y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
        #y_corners = [0,0,0,0,-h,-h,-h,-h];
        z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
        corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
        # corners_3d[0,:] = corners_3d[0,:] + center[0]
        # corners_3d[1,:] = corners_3d[1,:] + center[1]
        # corners_3d[2,:] = corners_3d[2,:] + center[2]
        corners_3d[0,:] = corners_3d[0,:] + center[0]
        corners_3d[1,:] = corners_3d[1,:] + center[1]
        corners_3d[2,:] = corners_3d[2,:] + center[2]
        corners_3d = np.transpose(corners_3d)
        distance_ = self.distance_box(corners_3d)
        #return corners_3d
        return distance_

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """
        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        pc_ = data_dict['points']
        default_mask = self.distance(pc_)
        pc_1_mask = (default_mask>0) * (default_mask<10)
        pc_2_mask = (default_mask>=10) * (default_mask<20)
        pc_3_mask = (default_mask>=20) * (default_mask<30)
        pc_4_mask = (default_mask>=30) * (default_mask<40)
        pc_5_mask = (default_mask>=40) * (default_mask<50)
        pc_6_mask = (default_mask>=50)

        pc_1 = pc_[pc_1_mask]
        pc_2 = pc_[pc_2_mask]
        pc_3 = pc_[pc_3_mask]
        pc_4 = pc_[pc_4_mask]
        pc_5 = pc_[pc_5_mask]
        pc_6 = pc_[pc_6_mask]

        box = data_dict['gt_boxes']

        default_gt_close = np.zeros((box.shape[0]))
        for num in range(box.shape[0]):
            box_ = box[num]
            box_points = self.get_3d_box(box_[:3], box_[3], box_[4:7])
            default_gt_close[num] = box_points.min()

        default_gt_center = self.distance_box(box)
        gt_1_mask = (default_gt_close>0) * (default_gt_close<=10)
        gt_2_mask = (default_gt_close>=10) * (default_gt_close<=20)
        gt_3_mask = (default_gt_close>=20) * (default_gt_close<=30)
        gt_4_mask = (default_gt_close>=30) * (default_gt_close<=40)
        gt_5_mask = (default_gt_close>=40) * (default_gt_close<=50)
        gt_6_mask = (default_gt_close>=50)
        
        gt_1_mask__ = (default_gt_center>0) * (default_gt_center<=10)
        gt_2_mask__ = (default_gt_center>=10) * (default_gt_center<=20)
        gt_3_mask__ = (default_gt_center>=20) * (default_gt_center<=30)
        gt_4_mask__ = (default_gt_center>=30) * (default_gt_center<=40)
        gt_5_mask__ = (default_gt_center>=40) * (default_gt_center<=50)
        gt_6_mask__ = (default_gt_center>=50)

        gt_1_mask = np.logical_or(gt_1_mask, gt_1_mask__)
        gt_2_mask = np.logical_or(gt_2_mask, gt_2_mask__)
        gt_3_mask = np.logical_or(gt_3_mask, gt_3_mask__)
        gt_4_mask = np.logical_or(gt_4_mask, gt_4_mask__)
        gt_5_mask = np.logical_or(gt_5_mask, gt_5_mask__)
        gt_6_mask = np.logical_or(gt_6_mask, gt_6_mask__)

        gt_1 = box[gt_1_mask]
        gt_2 = box[gt_2_mask]
        gt_3 = box[gt_3_mask]
        gt_4 = box[gt_4_mask]
        gt_5 = box[gt_5_mask]
        gt_6 = box[gt_6_mask]

        name = data_dict['gt_names']
        name_1 = name[gt_1_mask]
        name_2 = name[gt_2_mask]
        name_3 = name[gt_3_mask]
        name_4 = name[gt_4_mask]
        name_5 = name[gt_5_mask]
        name_6 = name[gt_6_mask]

        data_dict_1, data_dict_2, data_dict_3, data_dict_4, data_dict_5, data_dict_6 = {},{},{},{},{},{}
        for key in data_dict.keys():
            if key in ['points', 'gt_names', 'gt_boxes']:
                if key == 'points':
                    data_dict_1[key] = pc_1
                    data_dict_2[key] = pc_2
                    data_dict_3[key] = pc_3
                    data_dict_4[key] = pc_4
                    data_dict_5[key] = pc_5
                    data_dict_6[key] = pc_6
                elif key == 'gt_names':
                    data_dict_1[key] = name_1
                    data_dict_2[key] = name_2
                    data_dict_3[key] = name_3
                    data_dict_4[key] = name_4
                    data_dict_5[key] = name_5
                    data_dict_6[key] = name_6
                elif key == 'gt_boxes':
                    data_dict_1[key] = gt_1
                    data_dict_2[key] = gt_2
                    data_dict_3[key] = gt_3
                    data_dict_4[key] = gt_4
                    data_dict_5[key] = gt_5
                    data_dict_6[key] = gt_6
            else:
                data_dict_1[key] = data_dict[key]
                data_dict_2[key] = data_dict[key]
                data_dict_3[key] = data_dict[key]
                data_dict_4[key] = data_dict[key]
                data_dict_5[key] = data_dict[key]
                data_dict_6[key] = data_dict[key]

        data_dict = self.data_processor.forward(data_dict=data_dict)

        data_dict_1 = self.data_processor.forward(data_dict=data_dict_1)
        data_dict_2 = self.data_processor.forward(data_dict=data_dict_2)
        data_dict_3 = self.data_processor.forward(data_dict=data_dict_3)
        data_dict_4 = self.data_processor.forward(data_dict=data_dict_4)
        data_dict_5 = self.data_processor.forward(data_dict=data_dict_5)
        data_dict_6 = self.data_processor.forward(data_dict=data_dict_6)
        data_dict['patch'] = [data_dict_1, data_dict_2, data_dict_3, data_dict_4, data_dict_5, data_dict_6]

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                if key == 'patch':
                    continue
                data_dict[key].append(val)
        
        try :
            if batch_list[0]['patch_shuffle'] == True:

                patch_flag = True
                for cur_sample in batch_list:
                    for key, val in cur_sample.items():
                        if key == 'patch':
                            continue
                        #del(data_dict[key][0])
                        #data_dict[key].append(val)
                
                    batch_1 = batch_list[0]['patch']
                    try:
                        batch_2 = batch_list[1]['patch']
                    except:
                        batch_2 = batch_list[0]['patch']
                        patch_flag = False
                                   
                
                data_1 = batch_1[0]
                data_2 = batch_2[0]

                for key in data_1.keys():
                    for num in range(1, 6):# 총 구간이 6개임
                        if key == 'frame_id':
                            # key_ = data_1[key] + "_" + data_2[key]
                            # data_1[key] = key_
                            # data_2[key] = key_
                            continue
                        elif key in ['flip_x', 'flip_y', 'rot_angle', 'scale', 'use_lead_xyz', 'image_shape', 'gt_names', 'calib', 'road_plane', 'patch_shuffle']:
                            continue
                        elif key == 'voxel_num_points':
                            if num%2 ==0:
                                data_1[key] = np.concatenate((data_1[key], batch_1[num][key]), axis=None)
                                data_2[key] = np.concatenate((data_2[key], batch_2[num][key]), axis=None)
                            elif num%2 !=0:
                                data_1[key] = np.concatenate((data_1[key], batch_2[num][key]), axis=None)
                                data_2[key] = np.concatenate((data_2[key], batch_1[num][key]), axis=None)
                        else:
                            if num%2 ==0:
                                data_1[key] = np.concatenate((data_1[key], batch_1[num][key]), axis=0)
                                data_2[key] = np.concatenate((data_2[key], batch_2[num][key]), axis=0)
                            elif num%2 !=0:
                                data_1[key] = np.concatenate((data_1[key], batch_2[num][key]), axis=0)
                                data_2[key] = np.concatenate((data_2[key], batch_1[num][key]), axis=0)

                if patch_flag == True:
                    for cur_sample in [data_1, data_2]:
                        for key, val in cur_sample.items():
                            if key == 'gt_names':
                                continue
                            data_dict[key].append(val)
                else:
                    for cur_sample in [data_1]:
                        for key, val in cur_sample.items():
                            if key == 'gt_names':
                                continue
                            data_dict[key].append(val)

            if patch_flag == True:
                batch_size= 4
                ret = {}
                ret['batch_size'] = batch_size
                #batch_size=len(batch_list) #2
            else:
                batch_size=1
                ret = {}
                ret['batch_size'] = batch_size
                for key in data_dict.keys():
                    del(data_dict[key][1])
                #batch_size=len(batch_list) #1

            # batch_size = len(batch_list) 
            # batch_size = len(batch_list)# + 2 #*patch_flag
        except:
            ret = {}
            batch_size = len(batch_list)
            ret['batch_size'] = len(batch_list) #batch_size
            #pass

        # ret = {}
        # batch_size=2


        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points', 'voxels_ema', 'voxel_num_points_ema']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords', 'points_ema', 'voxel_coords_ema']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes', 'gt_boxes_ema']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        # ret['batch_size'] = batch_size
        return ret
