#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import pickle
import torch
import tqdm
from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import filter_eval_boxes, load_prediction
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionConfig
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.eval.tracking.data_classes import TrackingBox
from nuscenes.eval.tracking.utils import category_to_tracking_name
from pyquaternion import Quaternion
from typing import Optional
from .omega_utils import trainval_split
from ..nuscenes.nuscenes_dataset import NuScenesDataset
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import common_utils, box_utils
from ..dataset import DatasetTemplate


class DetectionOmegaEval(DetectionEval):
    """
    This is the official nuScenes detection evaluation code.
    Results are written to the provided output_dir.

    nuScenes uses the following detection metrics:
    - Mean Average Precision (mAP): Uses center-distance as matching criterion; averaged over distance thresholds.
    - True Positive (TP) metrics: Average of translation, velocity, scale, orientation and attribute errors.
    - nuScenes Detection Score (NDS): The weighted sum of the above.

    Here is an overview of the functions in this method:
    - init: Loads GT annotations and predictions stored in JSON format and filters the boxes.
    - run: Performs evaluation and dumps the metric data to disk.
    - render: Renders various plots and dumps to disk.

    We assume that:
    - Every sample_token is given in the results, although there may be not predictions for that sample.

    Please see https://www.nuscenes.org/object-detection for more details.
    """

    def __init__(self,
                 nusc: NuScenes,
                 config: DetectionConfig,
                 result_path: str,
                 eval_set: str,
                 output_dir: str = None,
                 use_smAP: bool = False,
                 verbose: bool = True):
        """
        Initialize a DetectionEval object.
        :param nusc: A NuScenes object.
        :param config: A DetectionConfig object.
        :param result_path: Path of the nuScenes JSON result file.
        :param eval_set: The dataset split to evaluate on, e.g. train, val or test.
        :param output_dir: Folder to save plots and results to.
        :param verbose: Whether to print to stdout.
        """
        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config
        self.use_smAP = use_smAP

        # Check result file exists.
        assert os.path.exists(
            result_path), 'Error: The result file does not exist!'

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, 'plots')
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print('Initializing nuScenes detection evaluation')
        self.pred_boxes, self.meta = load_prediction(
            self.result_path,
            self.cfg.max_boxes_per_sample,
            DetectionBox,
            verbose=verbose)
        self.gt_boxes = load_gt_omega(self.nusc,
                                      self.eval_set,
                                      DetectionBox,
                                      verbose=verbose)

        assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), \
            "Samples in split doesn't match samples in predictions."

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print('Filtering predictions')
        self.pred_boxes = filter_eval_boxes(nusc,
                                            self.pred_boxes,
                                            self.cfg.class_range,
                                            verbose=verbose)
        if verbose:
            print('Filtering ground truth annotations')
        self.gt_boxes = filter_eval_boxes(nusc,
                                          self.gt_boxes,
                                          self.cfg.class_range,
                                          verbose=verbose)

        self.sample_tokens = self.gt_boxes.sample_tokens

        if self.use_smAP:
            smAP_15 = self._calc_smAP(self.pred_boxes, self.gt_boxes, 15)
            smAP_30 = self._calc_smAP(self.pred_boxes, self.gt_boxes, 30)
            smAP_50 = self._calc_smAP(self.pred_boxes, self.gt_boxes, 50)

    def _calc_smAP(self, pred_boxes, gt_boxes, max_dist, iou_thres=0.6):
        from nuscenes.eval.common.utils import quaternion_yaw
        from pyquaternion import Quaternion

        from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import \
            BboxOverlaps3D

        def form_converter(boxes, is_gt=False):
            # convert nus DetectionBox -> torch.tensor
            conv_boxes = []
            conv_labels = []
            for box in boxes:
                c_box = torch.zeros(7, dtype=torch.float)
                if pow(box.translation[0], 2) + pow(box.translation[1],
                                                    2) > pow(max_dist, 2):
                    continue
                for i in range(3):
                    c_box[i] = box.translation[i]
                    c_box[i + 3] = box.size[i]
                if is_gt:
                    c_box[[3, 4]] = c_box[[4, 3]]
                c_box[6] = quaternion_yaw(Quaternion(box.rotation))
                conv_boxes.append(c_box)
                conv_labels.append(box.detection_name)
            if len(conv_boxes) == 0:
                return [], []
            return torch.stack(conv_boxes), conv_labels

        iou_call = BboxOverlaps3D('lidar')
        detected_sample_num = 0
        not_sample_num = 0
        for sample_token in self.sample_tokens:
            s_gt_boxes, s_gt_labels = form_converter(gt_boxes[sample_token],
                                                     is_gt=True)
            s_pred_boxes, s_pred_labels = form_converter(
                pred_boxes[sample_token])
            if len(s_gt_boxes) == 0:
                not_sample_num += 1
                continue
            if len(s_pred_boxes) == 0:
                continue
            iou_mat = iou_call(s_gt_boxes, s_pred_boxes).max(1)
            for idx in range(len(iou_mat.values)):
                if s_gt_labels[idx] not in ['car', 'truck', 'bus']:
                    continue
                if iou_mat.values[idx] < iou_thres:
                    break
            if idx == len(iou_mat.values) - 1:
                detected_sample_num += 1
        smAP = (detected_sample_num -
                not_sample_num) / (len(self.sample_tokens) - not_sample_num)
        print('smAP_%d: pos sample: %d, neg sample: %d' %
              (max_dist, (detected_sample_num - not_sample_num),
               len(self.sample_tokens) - detected_sample_num - not_sample_num))
        print('smAP_%d: %f' % (max_dist, smAP))
        return smAP


def load_gt_omega(nusc: NuScenes,
                  eval_split: str,
                  box_cls,
                  verbose: bool = False) -> EvalBoxes:
    """
    Loads ground truth boxes from DB.
    :param nusc: A NuScenes instance.
    :param eval_split: The evaluation split for which we load GT boxes.
    :param box_cls: Type of box to load, e.g. DetectionBox or TrackingBox.
    :param verbose: Whether to print messages to stdout.
    :return: The GT boxes.
    """
    # Init.
    if box_cls == DetectionBox:
        attribute_map = {a['token']: a['name'] for a in nusc.attribute}

    if verbose:
        print('Loading annotations for {} split from nuScenes version: {}'.
              format(eval_split, nusc.version))
    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in nusc.sample]
    assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

    version = nusc.version
    # Only keep samples from this split.
    train_key = 'train_scenes'
    if 'train_l_scenes' in trainval_split[version]:
        train_key = 'train_l_scenes'
    splits = {
        'train': trainval_split[version][train_key],
        'val': trainval_split[version]['val_scenes']
    }

    # Check compatibility of split with nusc_version.
    if eval_split in {'train', 'val', 'train_detect', 'train_track'}:
        assert version.endswith('trainval'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split in {'mini_train', 'mini_val'}:
        assert version.endswith('mini'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    elif eval_split == 'test':
        assert version.endswith('test'), \
            'Error: Requested split {} which is not compatible with NuScenes version {}'.format(eval_split, version)
    else:
        raise ValueError(
            'Error: Requested split {} which this function cannot map to the correct NuScenes version.'
            .format(eval_split))

    if eval_split == 'test':
        # Check that you aren't trying to cheat :).
        assert len(nusc.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = nusc.get('sample', sample_token)['scene_token']
        scene_record = nusc.get('scene', scene_token)
        if scene_record['name'] in splits[eval_split]:
            sample_tokens.append(sample_token)

    all_annotations = EvalBoxes()

    # Load annotations and filter predictions and annotations.
    tracking_id_set = set()
    for sample_token in tqdm.tqdm(sample_tokens, leave=verbose):

        sample = nusc.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']

        sample_boxes = []
        for sample_annotation_token in sample_annotation_tokens:

            sample_annotation = nusc.get('sample_annotation',
                                         sample_annotation_token)
            if box_cls == DetectionBox:
                # Get label name in detection task and filter unused labels.
                detection_name = category_to_detection_name_omega(
                    sample_annotation['category_name'])
                if detection_name is None:
                    continue

                # rot 90
                np_loc = np.array(sample_annotation['translation']).copy()
                np_loc = np_loc[[1, 0, 2]]
                np_loc[1] = -np_loc[1]
                sample_annotation['translation'] = np_loc.tolist()
                rad = Quaternion(sample_annotation['rotation']).radians
                rad = (rad + np.pi / 2) % (np.pi * 2)
                rad_ = Quaternion(axis=[0, 0, 1], radians=rad).elements
                sample_annotation['rotation'] = rad_
                ########
                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(
                            sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'],
                        detection_name=detection_name,
                        detection_score=-1.0,  # GT samples do not have a score.
                        attribute_name=''))
            elif box_cls == TrackingBox:
                # Use nuScenes token as tracking id.
                tracking_id = sample_annotation['instance_token']
                tracking_id_set.add(tracking_id)

                # Get label name in detection task and filter unused labels.
                tracking_name = category_to_tracking_name(
                    sample_annotation['category_name'])
                if tracking_name is None:
                    continue

                sample_boxes.append(
                    box_cls(
                        sample_token=sample_token,
                        translation=sample_annotation['translation'],
                        size=sample_annotation['size'],
                        rotation=sample_annotation['rotation'],
                        velocity=nusc.box_velocity(
                            sample_annotation['token'])[:2],
                        num_pts=sample_annotation['num_lidar_pts'] +
                        sample_annotation['num_radar_pts'],
                        tracking_id=tracking_id,
                        tracking_name=tracking_name,
                        tracking_score=-1.0  # GT samples do not have a score.
                    ))
            else:
                raise NotImplementedError('Error: Invalid box_cls %s!' %
                                          box_cls)

        all_annotations.add_boxes(sample_token, sample_boxes)

    if verbose:
        print("Loaded ground truth annotations for {} samples.".format(
            len(all_annotations.sample_tokens)))

    return all_annotations


class NuScenesOmegaEval(DetectionOmegaEval):
    """
    Dummy class for backward-compatibility. Same as DetectionEval.
    """


def category_to_detection_name_omega(category_name: str) -> Optional[str]:
    """
    Default label mapping from nuScenes to nuScenes detection classes.
    Note that pedestrian does not include personal_mobility, stroller and wheelchair.
    :param category_name: Generic nuScenes class.
    :return: nuScenes detection class.
    """
    #  detection_mapping = {
        #  'Vehicle::Car': 'car',
        #  'Vehicle::Bus': 'bus',
        #  'Vehicle::Motorcycle': 'motorcycle',
        #  'Vehicle::Truck': 'truck',
        #  'Pedestrian::Pedestrian': 'pedestrian',
        #  'Vehicle::Other': 'bus'
    #  }
    detection_mapping = {
        'Vehicle::Car': 'car',
        'Vehicle::Bus': 'bus',
        'Vehicle::Motorcycle': 'motorcycle',
        'Vehicle::Truck': 'truck',
        'Pedestrian::Pedestrian': 'pedestrian'
    }

    if category_name in detection_mapping:
        return detection_mapping[category_name]
    else:
        return None


class OmegaDataset(NuScenesDataset):
    def include_nuscenes_data(self, mode):
        self.logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)

        self.infos.extend(nuscenes_infos)
        self.logger.info('Total samples for NuScenes dataset: %d' %
                         (len(nuscenes_infos)))

    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            'car': 'Car',
            'pedestrian': 'Pedestrian',
            'truck': 'Truck',
            'bus': 'Bus',
            'motorcycle': 'Motorcycle'
        }

        def transform_to_kitti_format(annos,
                                      info_with_fakelidar=False,
                                      is_gt=False):
            for anno in annos:
                if 'name' not in anno:
                    anno['name'] = anno['gt_names']
                    anno.pop('gt_names')

                for k in range(anno['name'].shape[0]):
                    if anno['name'][k] in map_name_to_kitti:
                        anno['name'][k] = map_name_to_kitti[anno['name'][k]]
                    else:
                        anno['name'][k] = 'Person_sitting'

                if 'boxes_lidar' in anno:
                    gt_boxes_lidar = anno['boxes_lidar'].copy()
                else:
                    gt_boxes_lidar = anno['gt_boxes'].copy()

                # filter by fov
                if is_gt and self.dataset_cfg.get('GT_FILTER', None):
                    if self.dataset_cfg.GT_FILTER.get('FOV_FILTER', None):
                        fov_gt_flag = self.extract_fov_gt(
                            gt_boxes_lidar, self.dataset_cfg['FOV_DEGREE'],
                            self.dataset_cfg['FOV_ANGLE'])
                        gt_boxes_lidar = gt_boxes_lidar[fov_gt_flag]
                        anno['name'] = anno['name'][fov_gt_flag]

                anno['bbox'] = np.zeros((len(anno['name']), 4))
                anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                anno['truncated'] = np.zeros(len(anno['name']))
                anno['occluded'] = np.zeros(len(anno['name']))

                if len(gt_boxes_lidar) > 0:
                    if info_with_fakelidar:
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(
                            gt_boxes_lidar)

                    #gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                    anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                    # anno['location'][:,0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                    # anno['location'][:,1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                    # anno['location'][:, 2] = gt_boxes_lidar[:,0]  # z = x_lidar
                    anno['location'][:,0] = gt_boxes_lidar[:, 0]  # x = -y_lidar
                    anno['location'][:,1] = gt_boxes_lidar[:, 1]  # y = -z_lidar
                    anno['location'][:, 2] = gt_boxes_lidar[:,2]  # z = x_lidar
                    dxdydz = gt_boxes_lidar[:, 3:6]
                    # anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                    anno['dimensions'] = dxdydz[:, [0, 1, 2]]  # lwh ==> lhw
                    # anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    anno['rotation_y'] = gt_boxes_lidar[:, 6]

                    anno['alpha'] = -np.arctan2(
                        -gt_boxes_lidar[:, 1],
                        gt_boxes_lidar[:, 0]) + anno['rotation_y']
                    #anno['alpha'] = gt_boxes_lidar[:, 6]
                else:
                    anno['location'] = anno['dimensions'] = np.zeros((0, 3))
                    anno['rotation_y'] = anno['alpha'] = np.zeros(0)

        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos,
                                  info_with_fakelidar=False,
                                  is_gt=True)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append('Person_sitting')
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos,
            dt_annos=eval_det_annos,
            current_classes=kitti_class_names)
        return ap_result_str, ap_dict

    def nuscene_eval(self, det_annos, class_names, **kwargs):
        import json
        from nuscenes.nuscenes import NuScenes
        from . import omega_utils
        nusc = NuScenes(version=self.dataset_cfg.VERSION,
                        dataroot=str(self.root_path),
                        verbose=True)
        nusc_annos = omega_utils.transform_det_annos_to_nusc_annos(
            det_annos, nusc)
        nusc_annos['meta'] = {
            'use_camera': False,
            'use_lidar': True,
            'use_radar': False,
            'use_map': False,
            'use_external': False,
        }

        output_path = Path(kwargs['output_path'])
        output_path.mkdir(exist_ok=True, parents=True)
        res_path = str(output_path / 'results_nusc.json')
        with open(res_path, 'w') as f:
            json.dump(nusc_annos, f)

        self.logger.info(
            f'The predictions of NuScenes have been saved to {res_path}')

        if self.dataset_cfg.VERSION == 'v1.0-test':
            return 'No ground-truth annotations for evaluation', {}

        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.detection.evaluate import NuScenesEval

        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
            'v0.5-omega-trainval': 'val',
            'v1.0-test': 'test'
        }
        try:
            eval_version = 'detection_cvpr_2019'
            eval_config = config_factory(eval_version)
        except:
            eval_version = 'cvpr_2019'
            eval_config = config_factory(eval_version)

        nusc_eval = NuScenesEval(
            nusc,
            config=eval_config,
            result_path=res_path,
            eval_set=eval_set_map[self.dataset_cfg.VERSION],
            output_dir=str(output_path),
            verbose=True,
        )
        metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

        with open(output_path / 'metrics_summary.json', 'r') as f:
            metrics = json.load(f)

        result_str, result_dict = omega_utils.format_omega_results(
            metrics, self.class_names, version=eval_version)
        return result_str, result_dict

    def create_groundtruth_database(self, used_classes=None, max_sweeps=10, split='xxx'):
        import torch

        database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
        #  db_info_save_path = self.root_path / f'omega_dbinfos_{max_sweeps}sweeps_withvelo.pkl'
        db_info_save_path = self.root_path / f'omega_dbinfos_{max_sweeps}sweeps_{split}.pkl'

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        for idx in tqdm.tqdm(range(len(self.infos))):
            sample_idx = idx
            info = self.infos[idx]
            points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
            gt_boxes = info['gt_boxes']
            #gt_boxes[:,6] = -(gt_boxes[:, 6]+np.pi/2.0) #gt_sampling때문
            #import pdb; pdb.set_trace()
            gt_names = info['gt_names']

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:,
                                        0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(
                    dim=0).float().cuda()).long().squeeze(dim=0).cpu().numpy()
            
            #gt_boxes[:,6] = -gt_boxes[:, 6]-np.pi/2.0

            for i in range(gt_boxes.shape[0]):
                filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or gt_names[i] in used_classes:
                    db_path = str(filepath.relative_to(
                        self.root_path))  # gt_database/xxxxx.bin
                    db_info = {
                        'name': gt_names[i],
                        'path': db_path,
                        'image_idx': sample_idx,
                        'gt_idx': i,
                        'box3d_lidar': gt_boxes[i],
                        'num_points_in_gt': gt_points.shape[0]
                    }
                    if gt_names[i] in all_db_infos:
                        all_db_infos[gt_names[i]].append(db_info)
                    else:
                        all_db_infos[gt_names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

def create_omega_info(version, data_path, save_path, max_sweeps=10):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import omega_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v0.5-omega-trainval']
    if version == 'v0.5-omega-trainval':
        train_scenes = trainval_split[version]['train_scenes']
        val_scenes = trainval_split[version]['val_scenes']
    else:
        raise NotImplementedError

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = omega_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    print('%s: train scene(%d), val scene(%d)' %
          (version, len(train_scenes), len(val_scenes)))

    train_nusc_infos, val_nusc_infos = omega_utils.fill_trainval_infos(
        data_path=data_path,
        nusc=nusc,
        train_scenes=train_scenes,
        val_scenes=val_scenes,
        test='test' in version,
        max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'omega_infos_{max_sweeps}sweeps_test.pkl',
                  'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, val sample: %d' %
              (len(train_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'omega_infos_{max_sweeps}sweeps_train.pkl',
                  'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'omega_infos_{max_sweeps}sweeps_val.pkl',
                  'wb') as f:
            pickle.dump(val_nusc_infos, f)

def create_part_dbinfos(version, data_path, save_path, max_sweeps=10, part_split='xxx'):
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    from . import omega_utils
    data_path = data_path / version
    save_path = save_path / version

    assert version in ['v0.5-omega-trainval']
    if version == 'v0.5-omega-trainval':
        train_scenes = trainval_split[version]['train_scenes']
        val_scenes = trainval_split[version]['val_scenes']
    else:
        raise NotImplementedError

    split_dir = data_path / 'ImageSets' / (part_split + '.txt')
    train_scenes = [x.strip() for x in open(split_dir).readlines()]

    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    available_scenes = omega_utils.get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    unlabeled_scenes = [
        x for x in available_scene_names if x not in train_scenes
    ]
    unlabeled_scenes = [x for x in unlabeled_scenes if x not in val_scenes]
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])
    unlabeled_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in unlabeled_scenes
    ])

    print('%s: train scene(%d), unlabeled_scene(%d), val scene(%d)' %
          (version, len(train_scenes), len(unlabeled_scenes), len(val_scenes)))

    train_nusc_infos, unlabeled_nusc_infos, val_nusc_infos = omega_utils.fill_trainval_part_infos(
        data_path=data_path,
        nusc=nusc,
        train_scenes=train_scenes,
        unlabeled_scenes=unlabeled_scenes,
        val_scenes=val_scenes,
        test='test' in version,
        max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        print('test sample: %d' % len(train_nusc_infos))
        with open(save_path / f'omega_infos_{max_sweeps}sweeps_test.pkl',
                  'wb') as f:
            pickle.dump(train_nusc_infos, f)
    else:
        print('train sample: %d, unlabeled sample: %d, val sample: %d' %
              (len(train_nusc_infos), len(unlabeled_nusc_infos), len(val_nusc_infos)))
        with open(save_path / f'omega_infos_{max_sweeps}sweeps_train_{part_split}.pkl',
                  'wb') as f:
            pickle.dump(train_nusc_infos, f)
        with open(save_path / f'omega_infos_{max_sweeps}sweeps_unlabeled_{part_split}.pkl',
                  'wb') as f:
            pickle.dump(unlabeled_nusc_infos, f)
        with open(save_path / f'omega_infos_{max_sweeps}sweeps_val_{part_split}.pkl',
                  'wb') as f:
            pickle.dump(val_nusc_infos, f)

if __name__ == '__main__':
    import yaml
    import argparse
    from pathlib import Path
    from easydict import EasyDict

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file',
                        type=str,
                        default=None,
                        help='specify the config of dataset')
    parser.add_argument('--func',
                        type=str,
                        default='create_omega_infos',
                        help='')
    parser.add_argument('--version',
                        type=str,
                        default='v0.5-omega-trainval',
                        help='')
    parser.add_argument('--split', type=str, default='xxx', help='')
    args = parser.parse_args()

    if args.func == 'create_omega_infos':
        dataset_cfg = EasyDict(yaml.full_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        create_omega_info(
            version=dataset_cfg.VERSION,
            data_path=ROOT_DIR / 'data' / 'omega',
            save_path=ROOT_DIR / 'data' / 'omega',
            max_sweeps=dataset_cfg.MAX_SWEEPS,
        )

        omega_dataset = OmegaDataset(dataset_cfg=dataset_cfg,
                                     class_names=None,
                                     root_path=ROOT_DIR / 'data' / 'omega',
                                     logger=common_utils.create_logger(),
                                     training=True)
        omega_dataset.create_groundtruth_database(
            max_sweeps=dataset_cfg.MAX_SWEEPS)

    if args.func == 'create_part_dbinfos':
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        dataset_cfg.VERSION = args.version
        create_part_dbinfos(version=dataset_cfg.VERSION,
                            data_path=ROOT_DIR / 'data' / 'omega',
                            save_path=ROOT_DIR / 'data' / 'omega',
                            max_sweeps=dataset_cfg.MAX_SWEEPS,
                            part_split=args.split)

        dataset_cfg.INFO_PATH['train'] = [
            f'omega_infos_{dataset_cfg.MAX_SWEEPS}sweeps_train_{args.split}.pkl'
        ]
        dataset_cfg.INFO_PATH['test'] = [
            f'omega_infos_{dataset_cfg.MAX_SWEEPS}sweeps_val_{args.split}.pkl'
        ]
        dataset_cfg.REPEAT = 1
        omega_dataset = OmegaDataset(dataset_cfg=dataset_cfg,
                                     class_names=None,
                                     root_path=ROOT_DIR / 'data' / 'omega',
                                     logger=common_utils.create_logger(),
                                     training=True)
        omega_dataset.create_groundtruth_database(
            max_sweeps=dataset_cfg.MAX_SWEEPS, split=args.split)