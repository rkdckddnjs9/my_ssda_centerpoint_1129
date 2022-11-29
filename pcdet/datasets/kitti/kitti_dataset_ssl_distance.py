import copy
import pickle
from collections import defaultdict

import numpy as np
from skimage import io

from pcdet.datasets.augmentor.augmentor_utils import *
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate


class KittiDatasetSSL(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.repeat = self.dataset_cfg.REPEAT

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        # self.test = self.split == 'test' or self.split == 'val'

        if not self.training:
            self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        else:
            self.sample_id_list = [x.strip().split(' ')[0] for x in
                               open(split_dir).readlines()] if split_dir.exists() else None
            self.sample_index_list = [x.strip().split(' ')[1] for x in
                                  open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)

        if self.training:
            all_train = len(self.kitti_infos)
            self.unlabeled_index_list = list(set(list(range(all_train))) - set(self.sample_index_list))  # float()!!!
            # print(self.unlabeled_index_list)
            self.unlabeled_kitti_infos = []

            temp = []
            for i in self.sample_index_list:
                temp.append(self.kitti_infos[int(i)])
            if len(self.sample_index_list) < 3712: # not 100%
                for i in self.unlabeled_index_list:
                    self.unlabeled_kitti_infos.append(self.kitti_infos[int(i)])
            else:
                self.unlabeled_index_list = list(range(len(self.sample_index_list)))
                for i in self.sample_index_list:
                    self.unlabeled_kitti_infos.append(self.kitti_infos[int(i)])
                print("full set", len(self.unlabeled_kitti_infos))
            self.kitti_infos = temp
            assert len(self.kitti_infos) == len(self.sample_id_list)

        if self.logger is not None:
            self.logger.info('Total samples for KITTI dataset: %d' % (len(self.kitti_infos)))

    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading KITTI SSL dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        # if self.logger is not None:
        #     self.logger.info('Total samples for KITTI dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.png' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = self.root_split_path / 'calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            calib = self.get_calib(sample_idx)

            P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = calib.R0
            V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['calib'] = calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                loc_lidar = calib.rect_to_lidar(loc)
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar

                info['annos'] = annotations

                if count_inside_pts:
                    points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annotations['num_points_in_gt'] = num_points_in_gt

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch

        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s_%d.pkl' % (split, len(self.sample_index_list)))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        # with open(info_path, 'rb') as f:
        #     infos = pickle.load(f)
        infos = self.kitti_infos

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        # # distance-wise eval
        # gt_keys = list(eval_gt_annos[0].keys())
        # det_keys = list(eval_det_annos[0].keys())

        # gt_00_09 = []
        # gt_10_19 = []
        # gt_20_29 = []
        # gt_30_39 = []
        # gt_40_49 = []
        # gt_50_59 = []
        # gt_60_69 = []
        # gt_70_all = []
        # det_00_09 = []
        # det_10_19 = []
        # det_20_29 = []
        # det_30_39 = []
        # det_40_49 = []
        # det_50_59 = []
        # det_60_69 = []
        # det_70_all = []



        # for num in range(len(eval_gt_annos)):
        #     data = eval_gt_annos[num]
        #     data_00_09 = {}
        #     data_10_19 = {}
        #     data_20_29 = {}
        #     data_30_39 = {}
        #     data_40_49 = {}
        #     data_50_59 = {}
        #     data_60_69 = {}
        #     data_70_all = {}
        #     dist = self.distance(data['location'])
            
        #     mask_00_09 = dist<10
        #     mask_10_19 = (dist>=10)*(dist<20)
        #     mask_20_29 = (dist>=20)*(dist<30)
        #     mask_30_39 = (dist>=30)*(dist<40)
        #     mask_40_49 = (dist>=40)*(dist<50)
        #     mask_50_59 = (dist>=50)*(dist<60)
        #     mask_60_69 = (dist>=60)*(dist<70)
        #     mask_70_all = (dist>=70)

        #     if True in mask_00_09:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_00_09[key] = data[key][mask_00_09]
        #             elif key == 'gt_boxes_lidar':
        #                 data_00_09[key] = data[key][mask_00_09[:len(data[key])]]
        #             else:
        #                 data_00_09[key] = data[key][mask_00_09]
        #     else:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_00_09[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_00_09[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_00_09[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_00_09[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_00_09[key] = np.array([]).reshape(-1,3)
        #             else:
        #                 data_00_09[key] = np.array([])
            
        #     if True in mask_10_19:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_10_19[key] = data[key][mask_10_19]
        #             elif key == 'gt_boxes_lidar':
        #                 data_10_19[key] = data[key][mask_10_19[:len(data[key])]]
        #             else:
        #                 data_10_19[key] = data[key][mask_10_19]
        #     else:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_10_19[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_10_19[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_10_19[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_10_19[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_10_19[key] = np.array([]).reshape(-1,3)
        #             else:
        #                 data_10_19[key] = np.array([])
            
        #     if True in mask_20_29:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_20_29[key] = data[key][mask_20_29]
        #             elif key == 'gt_boxes_lidar':
        #                 data_20_29[key] = data[key][mask_20_29[:len(data[key])]]
        #             else:
        #                 data_20_29[key] = data[key][mask_20_29]
        #     else:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_20_29[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_20_29[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_20_29[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_20_29[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_20_29[key] = np.array([]).reshape(-1,3)
        #             else:
        #                 data_20_29[key] = np.array([])
            
        #     if True in mask_30_39:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_30_39[key] = data[key][mask_30_39]
        #             elif key == 'gt_boxes_lidar':
        #                 data_30_39[key] = data[key][mask_30_39[:len(data[key])]]
        #             else:
        #                 data_30_39[key] = data[key][mask_30_39]
        #     else:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_30_39[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_30_39[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_30_39[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_30_39[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_30_39[key] = np.array([]).reshape(-1,3)
        #             else:
        #                 data_30_39[key] = np.array([])

        #     if True in mask_40_49:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_40_49[key] = data[key][mask_40_49]
        #             elif key == 'gt_boxes_lidar':
        #                 data_40_49[key] = data[key][mask_40_49[:len(data[key])]]
        #             else:
        #                 data_40_49[key] = data[key][mask_40_49]
        #     else:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_40_49[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_40_49[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_40_49[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_40_49[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_40_49[key] = np.array([]).reshape(-1,3)
        #             else:
        #                 data_40_49[key] = np.array([])

        #     if True in mask_50_59:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_50_59[key] = data[key][mask_50_59]
        #             elif key == 'gt_boxes_lidar':
        #                 data_50_59[key] = data[key][mask_50_59[:len(data[key])]]
        #             else:
        #                 data_50_59[key] = data[key][mask_50_59]
        #     else:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_50_59[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_50_59[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_50_59[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_50_59[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_50_59[key] = np.array([]).reshape(-1,3)
        #             else:
        #                 data_50_59[key] = np.array([])

        #     if True in mask_60_69:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_60_69[key] = data[key][mask_60_69]
        #             elif key == 'gt_boxes_lidar':
        #                 data_60_69[key] = data[key][mask_60_69[:len(data[key])]]
        #             else:
        #                 data_60_69[key] = data[key][mask_60_69]
        #     else:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_60_69[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_60_69[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_60_69[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_60_69[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_60_69[key] = np.array([]).reshape(-1,3)
        #             else:
        #                 data_60_69[key] = np.array([])
            
        #     if True in mask_70_all:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_70_all[key] = data[key][mask_70_all]
        #             elif key == 'gt_boxes_lidar':
        #                 data_70_all[key] = data[key][mask_70_all[:len(data[key])]]
        #             else:
        #                 data_70_all[key] = data[key][mask_70_all]
        #     else:
        #         for key in gt_keys:
        #             if key == 'name':
        #                 data_70_all[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_70_all[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_70_all[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_70_all[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_70_all[key] = np.array([]).reshape(-1,3)
        #             else:
        #                 data_70_all[key] = np.array([])
            
        #     gt_00_09.append(data_00_09)
        #     gt_10_19.append(data_10_19)
        #     gt_20_29.append(data_20_29)
        #     gt_30_39.append(data_30_39)
        #     gt_40_49.append(data_40_49)
        #     gt_50_59.append(data_50_59)
        #     gt_60_69.append(data_60_69)
        #     gt_70_all.append(data_70_all)
            
        # for num in range(len(eval_det_annos)):
        #     data = eval_det_annos[num]
        #     data_00_09 = {}
        #     data_10_19 = {}
        #     data_20_29 = {}
        #     data_30_39 = {}
        #     data_40_49 = {}
        #     data_50_59 = {}
        #     data_60_69 = {}
        #     data_70_all = {}
        #     dist = self.distance(data['location'])
            
        #     mask_00_09 = dist<10
        #     mask_10_19 = (dist>=10)*(dist<20)
        #     mask_20_29 = (dist>=20)*(dist<30)
        #     mask_30_39 = (dist>=30)*(dist<40)
        #     mask_40_49 = (dist>=40)*(dist<50)
        #     mask_50_59 = (dist>=50)*(dist<60)
        #     mask_60_69 = (dist>=60)*(dist<70)
        #     mask_70_all = (dist>=70)

        #     if True in mask_00_09:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_00_09[key] = data[key][mask_00_09]
        #             elif key == 'frame_id':
        #                 data_00_09[key] = data[key]
        #             else:
        #                 data_00_09[key] = data[key][mask_00_09]
        #     else:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_00_09[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_00_09[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_00_09[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_00_09[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_00_09[key] = np.array([]).reshape(-1,3)
        #             elif key == 'frame_id':
        #                 data_00_09[key] = data[key]
        #             else:
        #                 data_00_09[key] = np.array([])
            
        #     if True in mask_10_19:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_10_19[key] = data[key][mask_10_19]
        #             elif key == 'frame_id':
        #                 data_10_19[key] = data[key]
        #             else:
        #                 data_10_19[key] = data[key][mask_10_19]
        #     else:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_10_19[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_10_19[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_10_19[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_10_19[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_10_19[key] = np.array([]).reshape(-1,3)
        #             elif key == 'frame_id':
        #                 data_10_19[key] = data[key]
        #             else:
        #                 data_10_19[key] = np.array([])
            
        #     if True in mask_20_29:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_20_29[key] = data[key][mask_20_29]
        #             elif key == 'frame_id':
        #                 data_20_29[key] = data[key]
        #             else:
        #                 data_20_29[key] = data[key][mask_20_29]
        #     else:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_20_29[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_20_29[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_20_29[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_20_29[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_20_29[key] = np.array([]).reshape(-1,3)
        #             elif key == 'frame_id':
        #                 data_20_29[key] = data[key]
        #             else:
        #                 data_20_29[key] = np.array([])
            
        #     if True in mask_30_39:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_30_39[key] = data[key][mask_30_39]
        #             elif key == 'frame_id':
        #                 data_30_39[key] = data[key]
        #             else:
        #                 data_30_39[key] = data[key][mask_30_39]
        #     else:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_30_39[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_30_39[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_30_39[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_30_39[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_30_39[key] = np.array([]).reshape(-1,3)
        #             elif key == 'frame_id':
        #                 data_30_39[key] = data[key]
        #             else:
        #                 data_30_39[key] = np.array([])

        #     if True in mask_40_49:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_40_49[key] = data[key][mask_40_49]
        #             elif key == 'frame_id':
        #                 data_40_49[key] = data[key]
        #             else:
        #                 data_40_49[key] = data[key][mask_40_49]
        #     else:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_40_49[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_40_49[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_40_49[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_40_49[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_40_49[key] = np.array([]).reshape(-1,3)
        #             elif key == 'frame_id':
        #                 data_40_49[key] = data[key]
        #             else:
        #                 data_40_49[key] = np.array([])

        #     if True in mask_50_59:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_50_59[key] = data[key][mask_50_59]
        #             elif key == 'frame_id':
        #                 data_50_59[key] = data[key]
        #             else:
        #                 data_50_59[key] = data[key][mask_50_59]
        #     else:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_50_59[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_50_59[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_50_59[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_50_59[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_50_59[key] = np.array([]).reshape(-1,3)
        #             elif key == 'frame_id':
        #                 data_50_59[key] = data[key]
        #             else:
        #                 data_50_59[key] = np.array([])

        #     if True in mask_60_69:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_60_69[key] = data[key][mask_60_69]
        #             elif key == 'frame_id':
        #                 data_60_69[key] = data[key]
        #             else:
        #                 data_60_69[key] = data[key][mask_60_69]
        #     else:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_60_69[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_60_69[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_60_69[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_60_69[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_60_69[key] = np.array([]).reshape(-1,3)
        #             elif key == 'frame_id':
        #                 data_60_69[key] = data[key]
        #             else:
        #                 data_60_69[key] = np.array([])
            
        #     if True in mask_70_all:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_70_all[key] = data[key][mask_70_all]
        #             elif key == 'frame_id':
        #                 data_70_all[key] = data[key]
        #             else:
        #                 data_70_all[key] = data[key][mask_70_all]
        #     else:
        #         for key in det_keys:
        #             if key == 'name':
        #                 data_70_all[key] = np.array([], dtype='<U10')
        #             elif key == 'bbox':
        #                 data_70_all[key] = np.array([]).reshape(-1,4)
        #             elif key == 'gt_boxes_lidar':
        #                 data_70_all[key] = np.array([]).reshape(-1,7)
        #             elif key == 'dimensions':
        #                 data_70_all[key] = np.array([]).reshape(-1,3)
        #             elif key == 'location':
        #                 data_70_all[key] = np.array([]).reshape(-1,3)
        #             elif key == 'frame_id':
        #                 data_70_all[key] = data[key]
        #             else:
        #                 data_70_all[key] = np.array([])
            
        #     det_00_09.append(data_00_09)
        #     det_10_19.append(data_10_19)
        #     det_20_29.append(data_20_29)
        #     det_30_39.append(data_30_39)
        #     det_40_49.append(data_40_49)
        #     det_50_59.append(data_50_59)
        #     det_60_69.append(data_60_69)
        #     det_70_all.append(data_70_all)

        # ap_result_str_00_09, ap_dict_00_09 = kitti_eval.get_official_eval_result(gt_00_09, det_00_09, class_names)
        # ap_result_str_10_19, ap_dict_10_19 = kitti_eval.get_official_eval_result(gt_10_19, det_10_19, class_names)
        # ap_result_str_20_29, ap_dict_20_29 = kitti_eval.get_official_eval_result(gt_20_29, det_20_29, class_names)
        # ap_result_str_30_39, ap_dict_30_39 = kitti_eval.get_official_eval_result(gt_30_39, det_30_39, class_names)
        # ap_result_str_40_49, ap_dict_40_49 = kitti_eval.get_official_eval_result(gt_40_49, det_40_49, class_names)
        # ap_result_str_50_59, ap_dict_50_59 = kitti_eval.get_official_eval_result(gt_50_59, det_50_59, class_names)
        # ap_result_str_60_69, ap_dict_60_69 = kitti_eval.get_official_eval_result(gt_60_69, det_60_69, class_names)
        # ap_result_str_70_all, ap_dict_70_all = kitti_eval.get_official_eval_result(gt_70_all, det_70_all, class_names)
        
        # path = "/home/spalab/cwkang/3DIoUMatch-PVRCNN/output/cfgs/kitti_models/test/default/ssl_0.02_1/"
        # file = open(path + "00_09.txt", "w")
        # with open(path + "00_09.txt", "w") as file:
        #     file.writelines(ap_result_str_00_09)
        # file = open(path + "10_19.txt", "w")
        # with open(path + "10_19.txt", "w") as file:
        #     file.writelines(ap_result_str_10_19)
        # file = open(path + "20_29.txt", "w")
        # with open(path + "20_29.txt", "w") as file:
        #     file.writelines(ap_result_str_20_29)
        # file = open(path + "30_39.txt", "w")
        # with open(path + "30_39.txt", "w") as file:
        #     file.writelines(ap_result_str_30_39)
        # file = open(path + "40_49.txt", "w")
        # with open(path + "40_49.txt", "w") as file:
        #     file.writelines(ap_result_str_40_49)
        # file = open(path + "50_59.txt", "w")
        # with open(path + "50_59.txt", "w") as file:
        #     file.writelines(ap_result_str_50_59)
        # file = open(path + "60_69.txt", "w")
        # with open(path + "60_69.txt", "w") as file:
        #     file.writelines(ap_result_str_60_69)
        # file = open(path + "70_all.txt", "w")
        # with open(path + "70_all.txt", "w") as file:
        #     file.writelines(ap_result_str_70_all)

        return ap_result_str, ap_dict
    
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

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        if self.training:
            return len(self.kitti_infos) * self.repeat
        else:
            return len(self.kitti_infos)


    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        data_dict_labeled = self.get_item_single(info)

        if self.training:
            index_unlabeled = np.random.choice(self.unlabeled_index_list, 1)[0]
            info_unlabeled = copy.deepcopy(self.unlabeled_kitti_infos[index_unlabeled])

            data_dict_unlabeled = self.get_item_single(info_unlabeled, no_db_sample=True)
            return [data_dict_labeled, data_dict_unlabeled]
        else:
            return data_dict_labeled

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        # print(batch_list)
        if isinstance(batch_list[0], list):
            for cur_sample in batch_list:
                for key, val in cur_sample[0].items():
                    if key == 'patch':
                        continue
                    data_dict[key].append(val)
                data_dict['mask'].append(np.ones([len(batch_list)]))
                for key, val in cur_sample[1].items():
                    if key == 'patch':
                        continue
                    data_dict[key].append(val)
                data_dict['mask'].append(np.zeros([len(batch_list)]))
            batch_size = len(batch_list) * 2
        else:
            for cur_sample in batch_list:
                for key, val in cur_sample.items():
                    data_dict[key].append(val)
            batch_size = len(batch_list)

        try :
            if batch_list[0][0]['patch_shuffle'] == True:

                patch_flag = True
                batch_1 = batch_list[0][0]['patch']
                try:
                    batch_2 = batch_list[0][1]['patch']
                except:
                    batch_2 = batch_list[0][0]['patch']
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
                        elif key in ['flip_x', 'flip_y', 'rot_angle', 'scale', 'use_lead_xyz', 'image_shape', 'gt_names', 'calib', 'road_plane', 'patch_shuffle', 'image_shape']:
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
                    for cur_sample in [data_1]:
                        for key, val in cur_sample.items():
                            if key == 'gt_names':
                                continue
                            data_dict[key].append(val)
                        data_dict['mask'].append(np.zeros([len(batch_list)])+2)
                    for cur_sample in [data_2]:
                        for key, val in cur_sample.items():
                            if key == 'gt_names':
                                continue
                            data_dict[key].append(val)
                        data_dict['mask'].append(np.zeros([len(batch_list)])+3)
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
                batch_size=2
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

        # ret = {}

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

        #ret['batch_size'] = batch_size
        return ret

    def get_item_single(self, info, no_db_sample=False):
        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if self.dataset_cfg.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane

        data_dict = self.prepare_data(data_dict=input_dict, no_db_sample=no_db_sample)
        # if isinstance(data_dict, list):
        #     print(data_dict)
        data_dict['image_shape'] = img_shape

        try:
            for data in data_dict['patch']:
                data['image_shape'] = img_shape
        except:
            pass
        return data_dict


    def prepare_data(self, data_dict, no_db_sample=False):
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
                },
                no_db_sample=no_db_sample
            )
            # print(data_dict)
            points_ema = data_dict['points'].copy()
            gt_boxes_ema = data_dict['gt_boxes'].copy()
            gt_boxes_ema, points_ema, _ = global_scaling(gt_boxes_ema, points_ema, [0, 2],
                                                         scale_=1/data_dict['scale'])
            gt_boxes_ema, points_ema, _ = global_rotation(gt_boxes_ema, points_ema, [-1, 1],
                                                          rot_angle_=-data_dict['rot_angle'])
            gt_boxes_ema, points_ema, _ = random_flip_along_x(gt_boxes_ema, points_ema, enable_=data_dict['flip_x'])
            gt_boxes_ema, points_ema, _ = random_flip_along_y(gt_boxes_ema, points_ema, enable_=data_dict['flip_y'])
            data_dict['points_ema'] = points_ema
            data_dict['gt_boxes_ema'] = gt_boxes_ema

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            
            if self.training:
                data_dict['gt_boxes_ema'] = data_dict['gt_boxes_ema'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes
            if self.training:
                gt_boxes_ema = np.concatenate((data_dict['gt_boxes_ema'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
                data_dict['gt_boxes_ema'] = gt_boxes_ema

        # print((data_dict['points'] ** 2).sum(), (data_dict['points_ema'] ** 2).sum()*(data_dict['scale']**2))
        if self.training:
            points = data_dict['points'].copy()
            gt_boxes = data_dict['gt_boxes'].copy()
            # points_ema = data_dict['points_ema'].copy()
            data_dict['points'] = data_dict['points_ema']
            data_dict['gt_boxes'] = data_dict['gt_boxes_ema']
        data_dict = self.point_feature_encoder.forward(data_dict)
        data_dict = self.data_processor.forward(
            data_dict=data_dict
        )

        if self.training:
            data_dict['points_ema'] = data_dict['points']
            data_dict['gt_boxes_ema'] = data_dict['gt_boxes']
            data_dict['voxels_ema'] = data_dict['voxels']
            data_dict['voxel_coords_ema'] = data_dict['voxel_coords']
            data_dict['voxel_num_points_ema'] = data_dict['voxel_num_points']

            data_dict['points'] = points
            data_dict['gt_boxes'] = gt_boxes
            data_dict.pop('voxels', None)
            data_dict.pop('voxel_coords', None)
            data_dict.pop('voxel_num_points', None)
            data_dict = self.point_feature_encoder.forward(data_dict)
            data_dict = self.data_processor.forward(
                data_dict=data_dict
            )
            #modification
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

            # name = data_dict['gt_names']
            # name_1 = name[gt_1_mask]
            # name_2 = name[gt_2_mask]
            # name_3 = name[gt_3_mask]
            # name_4 = name[gt_4_mask]
            # name_5 = name[gt_5_mask]
            # name_6 = name[gt_6_mask]

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
                    # elif key == 'gt_names':
                    #     data_dict_1[key] = name_1
                    #     data_dict_2[key] = name_2
                    #     data_dict_3[key] = name_3
                    #     data_dict_4[key] = name_4
                    #     data_dict_5[key] = name_5
                    #     data_dict_6[key] = name_6
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


        '''if self.training:
            if no_db_sample:
                data_dict['gt_boxes_ema'].fill(0)
                data_dict['gt_boxes'].fill(0)'''

        # TODO: currently commented out to prevent bug
        # if self.training and len(data_dict['gt_boxes']) == 0:
        #    new_index = np.random.randint(self.__len__())
        #    return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)

        return data_dict


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDatasetSSL(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'kitti',
            save_path=ROOT_DIR / 'data' / 'kitti'
        )
