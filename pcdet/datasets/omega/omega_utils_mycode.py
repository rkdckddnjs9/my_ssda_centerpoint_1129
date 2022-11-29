"""
The NuScenes data pre-processing and evaluation is modified from
https://github.com/traveller59/second.pytorch and https://github.com/poodarchu/Det3D
"""

import operator
from functools import reduce
from pathlib import Path

import numpy as np
import tqdm
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
    'Vehicle::Car': 'car',
    'Vehicle::Bus': 'bus',
    'Vehicle::Motorcycle': 'motorcycle',
    'Vehicle::Truck': 'truck',
    'Pedestrian::Pedestrian': 'pedestrian',
    'Vehicle::Other': 'bus'
}

cls_attr_dist = {
    'barrier': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bicycle': {
        'cycle.with_rider': 2791,
        'cycle.without_rider': 8946,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'bus': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 9092,
        'vehicle.parked': 3294,
        'vehicle.stopped': 3881,
    },
    'car': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 114304,
        'vehicle.parked': 330133,
        'vehicle.stopped': 46898,
    },
    'construction_vehicle': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 882,
        'vehicle.parked': 11549,
        'vehicle.stopped': 2102,
    },
    'ignore': {
        'cycle.with_rider': 307,
        'cycle.without_rider': 73,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 165,
        'vehicle.parked': 400,
        'vehicle.stopped': 102,
    },
    'motorcycle': {
        'cycle.with_rider': 4233,
        'cycle.without_rider': 8326,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'pedestrian': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 157444,
        'pedestrian.sitting_lying_down': 13939,
        'pedestrian.standing': 46530,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'traffic_cone': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 0,
        'vehicle.parked': 0,
        'vehicle.stopped': 0,
    },
    'trailer': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 3421,
        'vehicle.parked': 19224,
        'vehicle.stopped': 1895,
    },
    'truck': {
        'cycle.with_rider': 0,
        'cycle.without_rider': 0,
        'pedestrian.moving': 0,
        'pedestrian.sitting_lying_down': 0,
        'pedestrian.standing': 0,
        'vehicle.moving': 21339,
        'vehicle.parked': 55626,
        'vehicle.stopped': 11097,
    },
}

trainval_split = {
    'v0.1-omega-trainval': {
        'train_scenes': [
            'scene-0011', 'scene-0003', 'scene-0025', 'scene-0006',
            'scene-0004', 'scene-0010', 'scene-0027', 'scene-0016',
            'scene-0008', 'scene-0022', 'scene-0005', 'scene-0026',
            'scene-0007', 'scene-0012', 'scene-0015', 'scene-0009',
            'scene-0014', 'scene-0001', 'scene-0002', 'scene-0020',
            'scene-0017', 'scene-0021', 'scene-0018'
        ],
        'val_scenes':
        ['scene-0024', 'scene-0000', 'scene-0013', 'scene-0019', 'scene-0023']
    },
    'v0.2-omega-trainval': {
        'train_scenes': [
            'scene-0052', 'scene-0030', 'scene-0024', 'scene-0020',
            'scene-0013', 'scene-0045', 'scene-0006', 'scene-0029',
            'scene-0053', 'scene-0017', 'scene-0038', 'scene-0043',
            'scene-0035', 'scene-0040', 'scene-0015', 'scene-0004',
            'scene-0012', 'scene-0032', 'scene-0050', 'scene-0023',
            'scene-0009', 'scene-0016', 'scene-0042', 'scene-0037',
            'scene-0051', 'scene-0014', 'scene-0034', 'scene-0007',
            'scene-0039', 'scene-0022', 'scene-0005', 'scene-0010',
            'scene-0001', 'scene-0041', 'scene-0031', 'scene-0046',
            'scene-0011', 'scene-0027', 'scene-0018', 'scene-0044',
            'scene-0047', 'scene-0049', 'scene-0048'
        ],
        'val_scenes': [
            'scene-0033', 'scene-0019', 'scene-0008', 'scene-0025',
            'scene-0003', 'scene-0026', 'scene-0036', 'scene-0021',
            'scene-0028', 'scene-0002'
        ]
    },
    'v0.3-omega-trainval': {
        'train_scenes': [
            'scene-0072', 'scene-0066', 'scene-0076', 'scene-0067',
            'scene-0057', 'scene-0064', 'scene-0056', 'scene-0063',
            'scene-0070', 'scene-0055', 'scene-0071', 'scene-0075',
            'scene-0059', 'scene-0065', 'scene-0054', 'scene-0058',
            'scene-0073', 'scene-0061', 'scene-0060', 'scene-0077',
            'scene-0048', 'scene-0041', 'scene-0006', 'scene-0038',
            'scene-0046', 'scene-0024', 'scene-0043', 'scene-0053',
            'scene-0035', 'scene-0018', 'scene-0004', 'scene-0045',
            'scene-0027', 'scene-0047', 'scene-0023', 'scene-0009',
            'scene-0040', 'scene-0051', 'scene-0007', 'scene-0042',
            'scene-0016', 'scene-0037', 'scene-0010', 'scene-0012',
            'scene-0011', 'scene-0030', 'scene-0052', 'scene-0022',
            'scene-0050', 'scene-0020', 'scene-0031', 'scene-0049',
            'scene-0032', 'scene-0013', 'scene-0017', 'scene-0029',
            'scene-0034', 'scene-0039', 'scene-0014', 'scene-0001',
            'scene-0005', 'scene-0015', 'scene-0044'
        ],
        'val_scenes': [
            'scene-0069', 'scene-0068', 'scene-0078', 'scene-0074',
            'scene-0062', 'scene-0021', 'scene-0033', 'scene-0008',
            'scene-0026', 'scene-0019', 'scene-0002', 'scene-0025',
            'scene-0028', 'scene-0036', 'scene-0003'
        ]
    },
    'v0.4-50%-omega-trainval': {
        'train_l_scenes': [
            'scene-0052', 'scene-0030', 'scene-0024', 'scene-0020',
            'scene-0013', 'scene-0045', 'scene-0006', 'scene-0029',
            'scene-0053', 'scene-0017', 'scene-0038', 'scene-0043',
            'scene-0035', 'scene-0040', 'scene-0015', 'scene-0004',
            'scene-0012', 'scene-0032', 'scene-0050', 'scene-0023',
            'scene-0009', 'scene-0016', 'scene-0042', 'scene-0037',
            'scene-0051', 'scene-0014', 'scene-0034', 'scene-0007',
            'scene-0039', 'scene-0022', 'scene-0005', 'scene-0010',
            'scene-0001', 'scene-0041', 'scene-0031', 'scene-0046',
            'scene-0011', 'scene-0027', 'scene-0018', 'scene-0044',
            'scene-0047', 'scene-0049', 'scene-0048'
        ],
        'train_u_scenes': [
            'scene-0073', 'scene-0067', 'scene-0083', 'scene-0085',
            'scene-0089', 'scene-0056', 'scene-0080', 'scene-0090',
            'scene-0054', 'scene-0060', 'scene-0099', 'scene-0072',
            'scene-0065', 'scene-0075', 'scene-0071', 'scene-0086',
            'scene-0087', 'scene-0096', 'scene-0095', 'scene-0076',
            'scene-0061', 'scene-0059', 'scene-0070', 'scene-0079',
            'scene-0088', 'scene-0077', 'scene-0098', 'scene-0093',
            'scene-0084', 'scene-0091', 'scene-0064', 'scene-0097',
            'scene-0081', 'scene-0055', 'scene-0066', 'scene-0058',
            'scene-0092', 'scene-0057', 'scene-0094', 'scene-0063',
            'scene-0082'
        ],
        'val_scenes': [
            'scene-0069', 'scene-0068', 'scene-0078', 'scene-0074',
            'scene-0062', 'scene-0021', 'scene-0033', 'scene-0008',
            'scene-0026', 'scene-0019', 'scene-0002', 'scene-0025',
            'scene-0028', 'scene-0036', 'scene-0003'
        ]
    },
    'v0.4-omega-trainval': {
        'train_scenes': [
            'scene-0085', 'scene-0088', 'scene-0093', 'scene-0084',
            'scene-0087', 'scene-0082', 'scene-0090', 'scene-0079',
            'scene-0095', 'scene-0099', 'scene-0083', 'scene-0097',
            'scene-0086', 'scene-0091', 'scene-0098', 'scene-0096',
            'scene-0080', 'scene-0064', 'scene-0009', 'scene-0042',
            'scene-0038', 'scene-0030', 'scene-0032', 'scene-0070',
            'scene-0050', 'scene-0051', 'scene-0061', 'scene-0071',
            'scene-0041', 'scene-0007', 'scene-0076', 'scene-0054',
            'scene-0063', 'scene-0055', 'scene-0077', 'scene-0058',
            'scene-0012', 'scene-0001', 'scene-0035', 'scene-0039',
            'scene-0073', 'scene-0016', 'scene-0043', 'scene-0053',
            'scene-0052', 'scene-0065', 'scene-0046', 'scene-0067',
            'scene-0059', 'scene-0060', 'scene-0024', 'scene-0018',
            'scene-0027', 'scene-0040', 'scene-0031', 'scene-0049',
            'scene-0015', 'scene-0004', 'scene-0056', 'scene-0017',
            'scene-0011', 'scene-0014', 'scene-0075', 'scene-0072',
            'scene-0023', 'scene-0029', 'scene-0034', 'scene-0006',
            'scene-0010', 'scene-0022', 'scene-0020', 'scene-0045',
            'scene-0037', 'scene-0005', 'scene-0066', 'scene-0048',
            'scene-0013', 'scene-0057', 'scene-0044', 'scene-0047'
        ],
        'val_scenes': [
            'scene-0081', 'scene-0089', 'scene-0092', 'scene-0094',
            'scene-0028', 'scene-0069', 'scene-0021', 'scene-0033',
            'scene-0003', 'scene-0019', 'scene-0036', 'scene-0078',
            'scene-0008', 'scene-0002', 'scene-0074', 'scene-0025',
            'scene-0026', 'scene-0068', 'scene-0062'
        ]
    },
    'v0.5-omega-trainval': {
        'train_scenes': [
            'scene-0133', 'scene-0104', 'scene-0057', 'scene-0009',
            'scene-0124', 'scene-0131', 'scene-0061', 'scene-0026',
            'scene-0142', 'scene-0147', 'scene-0012', 'scene-0038',
            'scene-0118', 'scene-0056', 'scene-0051', 'scene-0094',
            'scene-0132', 'scene-0128', 'scene-0102', 'scene-0110',
            'scene-0105', 'scene-0058', 'scene-0076', 'scene-0053',
            'scene-0126', 'scene-0086', 'scene-0020', 'scene-0023',
            'scene-0071', 'scene-0109', 'scene-0123', 'scene-0107',
            'scene-0004', 'scene-0087', 'scene-0039', 'scene-0055',
            'scene-0036', 'scene-0143', 'scene-0001', 'scene-0141',
            'scene-0137', 'scene-0046', 'scene-0090', 'scene-0091',
            'scene-0139', 'scene-0072', 'scene-0122', 'scene-0015',
            'scene-0003', 'scene-0098', 'scene-0030', 'scene-0120',
            'scene-0034', 'scene-0014', 'scene-0127', 'scene-0024',
            'scene-0099', 'scene-0117', 'scene-0093', 'scene-0069',
            'scene-0047', 'scene-0092', 'scene-0121', 'scene-0042',
            'scene-0005', 'scene-0067', 'scene-0043', 'scene-0111',
            'scene-0010', 'scene-0032', 'scene-0019', 'scene-0066',
            'scene-0063', 'scene-0081', 'scene-0130', 'scene-0064',
            'scene-0027', 'scene-0040', 'scene-0049', 'scene-0025',
            'scene-0059', 'scene-0108', 'scene-0116', 'scene-0083',
            'scene-0062', 'scene-0048', 'scene-0028', 'scene-0013',
            'scene-0075', 'scene-0138', 'scene-0018', 'scene-0065',
            'scene-0140', 'scene-0035', 'scene-0074', 'scene-0050',
            'scene-0084', 'scene-0144', 'scene-0103', 'scene-0101',
            'scene-0007', 'scene-0088', 'scene-0080', 'scene-0070',
            'scene-0113', 'scene-0136', 'scene-0029', 'scene-0095',
            'scene-0145', 'scene-0134', 'scene-0146', 'scene-0011',
            'scene-0085', 'scene-0078', 'scene-0115', 'scene-0021',
            'scene-0129', 'scene-0017'
        ],
        'val_scenes': [
            'scene-0033', 'scene-0022', 'scene-0016', 'scene-0135',
            'scene-0002', 'scene-0037', 'scene-0052', 'scene-0077',
            'scene-0125', 'scene-0097', 'scene-0119', 'scene-0041',
            'scene-0008', 'scene-0006', 'scene-0045', 'scene-0044',
            'scene-0079', 'scene-0112', 'scene-0082', 'scene-0073',
            'scene-0096', 'scene-0060', 'scene-0100', 'scene-0114',
            'scene-0068', 'scene-0054', 'scene-0106', 'scene-0089',
            'scene-0031'
        ]
    }
}


def get_available_scenes(nusc):
    available_scenes = []
    print('total scene num:', len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            # if not sd_rec['next'] == '':
            #     sd_rec = nusc.get('sample_data', sd_rec['next'])
            # else:
            #     has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num:', len(available_scenes))
    return available_scenes


def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
    """
    Returns the data path as well as all annotations related to that sample_data.
    Note that the boxes are transformed into the current sensor's coordinate frame.
    Args:
        nusc:
        sample_data_token: Sample_data token.
        selected_anntokens: If provided only return the selected annotation.

    Returns:

    """
    # Retrieve sensor & pose records
    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)

    if sensor_record['modality'] == 'camera':
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (sd_record['width'], sd_record['height'])
    else:
        cam_intrinsic = imsize = None

    # Retrieve all sample annotations and map to sensor coordinate system.
    if selected_anntokens is not None:
        boxes = list(map(nusc.get_box, selected_anntokens))
    else:
        boxes = nusc.get_boxes(sample_data_token)

    # Make list of Box objects including coord system transforms.
    box_list = []
    for box in boxes:
        box.velocity = nusc.box_velocity(box.token)
        box_list.append(box)

    return data_path, box_list, cam_intrinsic


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def fill_trainval_part_infos(data_path,
                        nusc,
                        train_scenes,
                        unlabeled_scenes,
                        val_scenes,
                        test=False,
                        max_sweeps=10):
    train_nusc_infos = []
    unlabeled_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(nusc.sample),
                             desc='create_info',
                             dynamic_ncols=True)

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()

        ref_sd_token = sample['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_cs_rec = nusc.get('calibrated_sensor',
                              ref_sd_rec['calibrated_sensor_token'])
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'],
                                        Quaternion(ref_cs_rec['rotation']),
                                        inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec['translation'],
            Quaternion(ref_pose_rec['rotation']),
            inverse=True,
        )

        info = {
            'lidar_path':
            Path(ref_lidar_path).relative_to(data_path).__str__(),
            'token': sample['token'],
            'sweeps': [],
            'ref_from_car': ref_from_car,
            'car_from_global': car_from_global,
            'timestamp': ref_time,
        }

        sample_data_token = sample['data'][chan]
        curr_sd_rec = nusc.get('sample_data', sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec['prev'] == '':
                if len(sweeps) == 0:
                    sweep = {
                        'lidar_path':
                        Path(ref_lidar_path).relative_to(data_path).__str__(),
                        'sample_data_token':
                        curr_sd_rec['token'],
                        'transform_matrix':
                        None,
                        'time_lag':
                        curr_sd_rec['timestamp'] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])

                # Get past pose
                current_pose_rec = nusc.get('ego_pose',
                                            curr_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(
                    current_pose_rec['translation'],
                    Quaternion(current_pose_rec['rotation']),
                    inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    'calibrated_sensor',
                    curr_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(
                    current_cs_rec['translation'],
                    Quaternion(current_cs_rec['rotation']),
                    inverse=False,
                )

                tm = reduce(np.dot, [
                    ref_from_car, car_from_global, global_from_car,
                    car_from_current
                ])

                lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])

                time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']

                sweep = {
                    'lidar_path':
                    Path(lidar_path).relative_to(data_path).__str__(),
                    'sample_data_token': curr_sd_rec['token'],
                    'transform_matrix': tm,
                    'global_from_car': global_from_car,
                    'car_from_current': car_from_current,
                    'time_lag': time_lag,
                }
                sweeps.append(sweep)

        info['sweeps'] = sweeps

        assert len(info['sweeps']) == max_sweeps - 1, \
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, " \
            f"you should duplicate to sweep num {max_sweeps - 1}"

        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]

            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array(
                [anno['num_lidar_pts'] for anno in annotations])
            mask = (num_lidar_pts > 0)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            # dims = np.array([b.wlh for b in ref_boxes
            #                  ]).reshape(-1, 3)[:, [1, 0,
            #                                        2]]  # wlh == > dxdydz (lwh)
            dims = np.array([b.wlh for b in ref_boxes
                             ]).reshape(-1, 3)[:,[0,2,1]]  # lhw == > dxdydz (lwh)
            # dims = np.array([b.wlh for b in ref_boxes
            #                  ]).reshape(-1, 3)  # lhw == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            rots = np.array([quaternion_yaw(b.orientation)
                             for b in ref_boxes]).reshape(-1, 1)
            rots_ = np.array([b.orientation.angle
                             for b in ref_boxes]).reshape(-1, 1)           
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])

            #여기껄 바꿔줘야됨 ㅅㅂ
            # loc : xyz, dims : wlh->lwh
            rots_ = rots_ + np.pi/2
            # rots_ = -np.arctan2(locs[:, 0], -locs[:,1])+ rots_
            #rots_ = -np.arctan2(locs[:, 0], -locs[:,1]) + np.zeros_like(rots) # - rots_

            # gt_boxes = np.concatenate([locs, dims, -rots_-np.pi/2), velocity[:, :2]], axis=1)
            gt_boxes = np.concatenate([locs, dims, rots_, velocity[:, :2]], axis=1)
            # rot 90
            gt_boxes = gt_boxes[:, [1, 0, 2, 3, 4, 5, 6]]
            #gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi / 2
            gt_boxes[:, 1] = -gt_boxes[:, 1]
            #gt_boxes[:, 0] = -gt_boxes[:, 0]
            #gt_boxes[:, 6] = -gt_boxes[:, 6]
            # anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            # anno['alpha'] = -np.arctan2(
            #     -gt_boxes_lidar[:, 1],
            #     gt_boxes_lidar[:, 0]) + anno['rotation_y']

            assert len(annotations) == len(gt_boxes) == len(velocity)

            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = np.array([
                map_name_from_general_to_detection[name] for name in names
            ])[mask]
            info['gt_boxes_token'] = tokens[mask]
            info['num_lidar_pts'] = num_lidar_pts[mask]
            info['num_radar_pts'] = num_lidar_pts[mask]

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        elif sample['scene_token'] in unlabeled_scenes:
            unlabeled_nusc_infos.append(info)
        elif sample['scene_token'] in val_scenes:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, unlabeled_nusc_infos, val_nusc_infos

def fill_trainval_infos(data_path,
                        nusc,
                        train_scenes,
                        val_scenes,
                        test=False,
                        max_sweeps=10):
    train_nusc_infos = []
    val_nusc_infos = []
    progress_bar = tqdm.tqdm(total=len(nusc.sample),
                             desc='create_info',
                             dynamic_ncols=True)

    ref_chan = 'LIDAR_TOP'  # The radar channel from which we track back n sweeps to aggregate the point cloud.
    chan = 'LIDAR_TOP'  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for index, sample in enumerate(nusc.sample):
        progress_bar.update()

        ref_sd_token = sample['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_cs_rec = nusc.get('calibrated_sensor',
                              ref_sd_rec['calibrated_sensor_token'])
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'],
                                        Quaternion(ref_cs_rec['rotation']),
                                        inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(
            ref_pose_rec['translation'],
            Quaternion(ref_pose_rec['rotation']),
            inverse=True,
        )

        info = {
            'lidar_path':
            Path(ref_lidar_path).relative_to(data_path).__str__(),
            'token': sample['token'],
            'sweeps': [],
            'ref_from_car': ref_from_car,
            'car_from_global': car_from_global,
            'timestamp': ref_time,
        }

        sample_data_token = sample['data'][chan]
        curr_sd_rec = nusc.get('sample_data', sample_data_token)
        sweeps = []
        while len(sweeps) < max_sweeps - 1:
            if curr_sd_rec['prev'] == '':
                if len(sweeps) == 0:
                    sweep = {
                        'lidar_path':
                        Path(ref_lidar_path).relative_to(data_path).__str__(),
                        'sample_data_token':
                        curr_sd_rec['token'],
                        'transform_matrix':
                        None,
                        'time_lag':
                        curr_sd_rec['timestamp'] * 0,
                    }
                    sweeps.append(sweep)
                else:
                    sweeps.append(sweeps[-1])
            else:
                curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])

                # Get past pose
                current_pose_rec = nusc.get('ego_pose',
                                            curr_sd_rec['ego_pose_token'])
                global_from_car = transform_matrix(
                    current_pose_rec['translation'],
                    Quaternion(current_pose_rec['rotation']),
                    inverse=False,
                )

                # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                current_cs_rec = nusc.get(
                    'calibrated_sensor',
                    curr_sd_rec['calibrated_sensor_token'])
                car_from_current = transform_matrix(
                    current_cs_rec['translation'],
                    Quaternion(current_cs_rec['rotation']),
                    inverse=False,
                )

                tm = reduce(np.dot, [
                    ref_from_car, car_from_global, global_from_car,
                    car_from_current
                ])

                lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])

                time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']

                sweep = {
                    'lidar_path':
                    Path(lidar_path).relative_to(data_path).__str__(),
                    'sample_data_token': curr_sd_rec['token'],
                    'transform_matrix': tm,
                    'global_from_car': global_from_car,
                    'car_from_current': car_from_current,
                    'time_lag': time_lag,
                }
                sweeps.append(sweep)

        info['sweeps'] = sweeps

        assert len(info['sweeps']) == max_sweeps - 1, \
            f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, " \
            f"you should duplicate to sweep num {max_sweeps - 1}"

        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]

            # the filtering gives 0.5~1 map improvement
            num_lidar_pts = np.array(
                [anno['num_lidar_pts'] for anno in annotations])
            mask = (num_lidar_pts > 0)

            locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
            # dims = np.array([b.wlh for b in ref_boxes
            #                  ]).reshape(-1, 3)[:, [1, 0,
            #                                        2]]  # wlh == > dxdydz (lwh)
            dims = np.array([b.wlh for b in ref_boxes
                             ]).reshape(-1, 3)#[:,[0,2,1]]  # lhw == > dxdydz (lwh)
            velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
            # rots = np.array([quaternion_yaw(b.orientation)
            #                  for b in ref_boxes]).reshape(-1, 1)
            rots = np.array([b.orientation.angle
                             for b in ref_boxes]).reshape(-1, 1)           
            names = np.array([b.name for b in ref_boxes])
            tokens = np.array([b.token for b in ref_boxes])

            #여긴 lodaer에서 사용함
            # loc : xyz, dims : wlh->lwh
            gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)
            #gt_boxes = np.concatenate([locs, dims, rots + np.pi/2, velocity[:, :2]], axis=1)
            # rot 90
            #gt_boxes = gt_boxes[:, [1, 0, 2, 3, 4, 5, 6]]
            #gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi / 2
            #gt_boxes[:, 1] = -gt_boxes[:, 1]
            #gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi
            #gt_boxes[:, 6] = -gt_boxes[:, 6]
            #gt_boxes[:, 6] = np.arctan2(-gt_boxes[:,1], gt_boxes[:, 0] + gt_boxes[:,6])
            
            # gt_boxes = np.concatenate([locs, dims, rots + np.pi / 2], axis=1)
            # # rot 90
            # gt_boxes = gt_boxes[:, [1, 0, 2, 3, 4, 5, 6]]
            # gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi / 2
            # gt_boxes[:, 1] = -gt_boxes[:, 1]
            # gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi

            assert len(annotations) == len(gt_boxes) == len(velocity)

            info['gt_boxes'] = gt_boxes[mask, :]
            info['gt_boxes_velocity'] = velocity[mask, :]
            info['gt_names'] = np.array([
                map_name_from_general_to_detection[name] for name in names
            ])[mask]
            info['gt_boxes_token'] = tokens[mask]
            info['num_lidar_pts'] = num_lidar_pts[mask]
            info['num_radar_pts'] = num_lidar_pts[mask]

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    progress_bar.close()
    return train_nusc_infos, val_nusc_infos


def boxes_lidar_to_nusenes(det_info):
    boxes3d = det_info['boxes_lidar']
    scores = det_info['score']
    labels = det_info['pred_labels']

    box_list = []
    for k in range(boxes3d.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
        velocity = (*boxes3d[k, 7:9],
                    0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
        box = Box(
            boxes3d[k, :3],
            boxes3d[k, [4, 3, 5]],  # wlh
            quat,
            label=labels[k],
            score=scores[k],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(nusc, boxes, sample_token):
    s_record = nusc.get('sample', sample_token)
    sample_data_token = s_record['data']['LIDAR_TOP']

    sd_record = nusc.get('sample_data', sample_data_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_record['calibrated_sensor_token'])
    sensor_record = nusc.get('sensor', cs_record['sensor_token'])
    pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

    data_path = nusc.get_sample_data_path(sample_data_token)
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record['rotation']))
        box.translate(np.array(cs_record['translation']))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record['rotation']))
        box.translate(np.array(pose_record['translation']))
        box_list.append(box)
    return box_list


def transform_det_annos_to_nusc_annos(det_annos, nusc):
    nusc_annos = {
        'results': {},
        'meta': None,
    }

    for det in det_annos:
        annos = []
        box_list = boxes_lidar_to_nusenes(det)
        box_list = lidar_nusc_box_to_global(
            nusc=nusc, boxes=box_list, sample_token=det['metadata']['token'])

        for k, box in enumerate(box_list):
            name = det['name'][k]
            if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                if name in [
                        'car', 'construction_vehicle', 'bus', 'truck',
                        'trailer'
                ]:
                    attr = 'vehicle.moving'
                elif name in ['bicycle', 'motorcycle']:
                    attr = 'cycle.with_rider'
                else:
                    attr = None
            else:
                if name in ['pedestrian']:
                    attr = 'pedestrian.standing'
                elif name in ['bus']:
                    attr = 'vehicle.stopped'
                else:
                    attr = None
            attr = attr if attr is not None else max(
                cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
            nusc_anno = {
                'sample_token': det['metadata']['token'],
                'translation': box.center.tolist(),
                'size': box.wlh.tolist(),
                'rotation': box.orientation.elements.tolist(),
                'velocity': box.velocity[:2].tolist(),
                'detection_name': name,
                'detection_score': box.score,
                'attribute_name': attr
            }
            annos.append(nusc_anno)

        nusc_annos['results'].update({det["metadata"]["token"]: annos})

    return nusc_annos


def format_omega_results(metrics, class_names, version='default'):
    result = '----------------Nuscene %s results-----------------\n' % version
    for name in class_names:
        threshs = ', '.join(list(metrics['label_aps'][name].keys()))
        ap_list = list(metrics['label_aps'][name].values())

        err_name = ', '.join([
            x.split('_')[0]
            for x in list(metrics['label_tp_errors'][name].keys())
        ])
        error_list = list(metrics['label_tp_errors'][name].values())

        result += f'***{name} error@{err_name} | AP@{threshs}\n'
        result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
        result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
        result += f" | mean AP: {metrics['mean_dist_aps'][name]}"
        result += '\n'

    result += '--------------average performance-------------\n'
    details = {}
    for key, val in metrics['tp_errors'].items():
        result += '%s:\t %.4f\n' % (key, val)
        details[key] = val

    result += 'mAP:\t %.4f\n' % metrics['mean_ap']
    result += 'NDS:\t %.4f\n' % metrics['nd_score']

    details.update({
        'mAP': metrics['mean_ap'],
        'NDS': metrics['nd_score'],
    })

    return result, details
