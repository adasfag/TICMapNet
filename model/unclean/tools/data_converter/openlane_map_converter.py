import json
import os

import mmengine
import numpy as np
from pyquaternion import Quaternion
from shapely import ops
from shapely.geometry import box, LineString, MultiLineString

len_line_max=0.10

map_class = ('white-dash', 'white-solid',
             'yellow-dash', 'yellow-solid',
             'curbside')
id2class = {
    0: 'curbside',
    1: 'white-dash',
    2: 'white-solid',
    3: 'white-dash',
    4: 'white-solid',
    5: 'white-dash',
    6: 'white-dash',
    7: 'yellow-dash',
    8: 'yellow-solid',
    9: 'yellow-dash',
    10: 'yellow-solid',
    11: 'yellow-dash',
    12: 'yellow-dash',
    20: 'curbside',
    21: 'curbside'
}


def _fill_trainval_map_infos(root_path, param):
    map_infos = []
    folder_path = os.path.join(root_path, param)
    file_list = os.listdir(folder_path)
    patch_box = box(-15, 6.5, 15, 39.7)
    rx_1 = np.array([[0, -1, 0, 0],
                     [1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1.0]])
    rx_2 = np.array([[0, 0, 1, 0],
                     [-1, 0, 0, 0],
                     [0, -1, 0, 0],
                     [0, 0, 0, 1.0]])
    for filename in mmengine.track_iter_progress(file_list):
        frame_idx = 0
        frame_path = os.path.join(folder_path, filename)
        frame_list = os.listdir(frame_path)
        for file in frame_list:
            f = open(os.path.join(frame_path, file))
            js = json.load(f)
            f.close()
            info = {
                'token': file,
                'prev': frame_list[frame_idx - 1] if frame_idx > 0 else '',
                'lidar_path': '',
                'frame_idx': frame_idx,
                'cams': {},
                'timestamp': 0,
            }
            e2g = np.array(js['pose'])
            info['ego2global_translation'] = e2g[:3, 3].tolist()
            info['ego2global_rotation'] = Quaternion(matrix=e2g).elements.tolist()
            c2e = np.array(js['extrinsic'])
            l2e = c2e.copy()
            l2e[2, 3] = 0
            c2l = rx_1 @ np.linalg.inv(l2e) @ c2e
            # gts
            # 6.753086
            vectors = []
            for lane in js['lane_lines']:
                line = np.array(lane['xyz'])
                line = c2l[:3, :3] @ line + c2l[:3, [3]]
                line = LineString(line.transpose(1, 0)).intersection(patch_box)
                if line.geom_type == 'MultiLineString':
                    line = ops.linemerge(line)
                if line.is_empty:
                    continue
                if line.geom_type == 'LineString':
                    line = MultiLineString([line])
                for l in line.geoms:
                    if l.length >= len_line_max:
                        cls_name = id2class[lane['category']]
                        vectors.append(dict(
                            cls_name=cls_name,
                            type=map_class.index(cls_name),
                            pts=l
                        ))
            info['gts'] = vectors
            l2e = l2e @ np.linalg.inv(rx_1)
            info['lidar2ego_translation'] = l2e[:3, 3].tolist()
            info['lidar2ego_rotation'] = Quaternion(matrix=l2e).elements.tolist()
            camera_types = (
                'CAM_FRONT',
            )
            for cam in camera_types:
                c = {
                    'data_path': js['file_path'],
                }
                c2e = c2e @ rx_2
                c['sensor2ego_translation'] = c2e[:3, 3].tolist()
                c['sensor2ego_rotation'] = Quaternion(matrix=c2e).elements.tolist()
                c['ego2global_translation'] = info['ego2global_translation']
                c['ego2global_rotation'] = info['ego2global_rotation']
                c2l = np.linalg.inv(l2e) @ c2e
                c['sensor2lidar_rotation'] = c2l[:3, :3]  # points @ R.T + T
                c['sensor2lidar_translation'] = c2l[:3, 3]
                c['cam_intrinsic'] = np.array(js['intrinsic'])
                info['cams'].update({cam: c})
            frame_idx += 1
            map_infos.append(info)

    return map_infos


def create_openlane_map_infos(root_path,
                              extra_tag,
                              out_dir,
                              name=None):
    train_map_infos = _fill_trainval_map_infos(root_path, 'training/')
    val_map_infos = _fill_trainval_map_infos(root_path, 'validation/')
    metadata = dict(
        map_class=map_class
    )
    print('train sample: {}, val sample: {}'.format(
        len(train_map_infos), len(val_map_infos)))
    data = dict(infos=train_map_infos, metadata=metadata)
    if name is not None:
        info_path = os.path.join(out_dir,
                                 '{}_train.pkl'.format(name))
    else:
        info_path = os.path.join(out_dir,
                                 'openlane_map_train.pkl')
    mmengine.dump(data, info_path)
    data['infos'] = val_map_infos
    if name is not None:
        info_path = os.path.join(out_dir,
                                 '{}_val.pkl'.format(name))
    else:
        info_path = os.path.join(out_dir,
                                 'openlane_map_val.pkl')
    mmengine.dump(data, info_path)
