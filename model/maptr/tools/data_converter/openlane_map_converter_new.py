import json
import os

import mmengine
import numpy as np
from pyquaternion import Quaternion
from shapely import ops
from shapely.geometry import box, LineString, MultiLineString
import copy

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


def _fill_trainval_map_infos(root_path, param,dataset_base_dir = '/home/share/ssd/dataset/OpenLane/images/'):
    map_infos = []#问题的关键是这个不是按照时序组织的
    folder_path = os.path.join(root_path, param)
    file_list = os.listdir(folder_path)
    for filename in mmengine.track_iter_progress(file_list):#遍历文件夹下面所有片段
        frame_idx = 0
        frame_path = os.path.join(folder_path, filename)
        frame_list = os.listdir(frame_path)
        frame_list.sort()#这个排序很关键
        dataset_base_dir = dataset_base_dir
        
        
        for file in frame_list:#遍历文件夹下各个片段所有场景
            label_file=os.path.join(frame_path, file)
            with open(label_file, 'r') as anno_file:
                file_lines = [line for line in anno_file]
                info_dict = json.loads(file_lines[0])

            image_path = os.path.join(dataset_base_dir, info_dict['file_path'])
            assert os.path.exists(image_path), '{:s} not exist'.format(image_path)
            _label_image_path = image_path

            cam_extrinsics = np.array(info_dict['extrinsic'])  # waymo camera --> waymo vehicle
            # Re-calculate extrinsic matrix based on ground coordinate
            R_vg = np.array([[0, 1, 0],
                             [-1, 0, 0],
                             [0, 0, 1]], dtype=float)  # laneNet camera --> waymo camera
            R_gc = np.array([[1, 0, 0],
                             [0, 0, 1],
                             [0, -1, 0]], dtype=float)  # normal camera --> laneNet camera
            cam_extrinsics[:3, :3] = np.matmul(np.matmul(
                np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3, :3]),
                R_vg), R_gc)  # normal camera --> laneNet vehicle
            cam_extrinsics[0:2, 3] = 0.0  # normal camera  --> laneNet ego

            if 'intrinsic' in info_dict:
                cam_intrinsics = info_dict['intrinsic']
                cam_intrinsics = np.array(cam_intrinsics)  # camera --> image pixel
            else:
                cam_intrinsics = K

            gt_lanes_packed = info_dict['lane_lines']
            gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []
            for i, gt_lane_packed in enumerate(gt_lanes_packed):  # 一张图片里的每一条车道线
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = np.array(gt_lane_packed['xyz'])
                lane_visibility = np.array(gt_lane_packed['visibility'])

                # Coordinate convertion for openlane_300 data
                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))  # ndarray(4,561)
                cam_representation = np.linalg.inv(  # inv后 waymo camera --> normal camera
                    np.array([[0, 0, 1, 0],  # inv前 normal camera --> waymo camera
                              [-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, 0, 1]], dtype=float))  # transformation from apollo camera to openlane camera
                lane = np.matmul(cam_extrinsics, np.matmul(cam_representation,
                                                           lane))  # waymo camera coordinate --> laneNet ego coordinate

                lane = lane[0:3, :].T  # ndarray(516,3)
                gt_lane_pts.append(lane)  # list
                gt_lane_visibility.append(lane_visibility)  # list

                if 'category' in gt_lane_packed:
                    lane_cate = gt_lane_packed['category']

                    if lane_cate == 21:  # merge left and right road edge into road edge
                        lane_cate = 20
                    if lane_cate == 20:  # 所有的20类转化为15类
                        lane_cate = 15
                    # lane_cate = 0   #changed_openlane_test 将所有类别车道线先置为一个类
                    new_lane_cate_dict = {
                        0: 4,
                        1: 0,
                        2: 1,
                        3: 0,
                        4: 1,
                        5: 0,
                        6: 0,
                        7: 2,
                        8: 3,
                        9: 2,
                        10: 3,
                        11: 2,
                        12: 2,
                        13: 4,
                        14: 4,
                        15: 4,

                    }

                    gt_laneline_category.append(new_lane_cate_dict[lane_cate])



                else:
                    gt_laneline_category.append(4)  # 否则的话归到第5类
            _gt_laneline_category_org = copy.deepcopy(np.array(gt_laneline_category))

            cam_K = cam_intrinsics  # camera --> image pixel

            cam_E = cam_extrinsics  # normal camera  --> laneNet ego
            P_g2im = projection_g2im_extrinsic(cam_E, cam_K)  # laneNet ego  --> normal image pixel  3*3 3*4 --> 3*4
            H_g2im = homograpthy_g2im_extrinsic(cam_E,
                                                cam_K)  # laneNet ego  --> normal image pixel  3*3 3*3 --> 3*3

            H_im2g = np.linalg.inv(H_g2im)  # normal image pixel   --> laneNet ego  3*3

            P_g2gflat = np.matmul(H_im2g, P_g2im)  # laneNet ego (with Height 3*4) -->laneNet ego  (without Height 3*4)

            gt_lanes = gt_lane_pts
            gt_visibility = gt_lane_visibility
            gt_category = gt_laneline_category

            # prune gt lanes by visibility labels
            gt_lanes = [prune_3d_lane_by_visibility(gt_lane, gt_visibility[k]) for k, gt_lane in
                        enumerate(gt_lanes)]  # ndarray(516,3)

            # prune out-of-range points are necessary before transformation
            gt_lanes = [prune_3d_lane_by_range(gt_lane, -15, 15) for gt_lane in gt_lanes]

            gt_category = [gt_cate for (gt_cate, lane) in zip(gt_category, gt_lanes) if
                        lane.shape[0] > 20]  # 先排除筛选后的category
            gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 20]  # 再排除只有一个点的车道线，筛选后的车道线

            gt_lanes = [gt_lane.T for gt_lane in gt_lanes]  # 回到ndarray(3,134)

            _label_laneline_org = copy.deepcopy(gt_lanes)
            
            
            
            
            
            
            
            
            
            vectors = []
            for index,gt_lane in enumerate(gt_lanes):
                cls_name=gt_category[index]
                
                gt_lane=gt_lane
                line = LineString(gt_lane.transpose(1, 0))
                
                
                vectors.append(dict(
                                cls_name=map_class[cls_name],
                                type=cls_name,
                                pts=line
                            ))
                
                # if line.geom_type == 'MultiLineString':
                #     line = ops.linemerge(line)
                # if line.is_empty:
                #     continue
                # if line.geom_type == 'LineString':
                #     line = MultiLineString([line])
                
                # for l in line.geoms:
                #     pt=l
                #     vectors.append(dict(
                #                 cls_name=map_class[cls_name],
                #                 type=cls_name,
                #                 pts=pt
                #             ))
                
            info = {
                'token': filename,
                'prev': frame_list[frame_idx - 1] if frame_idx > 0 else '',
                'lidar_path': '',
                'frame_idx': frame_idx,
                'cams': {},
                'timestamp': 0,
            }    
            info['ego2global_translation'] = ''
            info['ego2global_rotation'] = ''
            
            info['gts'] = vectors
            info['ego2global_translation'] = ''
            info['ego2global_rotation'] = ''
            camera_types = (
                'CAM_FRONT',
            )
            for cam in camera_types:
                c = {
                    'data_path': _label_image_path,#图像路径
                }
                c['sensor2ego_translation'] = ''
                c['sensor2ego_rotation'] = ''
                c['ego2global_translation'] =''
                c['ego2global_rotation'] = ''
                c['sensor2lidar_rotation'] = ''
                c['sensor2lidar_translation'] = ''
                c['cam_extrinsics']=cam_extrinsics
                c['cam_intrinsic'] = cam_intrinsics
                info['cams'].update({cam: c})#更新相机参数
            frame_idx += 1
            map_infos.append(info)

    return map_infos


def create_openlane_map_infos(root_path,
                              extra_tag,
                              out_dir,
                              name=None,
                              dataset_base_dir=None):
    train_map_infos = _fill_trainval_map_infos(root_path, 'training/',dataset_base_dir)
    val_map_infos = _fill_trainval_map_infos(root_path, 'validation/',dataset_base_dir)
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
                                 'openlane_map_train_new_300.pkl')
    mmengine.dump(data, info_path)
    data['infos'] = val_map_infos
    if name is not None:
        info_path = os.path.join(out_dir,
                                 '{}_val.pkl'.format(name))
    else:
        info_path = os.path.join(out_dir,
                                 'openlane_map_val_new_300.pkl')
    mmengine.dump(data, info_path)
    
    



#--cjj-o
#--cjj-s add utils
def homograpthy_g2im_extrinsic(E, K): #利用内外参转换
    """E: extrinsic matrix, 4*4"""
    E_inv = np.linalg.inv(E)[0:3, :]# laneNet ego  --> normal camera
    H_g2c = E_inv[:, [0,1,3]] #只取1，2，3列
    H_g2im = np.matmul(K, H_g2c)# laneNet ego  --> normal image pixel  3*3 3*3 --> 3*3
    return H_g2im


def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)[0:3, :] # laneNet ego  --> normal camera  4*4 --> 3*4
    P_g2im = np.matmul(K, E_inv)# laneNet ego  --> normal image pixel  3*3 3*4 --> 3*4
    return P_g2im


def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d

def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 40), ...]#在这里会进行筛选

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d
#--cjj-e


