# Copyright (c) OpenMMLab. All rights reserved.ainval_map_infos
import os
from os import path as osp

import mmengine
import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from shapely import ops, affinity
from shapely.geometry import box, MultiPolygon, Polygon

nus_map_range = (30.1, 60.1)
len_line_max=0.10
area_poly_max=0.05

map_class = ('divider', 'ped_crossing', 'boundary')#只取3个类

nus_class = dict(
    divider=('road_divider', 'lane_divider'),
    ped_crossing=('ped_crossing',),
    boundary=('road_segment', 'lane'),
    intersection=('road_segment',),
    stop_line=('stop_line',)
)


def create_nuscenes_map_infos(root_path,
                              info_prefix,
                              out_dir,
                              version='v1.0-trainval',
                              max_sweeps=10,
                              name=None):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)#NuScenes数据集句柄
    can_bus = NuScenesCanBus(dataroot=root_path)#NuScenesCanBus句柄
    nus_map = {}#NuScenesMap句柄
    for loc in ('boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown'):
        nus_map[loc] = NuScenesMap(dataroot=root_path, map_name=loc)
    # 分割训练验证集
    from nuscenes.utils import splits#从nuscenes.utils 中导入splits工具
    available_vers = ('v1.0-trainval', 'v1.0-test', 'v1.0-mini')
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train#得到train 场景id 'scene-0001'
        val_scenes = splits.val#得到val 场景id'scene-0003'
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')
    # 转换为对应的token
    t, v = 0, 0
    for scene in nusc.scene:#遍历所有nusc场景
        if scene['name'] == train_scenes[t]:
            train_scenes[t] = scene['token']
            t = t + 1
        elif scene['name'] == val_scenes[v]:
            val_scenes[v] = scene['token']
            v = v + 1
    train_scenes = set(train_scenes)#将所有train场景重新编码，从0开始到700
    val_scenes = set(val_scenes)#将所有val场景重新编码，从0开始到150
    # 输出划分长度
    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    # 单相机设置
    if info_prefix == 'one_cam':
        one_flag = True
    elif info_prefix == 'all_cam':
        one_flag = False
    else:
        raise f'get error extra-tag {info_prefix}'

    # 生成数据
    train_nusc_map_infos, val_nusc_map_infos = _fill_trainval_map_infos(
        nusc, can_bus, nus_map, train_scenes, val_scenes, test, one_flag)

    metadata = dict(
        version=version,
        map_class=map_class
    )
    if test:
        print('test sample: {}'.format(len(train_nusc_map_infos)))
        data = dict(infos=train_nusc_map_infos, metadata=metadata)
        if name is not None:
            info_path = osp.join(out_dir,
                                 '{}_{}_test.pkl'.format(info_prefix, name))
        else:
            info_path = osp.join(out_dir,
                                 '{}_nuscenes_map_test.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_map_infos), len(val_nusc_map_infos)))
        data = dict(infos=train_nusc_map_infos, metadata=metadata)
        if name is not None:
            info_path = osp.join(out_dir,
                                 '{}_{}_train.pkl'.format(info_prefix, name))
        else:
            info_path = osp.join(out_dir,
                                 '{}_nuscenes_map_train.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
        data['infos'] = val_nusc_map_infos
        if name is not None:
            info_val_path = osp.join(out_dir,
                                     '{}_{}_val.pkl'.format(info_prefix, name))
        else:
            info_val_path = osp.join(out_dir,
                                     '{}_nuscenes_map_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)


def _get_can_bus_info(nusc, can_bus, sample):#根据时间获取can_bus
    scene_name = nusc.get('scene', sample['scene_token'])['name']#场景名字
    sample_timestamp = sample['timestamp']#时间戳
    scene_id = int(scene_name[-4:])#场景id
    if scene_id in can_bus.can_blacklist:
        return [0. for _ in range(18)]
    pose_list = can_bus.get_messages(scene_name, 'pose')
    can_bus = []
    last_pose = pose_list[0]
    for pose in pose_list:
        if pose['utime'] > sample_timestamp:
            break
        last_pose = pose
    can_bus.extend(last_pose['pos'])#位置 3
    can_bus.extend(last_pose['orientation'])#取向 4
    can_bus.extend(last_pose['accel'])#加速度 3
    can_bus.extend(last_pose['rotation_rate'])#取向速度 3
    can_bus.extend(last_pose['vel'])#速度 3
    can_bus.extend([0., 0.])
    return can_bus


def _get_intersection_geom(patch_box, patch_angle, layer_name, nus_map):
    patch_x = patch_box[0]
    patch_y = patch_box[1]

    patch = nus_map.explorer.get_patch_coord(patch_box, patch_angle)

    records = getattr(nus_map, layer_name)

    polygon_list = []
    for record in records:
        # 判断是否是十字路口
        if record['is_intersection']:
            polygon = nus_map.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                  origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon,
                                                            [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)
    return [(layer_name, polygon_list)]


def _get_poly(minx, miny, maxx, maxy):
    """
    计算返回单相机时的裁剪区域
    """
    b = box(minx, miny, maxx, maxy)
    # 常量与相机参数有关
    x = (maxy - 0.75707819) * 0.7
    p = Polygon(((0, 0.75707819), (x, maxy), (-x, maxy)))
    return p.intersection(b)


def _get_map_samples(nus_map,
                     l2e_t,
                     l2e_r_mat,
                     e2g_t,
                     e2g_r_mat,
                     flag):
    l2e = np.eye(4)
    l2e[:3, :3] = l2e_r_mat
    l2e[:3, 3] = l2e_t
    e2g = np.eye(4)
    e2g[:3, :3] = e2g_r_mat
    e2g[:3, 3] = e2g_t
    l2g = e2g @ l2e

    pose_center = tuple(l2g[:2, 3])
    rotation = Quaternion(matrix=l2g)

    patch_box = (pose_center[0], pose_center[1], nus_map_range[1], nus_map_range[0])
    patch_angle = quaternion_yaw(rotation) / np.pi * 180

    vectors = []
    if flag:#单目
        # 5.625 4.685 -3.02 2.9
        # real_box = _get_poly(-15, 4.5, 15, 39)
        real_box = box(-15, 4.685, 15, 29.7)
    else:#多目
        real_box = box(-15, -29.7, 15, 29.7)
    for cls in map_class:
        if cls == 'intersection':
            geoms = _get_intersection_geom(patch_box, patch_angle, nus_class[cls][0], nus_map)
            geom = geoms[0][1]
            polygon = ops.unary_union(geom)
            polygon = polygon.intersection(real_box)
            if polygon.geom_type == 'Polygon':
                polygon = MultiPolygon([polygon])
            for poly in polygon.geoms:
                if poly.area > area_poly_max:
                    vectors.append(dict(
                        cls_name=cls,
                        type=map_class.index(cls),
                        pts=poly
                    ))
            continue
        geoms = nus_map.get_map_geom(patch_box, patch_angle, nus_class[cls])
        if cls == 'divider':
            for geom in geoms:
                for line in geom[1]:
                    line = line.intersection(real_box)
                    if line.geom_type == 'MultiLineString':
                        line = ops.linemerge(line)
                    if line.geom_type == 'MultiLineString':
                        for l in line.geoms:
                            if l.length >= len_line_max:#只保留长度大于0.10的线
                                vectors.append(dict(
                                    cls_name=cls,
                                    type=map_class.index(cls),
                                    pts=l
                                ))
                    elif line.length >= len_line_max:#只保留长度大于0.10的线
                        vectors.append(dict(
                            cls_name=cls,
                            type=map_class.index(cls),
                            pts=line
                        ))
        elif cls == 'ped_crossing':
            geom = geoms[0][1]
            # 合并
            # geom = ops.unary_union(geom)
            # if geom.geom_type != 'MultiPolygon':
            #     geom = MultiPolygon([geom])
            # multipolygon = list(geom.geoms)
            # for polygon in multipolygon:
            #     polygon = polygon.intersection(real_box)
            #     if polygon.geom_type == 'Polygon':
            #         polygon = MultiPolygon([polygon])
            #     for poly in polygon.geoms:
            #         if poly.area > 0.05:
            #             vectors.append(dict(
            #                 cls_name=cls,
            #                 type=map_class.index(cls),
            #                 pts=poly
            #             ))
            # 不合并
            for multipolygon in geom:
                multipolygon = list(multipolygon.geoms)
                for polygon in multipolygon:
                    polygon = polygon.intersection(real_box)
                    if polygon.geom_type == 'Polygon':
                        polygon = MultiPolygon([polygon])
                    for poly in polygon.geoms:
                        if poly.area > area_poly_max:
                            vectors.append(dict(
                                cls_name=cls,
                                type=map_class.index(cls),
                                pts=poly
                            ))
        elif cls == 'stop_line':
            roads = geoms[0][1]
            geom = ops.unary_union(roads)
            if geom.geom_type != 'MultiPolygon':
                geom = MultiPolygon([geom])
            multipolygon = list(geom.geoms)
            for polygon in multipolygon:
                polygon = polygon.intersection(real_box)
                if polygon.geom_type == 'Polygon':
                    polygon = MultiPolygon([polygon])
                for poly in polygon.geoms:
                    if poly.area > area_poly_max:
                        vectors.append(dict(
                            cls_name=cls,
                            type=map_class.index(cls),
                            pts=poly
                        ))
        elif cls == 'boundary':
            multipolygon = geoms[0][1]
            multipolygon.extend(geoms[1][1])
            geom = ops.unary_union(multipolygon)
            if geom.geom_type != 'MultiPolygon':
                geom = MultiPolygon([geom])
            geoms = list(geom.geoms)
            for geom in geoms:
                lines = geom.exterior.intersection(real_box)
                if lines.geom_type == 'MultiLineString':
                    lines = ops.linemerge(lines)
                if lines.geom_type == 'MultiLineString':
                    for line in lines.geoms:
                        if line.length >= len_line_max:
                            vectors.append(dict(
                                cls_name=cls,
                                type=map_class.index(cls),
                                pts=line
                            ))
                elif lines.length >= len_line_max:
                    vectors.append(dict(
                        cls_name=cls,
                        type=map_class.index(cls),
                        pts=lines
                    ))
                for inter in geom.interiors:
                    lines = inter.intersection(real_box)
                    if lines.is_empty:
                        continue
                    if lines.geom_type == 'MultiLineString':
                        # 合并公共点线
                        lines = ops.linemerge(lines)
                    if lines.geom_type == 'MultiLineString':
                        for line in lines.geoms:
                            if line.length >= len_line_max:
                                vectors.append(dict(
                                    cls_name=cls,
                                    type=map_class.index(cls),
                                    pts=line
                                ))
                    elif lines.length >= len_line_max:
                        vectors.append(dict(
                            cls_name=cls,
                            type=map_class.index(cls),
                            pts=lines
                        ))
        else:
            raise ValueError(f'WRONG vec_class: {cls}')
    return vectors


def _fill_trainval_map_infos(nusc,
                             can_bus,
                             nus_map,
                             train_scenes,
                             val_scenes,
                             test=False,
                             flag=False):
    train_nusc_map_infos = []
    val_nusc_map_infos = []
    # 按sample处理存储
    frame_idx = 0
    for sample in mmengine.track_iter_progress(nusc.sample):#在这里迭代的时候可以生成进度条，遍历所有sample
        map_location = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])['location']#地图位置
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])#雷达顶部数据得句柄
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])#车辆位姿 旋转4元数，平移3个数
        lidar_path = sd_rec['filename']#lidar 文件名
        # 读取can_bus数据
        can_bus_data = _get_can_bus_info(nusc, can_bus, sample)
        info = {
            'token': sample['token'],#token
            'prev': sample['prev'],#之前token
            'next': sample['next'],#下一个token
            'lidar_path': lidar_path,#雷达路径
            'frame_idx': frame_idx,#帧id
            'can_bus': can_bus_data,#canbus数据
            'lidar2ego_translation': cs_record['translation'],#传感器数据 lidar2ego 3
            'lidar2ego_rotation': cs_record['rotation'],#传感器数据 lidar2ego 4
            'ego2global_translation': pose_record['translation'],#ego数据 ego2global 3
            'ego2global_rotation': pose_record['rotation'],#ego数据 ego2global 4
            'map_location': map_location,#地图位置
            'cams': {},
            'timestamp': sample['timestamp']
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1
        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        # 相机参数部分
        camera_types = (
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        )
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)#相机内外参
            info['cams'].update({cam: cam_info})
            if flag:
                break

        # 生成地图标注
        if not test:
            vectors = _get_map_samples(nus_map[map_location],
                                       l2e_t, l2e_r_mat,
                                       e2g_t, e2g_r_mat,
                                       flag)
            info['gts'] = vectors

        if sample['scene_token'] in train_scenes:
            train_nusc_map_infos.append(info)
        else:
            val_nusc_map_infos.append(info)
    return train_nusc_map_infos, val_nusc_map_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(sd_rec['filename'])
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep
