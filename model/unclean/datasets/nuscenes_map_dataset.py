import os
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from pyquaternion import Quaternion
class CustomNuScenesMapDataset(BaseDataset):
    """
    自定义实现的数据集，用于读取重新生成的标注文件
    实现地面要素检测
    """

    def __init__(self,
                 ann_file: Optional[str] = '',
                 data_path: Optional[str] = '',
                 queue_length: int = 1,
                 pipeline: List[Union[dict, Callable]] = None,
                 test_mode: bool = False,
                 *args,
                 **kwargs):
        if pipeline is None:
            pipeline = []
        # queue_length固定为1或2
        self.queue_length = 1 if queue_length < 2 else 2
        self.data_path = data_path
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            *args,
            **kwargs
        )
        self.map_class = self._metainfo['map_class']

    def prepare_data(self, idx) -> Any:
        """
        关键函数，根据index和len_queue返回对应结果
        test时无视len_queue参数
        :param idx: The index of ``data_info``.
        :return: 返回值取决于``self.pipeline``.
        """
        
        
        
        
        data_info = self.get_data_info(idx)#似乎在乱跳
        if data_info is None:
            return None
        data_info = self.pipeline(data_info)
        if self.test_mode:
            return data_info
        data_queue = [data_info]
        # 训练时保证输出的时序序列都是连续的
        if self.queue_length > 1:
            if data_info['data_samples'].frame_idx > 0:
                idx = idx - 1
            data_info = self.get_data_info(idx)
            data_info = self.pipeline(data_info)
            data_queue.insert(0, data_info)
        data_queue = self.union(data_queue)
        return data_queue

    def union(self, queue):
        """
        将多个序列合为一个
        """
        data = dict()
        data['data_samples'] = [a['data_samples'] for a in queue]
        # inputs = dict()
        # if 'points' in queue[-1]['inputs']:
        #     inputs['points'] = [a['inputs']['points'] for a in queue]
        # inputs['img'] = torch.stack([a['inputs']['img'] for a in queue])
        data['inputs'] = torch.stack([a['inputs'] for a in queue])
        return data

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        # 格式转换
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(raw_data_info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = raw_data_info["lidar2ego_translation"]
        raw_data_info["lidar2ego"] = lidar2ego
        image_paths = []
        lidar2img_rts = []
        lidar2cam_rts = []
        cam_intrinsics = []
        cam2ego = []
        for cam_type, cam_info in raw_data_info['cams'].items():
            image_paths.append(os.path.join(self.data_path, cam_info['data_path']))
            # 生成 lidar to image 的转换矩阵
            lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
            lidar2cam_t = cam_info[
                              'sensor2lidar_translation'] @ lidar2cam_r.T
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = lidar2cam_r.T
            lidar2cam_rt[3, :3] = -lidar2cam_t
            lidar2cam_rt_t = lidar2cam_rt.T

            intrinsic = cam_info['cam_intrinsic']
            view_pad = np.eye(4)
            view_pad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            lidar2img_rt = view_pad @ lidar2cam_rt_t

            lidar2img_rts.append(lidar2img_rt)
            cam_intrinsics.append(view_pad)
            lidar2cam_rts.append(lidar2cam_rt_t)

            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix
            camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
            cam2ego.append(camera2ego)

        raw_data_info['img_filename'] = image_paths
        raw_data_info['lidar2img'] = lidar2img_rts
        raw_data_info['cam_intrinsics'] = cam_intrinsics
        raw_data_info['lidar2cam'] = lidar2cam_rts
        raw_data_info["camera2ego"] = cam2ego
        del raw_data_info['cams']
        # 更新can_bus数据
        e2g_r = raw_data_info['ego2global_rotation']
        e2g_t = raw_data_info['ego2global_translation']
        # can_bus = raw_data_info['can_bus']
        # can_bus[:3] = e2g_t
        # can_bus[3:7] = e2g_r
        q_e2g_r = Quaternion(e2g_r)
        # patch_angle = quaternion_yaw(q_e2g_r)
        # if patch_angle < 0:
        #     patch_angle += 2 * np.pi
        # can_bus[-2] = float(patch_angle)
        # can_bus[-1] = float(180 * patch_angle / np.pi)
        # raw_data_info['can_bus'] = can_bus

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(raw_data_info['lidar2ego_rotation']).rotation_matrix
        lidar2ego[:3, 3] = raw_data_info['lidar2ego_translation']
        ego2global = np.eye(4)
        ego2global[:3, :3] = q_e2g_r.rotation_matrix
        ego2global[:3, 3] = e2g_t
        lidar2global = ego2global @ lidar2ego
        raw_data_info['lidar2global'] = lidar2global
        # 转换雷达数据格式标签为使用官方雷达pipeline做准备
        raw_data_info['lidar_points'] = dict(
            lidar_path=os.path.join(self.data_path, raw_data_info['lidar_path'])
        )
        del raw_data_info['lidar_path']
        return raw_data_info

    def load_data_list(self) -> List[dict]:
        """
        
        
        
        
        
        
        
        
        
        读取标注文件并返回实例
        :return: 实例列表
        在这里首先进行执行
        """
        annotations = load(self.ann_file)#dict(info metadata)
        if not isinstance(annotations, dict):
            raise TypeError(f'The annotations loaded from annotation file '
                            f'should be a dict, but got {type(annotations)}!')
        if 'infos' not in annotations or 'metadata' not in annotations:
            raise ValueError('Annotation must have infos and metadata keys')
        metadata = annotations['metadata']#元数据，版本和类别
        data_infos = annotations['infos']#len:6019

        for k, v in metadata.items():
            self._metainfo.setdefault(k, v)#对注释文件的描述

        # load and parse data_infos.
        data_list = []
        for raw_data_info in data_infos:
            # parse raw data information to target format
            data_info = self.parse_data_info(raw_data_info)#对原来生的数据注释进行加工
            if isinstance(data_info, dict):
                # For image tasks, `data_info` should information if single
                # image, such as dict(img_path='xxx', width=360, ...)
                data_list.append(data_info)
            elif isinstance(data_info, list):
                # For video tasks, `data_info` could contain image
                # information of multiple frames, such as
                # [dict(video_path='xxx', timestamps=...),
                #  dict(video_path='xxx', timestamps=...)]
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError('data_info must be list of dict, but '
                                        f'got {type(item)}')
                data_list.extend(data_info)
            else:
                raise TypeError('data_info should be a dict or list of dict, '
                                f'but got {type(data_info)}')
        return data_list

    def get_map_class(self):
        return self.map_class
