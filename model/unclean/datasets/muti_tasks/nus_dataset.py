import os
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.structures import LiDARInstance3DBoxes


from mmengine.fileio import load
from pyquaternion import Quaternion

import os.path as osp

class CustomNuScenesDataset(NuScenesDataset):
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
                 load_interval:int=1,
                 *args,
                 **kwargs):
        if pipeline is None:
            pipeline = []
        # queue_length固定为1或2
        self.queue_length = 1 if queue_length < 2 else 2
        self.data_path = data_path
        self.load_interval = load_interval
        super().__init__(
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
            *args,
            **kwargs
        )
        self.CLASSES = self._metainfo['classes']
        


    def load_data_list(self) -> List[dict]:
        """
        
        
        
        
        
        
        
        
        
        读取标注文件并返回实例
        :return: 实例列表
        在这里首先进行执行
        """

        
        annotations = load(self.ann_file)
        
        
        metadata = annotations['metadata']#元数据，版本和类别
        data_infos = annotations['infos']#len:6019
       

        for k, v in metadata.items():
            self._metainfo.setdefault(k, v)#元数据：类别、版本、颜色
        data_infos = list(sorted(annotations["infos"], key=lambda e: e["timestamp"]))#按照时间对数据注释新型排序
        data_infos = data_infos[:: self.load_interval]
        self.CLASSES= self._metainfo['classes']
        
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
    
    
    
    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results["img_fields"] = []
        results["bbox3d_fields"] = []
        results["pts_mask_fields"] = []
        results["pts_seg_fields"] = []
        results["bbox_fields"] = []
        results["mask_fields"] = []
        results["seg_fields"] = []
        results['lidar_points']=dict(lidar_path=results['lidar_path'])
        results["box_type_3d"] = self.box_type_3d
        results["box_mode_3d"] = self.box_mode_3d
    
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        # 格式转换
        
        info=raw_data_info
        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info["location"],
        )

        # ego to global transform
        ego2global = np.eye(4).astype(np.float32)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = info["ego2global_translation"]
        data["ego2global"] = ego2global

        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            data["image_paths"] = []
            data["lidar2camera"] = []
            data["lidar2image"] = []
            data["camera2ego"] = []
            data["camera_intrinsics"] = []
            data["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                data["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                data["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
                data["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                data["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                data["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                data["camera2lidar"].append(camera2lidar)

        #加载雷达点云实例
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]#对注释进行筛选
        gt_names_3d = info["gt_names"][mask]#对类别进行筛选
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))#看是否是预定义的10个类，否则为-1
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)#类别转化为数字

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]#对速度进行筛选
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        # haotian: this is an important change: from 0.5, 0.5, 0.5 -> 0.5, 0.5, 0
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0)
        ).convert_to(self.box_mode_3d)#生成雷达3d标注

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,#雷达3d标注
            gt_labels_3d=gt_labels_3d,#类别标签
            gt_names=gt_names_3d,#类别名字
        )
        
        
        
        
        data["ann_info"] = anns_results
        return data
    
    


    def get_map_class(self):
        return self.map_class
