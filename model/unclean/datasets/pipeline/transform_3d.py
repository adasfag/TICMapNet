"""
这里一般都是关于图形变换处理的方法
"""
from typing import Optional, Union, Tuple

import mmcv
import numpy as np
import torch
from mmcv import BaseTransform, Normalize
from mmdet3d.datasets.transforms import PhotoMetricDistortion3D


class ScaleImageMultiViewImage(BaseTransform):
    """
    缩放图像，实际上是resize图像，将图像大小变为原来的scale倍。
    进行操作后对应的图像外参信息也要进行处理。
    Args:
        scale: 缩放系数
    """

    def __init__(self, scale: float = 0.5):
        self.scale = scale

    def transform(self, results: dict) -> Optional[dict]:
        y_size = [int(img.shape[0] * self.scale) for img in results['img']]
        x_size = [int(img.shape[1] * self.scale) for img in results['img']]
     
        
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= self.scale
        scale_factor[1, 1] *= self.scale

        if y_size[0]!=450:#openlane 数据集
            y_size[0]=450
            x_size[0]=800
            scale_factor[0, 0]=800.0/1920.0
            scale_factor[1, 1]=450.0/1280.0
        
        
        results['img'] = [mmcv.imresize(img, (x_size[idx], y_size[idx])) for idx, img in
                          enumerate(results['img'])]
        lidar2img = [scale_factor @ l2i for l2i in results['lidar2img']]
        img_aug_matrix = [scale_factor for _ in results['lidar2img']]
        # cam_intrinsics = [scale_factor @ c for c in results['cam_intrinsics']]
        results['lidar2img'] = lidar2img
        results['img_aug_matrix'] = img_aug_matrix
        # results['cam_intrinsics'] = cam_intrinsics
        results['img_shape'] = [img.shape for img in results['img']]
        results['ori_shape'] = [img.shape for img in results['img']]
        results['scale_factor'] = results['scale_factor'] * self.scale
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale})'
        return repr_str


class PhotoMetricDistortion3DMultiViewImage(PhotoMetricDistortion3D):
    """
    随机对图像进行增强操作，概率0.5。
    本函数是PhotoMetricDistortion3D的多图像实现版本，具体介绍参考父类
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (sequence): range of contrast.
        saturation_range (sequence): range of saturation.
        hue_delta (int): delta of hue.
    """

    def transform(self, results: dict) -> dict:
        assert 'img' in results, '`img` is not found in results'
        imgs = results['img']
        new_imgs = []
        for img in imgs:
            img = img.astype(np.float32)
            if 'photometric_param' not in results:
                photometric_param = self._random_flags()
                # results['photometric_param'] = photometric_param
            else:
                photometric_param = results['photometric_param']

            (mode, brightness_flag, contrast_flag, saturation_flag, hue_flag,
             swap_flag, delta_value, alpha_value, saturation_value, hue_value,
             swap_value) = photometric_param

            # random brightness
            if brightness_flag:
                img += delta_value

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            if mode == 1:
                if contrast_flag:
                    img *= alpha_value

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if saturation_flag:
                img[..., 1] *= saturation_value

            # random hue
            if hue_flag:
                img[..., 0] += hue_value
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if contrast_flag:
                    img *= alpha_value

            # randomly swap channels
            if swap_flag:
                img = img[..., swap_value]
            new_imgs.append(img)
        results['img'] = new_imgs
        return results


class NormalizeMultiViewImage(Normalize):
    """
    多图像版本。
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB before
            normlizing the image. If ``to_rgb=True``, the order of mean and std
            should be RGB. If ``to_rgb=False``, the order of mean and std
            should be the same order of the image. Defaults to True.
    """

    def transform(self, results: dict) -> dict:
        results['img'] = [mmcv.imnormalize(img, self.mean, self.std,
                                           self.to_rgb) for img in results['img']]
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results


class PointToDepthMultiViewImage(BaseTransform):
    """
    使用点云数据生成相应相机的深度图。
    Args:
        grid_config: 取样范围设置
        down_sample: 下采样倍数
    """

    def __init__(self,
                 grid_config: Union[tuple, list] = (1, 35),
                 down_sample: int = 1):
        self.config = grid_config
        self.down = down_sample

    def transform(self, results: dict) -> Optional[dict]:
        points = results['points'].tensor[:, :3]
        img_len = len(results['img'])
        img_h = results['img'][0].shape[0]
        img_w = results['img'][0].shape[1]
        depth_map_list = []

        for i in range(img_len):
            lidar2img = torch.tensor(results['lidar2img'][i], dtype=torch.float32)
            points_img = points @ lidar2img[:3, :3].T + lidar2img[:3, [3]].T
            points_img = torch.cat([points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]], 1)
            depth_map = self.point2depth(points_img, img_h, img_w)
            depth_map_list.append(depth_map)
        results['depth'] = depth_map_list
        return results

    def point2depth(self, points, img_h, img_w):
        height, width = img_h // self.down, img_w // self.down
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        point_id = torch.round(points[:, :2] / self.down)
        depth = points[:, 2]
        kept1 = (point_id[:, 0] >= 0) & (point_id[:, 0] < width) & \
                (point_id[:, 1] >= 0) & (point_id[:, 1] < height) & \
                (depth >= self.config[0]) & (depth < self.config[1])
        point_id, depth = point_id[kept1], depth[kept1]

        ranks = point_id[:, 0] + point_id[:, 1] * width
        sort = torch.argsort(ranks + depth / 100, descending=True)
        point_id, depth, ranks = point_id[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(point_id.shape[0], dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        point_id, depth = point_id[kept2], depth[kept2]
        point_id = point_id.to(torch.long)
        depth_map[point_id[:, 1], point_id[:, 0]] = depth
        return depth_map

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(grid_config={self.config}, '
        repr_str += f'down_sample={self.down})'
        return repr_str


class PadMultiViewImageAndDepth(BaseTransform):
    """
    填充图像和深度图到指定大小。
    Args:
        size: 直接指定填充大小
        size_divisor: 填充后的大小至少是size_divisor倍。不可与size同时设置
        pad_val： 填充值大小
        padding_mode： 填充模式
    """

    def __init__(self,
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_val: int = 0,
                 padding_mode: str = 'constant') -> None:
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding_mode = padding_mode

    def transform(self, results: dict) -> Optional[dict]:
        size = None
        if self.size_divisor is not None:
            if size is None:
                size = (results['img'][0].shape[0], results['img'][0].shape[1])
            pad_h = int(np.ceil(
                size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(
                size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size
        padded_img = [
            mmcv.impad(img, shape=size, pad_val=self.pad_val, padding_mode=self.padding_mode)
            for img in results['img']
        ]
        results['img'] = padded_img
        if 'depth' in results:
            padded_depth = [
                mmcv.impad(depth.numpy(), shape=size, pad_val=self.pad_val, padding_mode=self.padding_mode)
                for depth in results['depth']
            ]
            results['depth'] = padded_depth
        results['img_shape'] = padded_img[0].shape[:2]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'padding_mode={self.padding_mode})'
        return repr_str
