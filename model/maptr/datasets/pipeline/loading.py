"""
这里一般都是关于加载的方法
"""
from typing import Optional

import mmcv
import numpy as np
from mmcv import BaseTransform


class LoadMultiViewImageFromFiles(BaseTransform):
    """
    从序列中一次读取多张图片
    输入中需要包含['img_filename']
    Args:
         to_float32: 是否将图片转化为float32。
            默认为False。
        color_type: 文件颜色格式，默认 'unchanged'。
    """

    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def transform(self, results: dict) -> Optional[dict]:
        filename = results['img_filename']
        del results['img_filename']
        imgs = [
            mmcv.imread(name, flag=self.color_type)
            for name in filename
        ]
        # 处理图片大小不一致
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename

        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]

        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str
