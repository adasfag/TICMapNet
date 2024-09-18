"""
处理数据并转换输出作为模型输入
"""
from typing import Optional, Dict

import cv2
import numpy as np
from mmcv import BaseTransform
from mmdet3d.datasets.transforms.formating import to_tensor
from mmengine.structures import BaseDataElement
from shapely import affinity


class PackDataToInputs(BaseTransform):
    """
    自定义数据处理，将数据转换成tensor并保留必须数据输出。
    """
    # 需要保留的数据
    MATE_KEYS = (
        'filename', 'ori_shape', 'img_shape', 'lidar2img', 'scale_factor',
        'frame_idx', 'prev', 'next', 'token',
        'cam_intrinsics', 'lidar2global', 'img_aug_matrix',
        'depth', 'gt_label', 'gt_pts', 'gt_bev', 'gt_pv'
    )

    def transform(self, results: dict) -> Optional[dict]:
        # 处理图像和点云数据
        inputs = {}
        # if 'points' in results:
        #     if isinstance(results['points'], BasePoints):
        #         inputs['points'] = results['points'].tensor
        #         del results['points']
        if 'img' in results:
            if isinstance(results['img'], list):
                # process multiple imgs in single frame
                imgs = np.stack(results['img'], axis=0)
                if imgs.flags.c_contiguous:
                    imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
                else:
                    imgs = to_tensor(
                        np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
                inputs['img'] = imgs
            else:
                img = results['img']
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                if img.flags.c_contiguous:
                    img = to_tensor(img).permute(2, 0, 1).contiguous()
                else:
                    img = to_tensor(
                        np.ascontiguousarray(img.transpose(2, 0, 1)))
                inputs['img'] = img
            del results['img']
        # 处理标准键
        data_sample = BaseDataElement()
        data_metas = {}
        # 需要转换为tensor的数据
        for key in [
            'ori_shape', 'lidar2img', 'cam_intrinsics',
            'lidar2global', 'img_aug_matrix', 'depth',
            'gt_label', 'gt_pts', 'gt_bev', 'gt_pv'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = to_tensor(np.array(results[key]))
            else:
                results[key] = to_tensor(results[key])
        for key in self.MATE_KEYS:
            if key in results:
                data_metas[key] = results[key]
        data_sample.set_metainfo(data_metas)
        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs['img']
        return packed_results


class MakeLineGts(BaseTransform):
    def __init__(self,
                 num_vec_len=20,
                 bev=(100, 100),
                 bev_loss=False,
                 pv_loss=False,
                 bev_down_sample=1,
                 feat_down_sample=32,
                 pts_pattern='v1',
                 z_min=3):
        assert pts_pattern in ('v1', 'v2')
        self.pts_pattern = pts_pattern
        self.z_min = z_min
        self.bev_down_sample = bev_down_sample
        self.feat_down_sample = feat_down_sample
        self.num_vec_len = num_vec_len
        self.bev = bev
        self.pv_loss = pv_loss
        self.bev_loss = bev_loss

    def transform(self, results: Dict) -> Dict:
        cam = len(results['img'])
        gts = results['gts']
        del results['gts']
        if self.bev_loss:
            gt_semantic_mask = np.zeros((1, self.bev[0], self.bev[1]), dtype=np.uint8)
        if self.pv_loss:
            gt_pv_semantic_mask = np.zeros((cam, 1,
                                            results['img_shape'][0] // self.feat_down_sample,
                                            results['img_shape'][1] // self.feat_down_sample), dtype=np.uint8)
        gt_label = []
        gt_pts = []
        if self.pts_pattern == 'v1':
            final_shift_num = self.num_vec_len - 1
        else:
            final_shift_num = (self.num_vec_len - 1) * 2
        for gt in gts:
            gt_label.append(gt['type'])
            is_poly = False
            if gt['pts'].geom_type == 'Polygon':
                is_poly = True
                gt['pts'] = gt['pts'].exterior
            if self.bev_loss:
                gemo = gt['pts']
                gemo = affinity.scale(gemo, self.bev_down_sample, self.bev_down_sample, origin=(0, 0))
                if cam == 1:
                    gemo = affinity.affine_transform(gemo, (1, 0, 0, 1, self.bev[1] // 2, 0))
                else:
                    gemo = affinity.affine_transform(gemo, (1, 0, 0, 1, self.bev[1] // 2, self.bev[0] // 2))#仿射变换？
                coords = np.array(gemo.coords, dtype=np.int32)[:, :2]
                cv2.polylines(gt_semantic_mask[0],
                              np.int32([coords]),
                              False,
                              color=(1,),
                              thickness=3)
            if self.pv_loss:
                gemo = gt['pts']
                coords = np.array(gemo.coords)
                scale_factor = np.eye(4)
                scale_factor[0, 0] /= self.feat_down_sample
                scale_factor[1, 1] /= self.feat_down_sample
                for c in range(cam):
                    if not gemo.has_z:
                        z = np.full((coords.shape[0], 1), -results['camera2ego'][c][2, 3])
                        coord = np.concatenate([coords, z], axis=1).transpose(1, 0)
                    else:
                        coord = coords.copy().transpose(1, 0)
                    l2img = results['lidar2img'][c]
                    l2f = scale_factor @ l2img
                    coord = l2f[:3, :3] @ coord + l2f[:3, [3]]
                    # 此处大于的数为一个超参，应根据不同数据集的相机合理设置
                    valid_idx = coord[2, :] > self.z_min
                    coord = coord[:, valid_idx]
                    coord = (coord[:2, :] / coord[2, :]).transpose(1, 0)
                    cv2.polylines(gt_pv_semantic_mask[c][0],
                                  np.int32([coord]),
                                  False,
                                  color=(1,),
                                  thickness=1)
            gemo = gt['pts']
            distances = np.linspace(0, gemo.length, self.num_vec_len)
            points = np.array([gemo.interpolate(d).coords for d in distances])[..., :2].reshape(-1, 2)
            shift_num = []
            if is_poly:
                points = points[:-1, :]
                for i in range(self.num_vec_len - 1):
                    shift_point = np.roll(points, i, axis=0)
                    shift_point = np.concatenate((shift_point, shift_point[[0], :]), axis=0)
                    shift_num.append(shift_point)
                if self.pts_pattern == 'v2':
                    points = np.flip(points, axis=0)
                    for i in range(self.num_vec_len - 1):
                        shift_point = np.roll(points, i, axis=0)
                        shift_point = np.concatenate((shift_point, shift_point[[0], :]), axis=0)
                        shift_num.append(shift_point)
            else:
                shift_num.append(points)
                shift_num.append(np.flip(points, axis=0))
            shift_num = np.stack(shift_num, axis=0)
            s_num = shift_num.shape[0]
            if s_num < final_shift_num:
                shift_pad = np.full((final_shift_num - s_num, self.num_vec_len, 2), -10000)#用-10000填充
                shift_num = np.concatenate([shift_num, shift_pad], axis=0)
            gt_pts.append(shift_num)
        gt_label = np.array(gt_label, dtype=np.int64)
        gt_pts = np.array(gt_pts)
        results['gt_label'] = gt_label
        results['gt_pts'] = gt_pts
        if self.bev_loss:
            results['gt_bev'] = gt_semantic_mask
        if self.pv_loss:
            results['gt_pv'] = gt_pv_semantic_mask
        return results
