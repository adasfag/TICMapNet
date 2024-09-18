import copy
import os.path
from typing import Optional, Any, Sequence

import mmengine
import numpy as np
from mmdet.evaluation import average_precision
from mmengine import print_log
from mmengine.evaluator import BaseMetric
from terminaltables import AsciiTable

from model.evaluator.pool_utils import up_sample_pool, get_tp_fp_pool, get_mile_pool


class XjBevMetric(BaseMetric):
    def __init__(self,
                 metric: str = 'chamfer',
                 classes: Sequence = None,
                 score_thresh: float = 0.4,
                 collect_device: str = 'cpu',
                 num_sample: int = 100,
                 save: bool = False,
                 save_path: str = None,
                 prefix: Optional[str] = None):
        super().__init__(collect_device=collect_device, prefix=prefix)
        assert metric in ('chamfer', 'iou', 'huawei')
        self.metric = metric
        self.classes = classes
        self.score = score_thresh
        self.num_sample = num_sample
        self.save = save
        if self.save:
            self.save_path = save_path
        self.line_class = ('divider', 'boundary', 'white-dash', 'white-solid',
                           'yellow-dash', 'yellow-solid', 'curbside')
        if metric == 'iou':
            self.thresholds = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
            self.line_width = 1
        elif metric == 'chamfer':
            self.thresholds = (-1.5, -1.0, -0.5)
            self.line_width = 2
        else:
            self.thresholds = (0.5, 0.6, 0.7, 0.8, 0.9)
            self.line_width = 1

    def process(self, data_batch: Any, data_samples: Sequence[dict]) -> None:#先处理原始数据data_batch和预测结果data_samples得到合并结果
        data_samples[0]['gt_label'] = data_batch['data_samples'][0].gt_label.numpy()
        if data_batch['data_samples'][0].gt_label.shape[0] > 0:
            data_samples[0]['gt_pts'] = data_batch['data_samples'][0].gt_pts[:, 0, :, :].numpy()
        else:
            data_samples[0]['gt_pts'] = data_batch['data_samples'][0].gt_pts.numpy()
        mask = data_samples[0]['scores'] > self.score
        data_samples[0]['scores'] = data_samples[0]['scores'][mask].numpy()
        data_samples[0]['labels'] = data_samples[0]['labels'][mask].numpy()
        data_samples[0]['pts'] = data_samples[0]['pts'][mask].numpy()
        self.results.append(copy.deepcopy(data_samples[0]))

    def compute_metrics(self, results: list) -> dict:
        if self.save:
            # 目前保存文件名相对固定，如不想覆盖需要自己重命名
            mmengine.dump(results, os.path.join(self.save_path, 'result_{:.0f}.pkl'.format(self.score * 100)))
        timer = mmengine.Timer()
        gt_pts = []
        pre_pts = []
        scores = []
        for i in range(len(self.classes)):
            gt_pt, pt, score = up_sample_pool(results, i, self.num_sample)
            gt_pts.append(gt_pt)
            pre_pts.append(pt)
            scores.append(score)
        print('format data in {:.2f}s !!'.format(float(timer.since_last_check())))
        # 计算指标
        print('-*' * 10 + f'use metric:{self.metric}' + '-*' * 10)
        mAP = 0
        for thr in self.thresholds:
            print('-*' * 10 + f'threshhold:{thr}' + '-*' * 10)
            eval_result = []
            mean_ap = 0
            num_ap = 0
            class_name = []
            for i in range(len(self.classes)):
                if self.metric == 'huawei':
                    if self.classes[i] in self.line_class:
                        continue
                timer.since_last_check()
                class_name.append(self.classes[i])
                gt_pt = gt_pts[i]
                pre_pt = pre_pts[i]
                score = np.concatenate(scores[i])
                tps, fps, num_pres, num_gts = get_tp_fp_pool(gt_pt, pre_pt,
                                                             line_width=self.line_width,
                                                             metric=self.metric,
                                                             thr=thr)
                num_pres = sum(num_pres)
                num_gts = sum(num_gts)
                sort_ind = np.argsort(-score)
                tps = np.hstack(tps)[sort_ind].cumsum(axis=0)
                fps = np.hstack(fps)[sort_ind].cumsum(axis=0)
                eps = np.finfo(np.float32).eps
                recalls = tps / np.maximum(num_gts, eps)
                precisions = tps / np.maximum((tps + fps), eps)
                ap = average_precision(recalls, precisions)
                eval_result.append({
                    'num_gts': num_gts,
                    'num_dets': num_pres,
                    'recall': recalls,
                    'precision': precisions,
                    'ap': ap
                })
                mean_ap += ap if num_gts > 0 else 0
                num_ap += 1 if num_gts > 0 else 0
                print('cls {} done in {:.2f}s !!'.format(self.classes[i], float(timer.since_last_check())))
            mean_ap = (mean_ap / num_ap) if num_ap > 0 else 0.0
            print_map_summary(mean_ap, eval_result, class_name=class_name)
            mAP += mean_ap
        # 里程指标
        if self.metric == 'huawei':
            print('-*' * 10 + f'mile' + '-*' * 10)
            gts = []
            pres = []
            res = []
            precis = []
            clss = []
            aver = []
            for cls in self.line_class:
                if cls in self.classes:
                    index = self.classes.index(cls)
                    timer.since_last_check()
                    gt_pt = gt_pts[index]
                    pre_pt = pre_pts[index]
                    tp, distance_list, pre_length, gt_length = get_mile_pool(gt_pt, pre_pt)
                    tp = sum(tp)
                    pre_length = sum(pre_length)
                    gt_length = sum(gt_length)
                    recall = tp / gt_length if gt_length > 0 else 0
                    precision = tp / pre_length if pre_length > 0 else 0
                    distance_list = np.hstack(distance_list).mean().item()
                    gts.append(gt_length)
                    pres.append(pre_length)
                    res.append(recall)
                    precis.append(precision)
                    clss.append(cls)
                    aver.append(distance_list)
                    print('cls {} done in {:.2f}s !!'.format(cls, float(timer.since_last_check())))
            table_data = [['class', 'gt_mile', 'det_mile', 'recall', 'precision', 'average_distance']]
            for i in range(len(clss)):
                row_data = [
                    clss[i], f'{gts[i]:.3f}', f'{pres[i]:.3f}',
                    f'{res[i]:.3f}', f'{precis[i]:.3f}', f'{aver[i]:.3f}'
                ]
                table_data.append(row_data)
            table = AsciiTable(table_data)
            table.inner_footing_row_border = True
            print_log(table.table)
        return dict(mAP=mAP / len(self.thresholds))

    def offline_evaluate(self, path: str):
        self.save = False
        results = mmengine.load(path)
        return self.compute_metrics(results)



def print_map_summary(mean_ap,
                      results,
                      class_name=None):
    num_scales = 1
    num_classes = len(results)
    recalls = np.zeros((num_scales, num_classes), dtype=np.float32)
    pre = np.zeros((num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((num_scales, num_classes), dtype=int)
    for i, cls_result in enumerate(results):
        if cls_result['recall'].size > 0:
            recalls[0, i] = cls_result['recall'][-1]
        if cls_result['precision'].size > 0:
            pre[0, i] = cls_result['precision'][-1]
        aps[0, i] = cls_result['ap']
        num_gts[0, i] = cls_result['num_gts']

    if not isinstance(mean_ap, list):
        mean_ap = [mean_ap]

    header = ['class', 'gts', 'dets', 'recall', 'precision', 'ap']
    for i in range(num_scales):
        table_data = [header]
        for j in range(num_classes):
            row_data = [
                class_name[j], num_gts[i, j], results[j]['num_dets'],
                f'{recalls[i, j]:.3f}', f'{pre[i, j]:.3f}', f'{aps[i, j]:.3f}'
            ]
            table_data.append(row_data)
        table_data.append(['mAP', '', '', '', '', f'{mean_ap[i]:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log(table.table)
