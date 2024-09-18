import argparse
import os
import warnings
from os import path as osp

from mmengine import EVALUATOR
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Test a detector')
    parser.add_argument('--config',
                        default='/home/qui_wzh/git_code/final_project/config/muti_scale/openlane_maptr_muti.py',
                        help='train config file path')
    parser.add_argument('--load-from',default='/home/qui_wzh/git_code/final_project/output/openlane_maptrmuti_v1/epoch_1.pth')
    parser.add_argument('--work-dir',
                        default=None)
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0)
    parser.add_argument('--metric',
                        default='chamfer',
                        type=str,
                        help='in chamfer | iou | huawei | None')
    parser.add_argument('--thresh',
                        default=None,
                        type=float)
    parser.add_argument('--metric-from',
                        default=None,
                        type=str)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    config = Config.fromfile(args.config)
    if args.work_dir is None:
        config.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    else:
        config.work_dir = args.work_dir
    config.test_evaluator['save_path'] = config.work_dir
    if args.load_from is not None:
        config.load_from = args.load_from
        config.resume = False
    if args.metric is not None:
        if args.metric in ('chamfer', 'iou', 'huawei'):
            config.test_evaluator['metric'] = args.metric
        else:
            raise ValueError('metric only in chamfer | iou | huawei | None')
    if args.thresh is not None:
        if 0 <= args.thresh < 1:
            config.test_evaluator['score_thresh'] = args.thresh
        else:
            raise ValueError(f'thresh only between 0 and 1 but get {args.thresh}')
    if args.metric_from is not None:
        metric = EVALUATOR.build(config.test_evaluator)
        mAP = metric.offline_evaluate(args.metric_from)
    else:
        config.gpu_ids = [args.gpu_id]
        runner = Runner.from_cfg(config)
        runner.test()
