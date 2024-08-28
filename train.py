import argparse
import os
import warnings
from os import path as osp

from mmengine.config import Config
from mmengine.runner import Runner




def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',
                        default=None,
                        help='train config file path')
    parser.add_argument('--load-from',
                        default=None)
    parser.add_argument('--work-dir',
                        default=None,
                        )
    parser.add_argument('--launcher',
                        default=None)
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0)
    parser.add_argument('--gpus',
                        type=int,
                        default=1)
    parser.add_argument('--metric',
                        default='chamfer',
                        type=str,
                        help='in chamfer | iou | huawei | None')
    parser.add_argument('--thresh',
                        default=None,
                        type=float)
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
    if args.load_from is not None:
        config.load_from = args.load_from
        config.resume = True
    if args.launcher is not None:
        assert args.launcher == 'pytorch'
        config.launcher = args.launcher
        config.auto_scale_lr = dict(
            enable=True,
            base_batch_size=config.train_dataloader['batch_size'] * args.gpus
        )
    else:
        config.gpu_ids = [args.gpu_id]
    if args.metric is not None:
        if args.metric in ('chamfer', 'iou', 'huawei'):
            config.val_evaluator['metric'] = args.metric
        else:
            raise ValueError('metric only in chamfer | iou | huawei | None')
    if args.thresh is not None:
        if 0 <= args.thresh < 1:
            config.val_evaluator['score_thresh'] = args.thresh
        else:
            raise ValueError(f'thresh only between 0 and 1 but get {args.thresh}')

    runner = Runner.from_cfg(config)
    runner.train()
