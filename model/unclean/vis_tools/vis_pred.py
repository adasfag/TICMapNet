import argparse
import os.path
import shutil
import warnings

import mmengine
import torch.cuda
from PIL import Image
from matplotlib import pyplot as plt
from mmengine import MODELS
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',
                        default='/home/qui_wzh/git_code/final_project/config/fusion/nus_maptr_fusion.py',
                        help='train config file path')
    parser.add_argument('--load-from',
                        default='/home/qui_wzh/git_code/final_project/output/fusion_maptr_v3_2/epoch_1.pth')
    parser.add_argument('--work-dir',
                        default='./vis_pred')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0)
    parser.add_argument('--score-thresh',
                        default=0.2,
                        type=float,
                        help='samples to visualize')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    if 0 <= args.score_thresh < 1:
        score_thresh = args.score_thresh
    else:
        raise ValueError(f'thresh only between 0 and 1 but get {args.score_thresh}')
    config = Config.fromfile(args.config)
    file_path = args.work_dir
    car_img = Image.open('./fig/lidar_car.png')
    colors_plt = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
    # runner = Runner.from_cfg(config)
    # runner.load_or_resume()
    dataloader = Runner.build_dataloader(config.test_dataloader)
    model = MODELS.build(config.model)
    checkpoint = _load_checkpoint(args.load_from, map_location='cpu')
    _load_checkpoint_to_model(model, checkpoint)
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.gpu_id))
    else:
        device = torch.device('cpu')
    model.eval()
    model.to(device)
    index = 1
    prog_bar = mmengine.ProgressBar(len(dataloader.dataset))
    for data in dataloader:
        inputs = data['inputs'].to(device)
        data_samples = data['data_samples']
        with torch.no_grad():
            outputs = model(inputs, data_samples, 'predict')[0]
        data_samples = data_samples[-1][0]#时序信息
        mask = outputs['scores'] > score_thresh
        file_p = os.path.join(file_path, f'{index}')
        mmengine.mkdir_or_exist(file_p)
        for f in data_samples.filename:
            image_name = os.path.basename(f)
            image_name = image_name.split('__')[1] + '.jpg'
            image_name = os.path.join(file_p, image_name)
            shutil.copyfile(f, image_name)
        # 真值
        plt.figure(figsize=(3, 4))
        plt.xlim(-16, 16)
        plt.ylim(-30,30 )
        plt.axis('off')
        gt_label = data_samples.gt_label
        if gt_label.shape[0] > 0:
            gt_pts = data_samples.gt_pts[:, 0, :, :]
            for p, l in zip(gt_pts, gt_label):
                x = p[:, 0].numpy()
                y = p[:, 1].numpy()
                l = int(l)
                plt.plot(x, y, color=colors_plt[l], linewidth=1, alpha=0.8, zorder=-1)
                plt.scatter(x, y, color=colors_plt[l], s=1, alpha=0.8, zorder=-1)
        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        gt_polyline_map_path = os.path.join(file_p, 'GT_MAP.png')
        plt.savefig(gt_polyline_map_path, bbox_inches='tight', format='png', dpi=1200)
        plt.close()
        # 预测值
        plt.figure(figsize=(3, 4))
        plt.xlim(-16, 16)
        plt.ylim(-30, 30)
        plt.axis('off')
        label = outputs['labels'][mask]
        if label.shape[0] > 0:
            pts = outputs['pts'][mask]
            for p, l in zip(pts, label):
                x = p[:, 0].numpy()
                y = p[:, 1].numpy()
                l = int(l)
                plt.plot(x, y, color=colors_plt[l], linewidth=1, alpha=0.8, zorder=-1)
                plt.scatter(x, y, color=colors_plt[l], s=1, alpha=0.8, zorder=-1)
        plt.imshow(car_img, extent=[-1.2, 1.2, -1.5, 1.5])
        gt_polyline_map_path = os.path.join(file_p, 'PRED_MAP.png')
        plt.savefig(gt_polyline_map_path, bbox_inches='tight', format='png', dpi=1200)
        plt.close()
        index += 1
        prog_bar.update()
