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



import cv2
import os.path as osp

CAMS=['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config',
                        default='/home/qui_wzh/git_code/final_project/config/fusion/nus_maptr_fusion.py',
                        help='train config file path')
    parser.add_argument('--load-from',
                        default='/home/qui_wzh/git_code/final_project/output/fusion_maptr_v3_3/epoch_1.pth')
    parser.add_argument('--work-dir',
                        default='./vis_pred')
    parser.add_argument('--gpu-id',
                        type=int,
                        default=0)
    parser.add_argument('--score-thresh',
                        default=0.35,
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
        map_path = os.path.join(file_p, 'PRED_MAP.png')
        plt.savefig(map_path, bbox_inches='tight', format='png', dpi=1200)
        plt.close()
        #拼接生成，左边预测，中间图像，右边真值
        
        
        sample_dir=file_p
        row_1_list = []#第一行
        for cam in CAMS[:3]:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            row_1_list.append(cam_img)
        
        
        row_2_list = []#第二行
        for cam in CAMS[3:]:
            cam_img_name = cam + '.jpg'
            cam_img = cv2.imread(osp.join(sample_dir, cam_img_name))
            row_2_list.append(cam_img)
            
        row_1_img=cv2.hconcat(row_1_list)#横向拼接
        row_2_img=cv2.hconcat(row_2_list)#横向拼接
        cams_img = cv2.vconcat([row_1_img,row_2_img])#竖向拼接
        
        map_img = cv2.imread(map_path)#预测地图
        gt_map_img = cv2.imread(gt_polyline_map_path)#真值地图
        
        
        cams_h,cam_w,_ = cams_img.shape
        map_h,map_w,_ = map_img.shape
        resize_ratio = cams_h / map_h
        resized_w = map_w * resize_ratio
        resized_map_img = cv2.resize(map_img,(int(resized_w),int(cams_h)))#将预测地图动态调整适应图像
        resized_gt_map_img = cv2.resize(gt_map_img,(int(resized_w),int(cams_h)))#将真值地图动态调整适应图像
        
        # font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # fontScale
        fontScale = 1
        # Line thickness of 2 px
        thickness = 2
        # org
        org = (50, 50)      
        # Blue color in BGR
        color = (0, 0, 255)
        # Using cv2.putText() method
        resized_map_img = cv2.putText(resized_map_img, 'PRED', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        resized_gt_map_img = cv2.putText(resized_gt_map_img, 'GT', org, font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        sample_img = cv2.hconcat([resized_map_img,cams_img,resized_gt_map_img])#横向拼接
        sample_vis_path = osp.join(sample_dir, 'SAMPLE_VIS.jpg')
        cv2.imwrite(sample_vis_path, sample_img,[cv2.IMWRITE_JPEG_QUALITY, 70])
        
        
        index += 1
        prog_bar.update()
