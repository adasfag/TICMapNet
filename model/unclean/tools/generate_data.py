# Copyright (c) OpenMMLab. All rights reserved.
# 对原始代码进行修改，适配BEV地图任务，生成原始3D标注的同时为后续任务生成map标注
import argparse

from tools.data_converter import nuscenes_map_converter, openlane_map_converter_new


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10,
                       name=None):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_map_converter.create_nuscenes_map_infos(
        root_path, info_prefix, out_dir, version=version, max_sweeps=max_sweeps, name=name)







def openlane_data_prep(root_path,
                       extra_tag,
                       out_dir,
                       name=None,
                       dataset_base_dir = '/home/share/ssd/dataset/OpenLane/images/'):
    openlane_map_converter_new.create_openlane_map_infos(root_path, extra_tag, out_dir, name,dataset_base_dir )


parser = argparse.ArgumentParser(description='Data converter arg parser')
# 处理的数据集
parser.add_argument('--dataset', default='nuscenes', help='name of the dataset')
# 数据集路径
parser.add_argument(
    '--root-path',
    type=str,
    default='/home/qui_wzh/dataset/nuscese/nuscenes_base/',
    help='specify the root path of dataset', 
    )

parser.add_argument(
    '--dataset_base_dir',
    type=str,
    default='/home/share/ssd/dataset/OpenLane/images/',
    help='specify the root path of dataset', 
    )#openlane数据集有用
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False)
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')  # 不使用
# 生成的文件输出路径
parser.add_argument(
    '--out-dir',
    type=str,
    default='/home/qui_wzh/git_code/MAP_PER/data/',
    required=False,
    help='name of info pkl')
# 文件名设置
parser.add_argument('--name', type=str, default=None, help='file_name')
# 单双相机选择
parser.add_argument('--extra-tag', type=str, default='all_cam',
                    help='one_cam or all_cam')
args = parser.parse_args()

if __name__ == '__main__':
    if args.out_dir is None:
        args.out_dir = args.root_path
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'#v1.0-trainval
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps,
            name=args.name
        )
    if args.dataset == 'openlane':
        openlane_data_prep(
            root_path=args.root_path,
            extra_tag=args.extra_tag,
            out_dir=args.out_dir,
            name=args.name,
            dataset_base_dir = args.dataset_base_dir,
        )
