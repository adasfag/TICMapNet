#### maptrv2的一个复现项目。

#### 环境配置

使用conda配置虚拟环境并安装相应的关键包。

```bash
conda create -n mmlab python=3.9
conda activate mmlab
conda install cuda -c nvidia/label/cuda-12.1.1
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install openmim
mim install mmengine==0.9.0 mmcv==2.1.0 mmdet==3.2.0 mmdet3d==1.3.0
```

配置`CUDA_HOME`

```bash
export CUDA_HOME=/home/qui_wzh/miniconda3/envs/mmlab/
```

（可选）安装ninja加速编译

```bash
pip install ninja
```

linux下使用gcc编译器，windows下使用MSVC，在项目根目录执行

```bash
python setup.py develop
```

#### 训练

首先需要在配置文件中填入对应数据集的绝对路径`data_root`

单卡训练

```bash
python train.py --config {}
```

指定显卡0

```bash
python train.py --config {} --gpu-ids 0
```

分布式训练

```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py --config {} --launcher pytorch --gpus 8
```

指定显卡的编号

```bash
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 train.py --config {} --launcher pytorch --gpus 2
```

#### 测试

```bash
 python test.py --config {} --load-from {checkpoinkfile}
```




