#### Step-by-step installation instructions

#### Environment configuration

Create a conda virtual environment and activate it。

```bash
conda create -n tiofusion python=3.9
conda activate tiofusion
conda install cuda -c nvidia/label/cuda-12.1.1
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install openmim
mim install mmengine==0.9.0 mmcv==2.1.0 mmdet==3.2.0 mmdet3d==1.3.0
```

Configure `CUDA_HOME`

```bash
export CUDA_HOME={conda_path}/envs/mmlab
```

（optional）Install ninja to accelerate compilation 

```bash
pip install ninja
```

Using the gcc compiler for linux and MSVC for windows, execute in the project root directory 


```bash
python setup.py develop  
```  
Install GKT  

```bash
cd TIOFUSION\model\pv2bev_encoder\ops\geometric_kernel_attn
python setup.py develop
```


#### train
single gpu

```bash
python train.py --config {} --work-dir {}
```

Use a specific gpu 0

```bash
python train.py --config {} --work-dir {} --gpu-ids 0
```  
distributed training  

```bash
python -m torch.distributed.launch --nproc_per_node=8 train.py --config {} --work-dir {} --launcher pytorch --gpus 8
```  
Use specific gpu 
```bash
CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 train.py --config {} --work-dir {} --launcher pytorch --gpus 2
```

#### test

```bash
 python test.py --config {} --load-from {checkpoinkfile}
```




