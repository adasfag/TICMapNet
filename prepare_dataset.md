## NuScenes
Download nuScenes V1.0 full dataset data  and CAN bus expansion data [HERE](https://www.nuscenes.org/download). Prepare nuscenes data by running


**Download CAN bus expansion**  
```
# download 'can_bus.zip'
unzip can_bus.zip 
# move can_bus to data dir
```

**Prepare nuScenes data**

*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/generate_data.py --dataset nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes
```

Using the above code will generate `all_cam_nuscenes_map_{train,val}.pkl`, which contain local vectorized map annotations.


**Folder structure**
```
TIOFUSION
├── datasets/
├── model/
├── tools/
├── configs/
├── ckpts/
│   ├── resnet50-11ad3fa6.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── all_cam_nuscenes_map_train.pkl
|   |   ├── all_cam_nuscenes_map_val.pkl
```

## Openlane
Download the Openlane_v1 300 Sensor Dataset [here](https://github.com/OpenPerceptionX/OpenLane).

**Prepare openlane data**

*We genetate custom annotation files which are different from mmdet3d's*
```
python tools/generate_data.py --dataset openlane --root-path ./data/lane3d_300/ --dataset_base_dir ./data/images/ --out-dir ./data/
```

Using the above code will generate `openlane_map_{train,val}.pkl`, which contain local vectorized map annotations.

**Folder structure**
```
TIOFUSION
├── datasets/
├── model/
├── tools/
├── configs/
├── ckpts/
│   ├── resnet50-11ad3fa6.pth
├── data/
│   ├── images/
│   ├── lane3d_300/
│   │   ├── test/
│   │   ├── training/
│   │   ├── validation/
│   ├── openlane_map_train.pkl
│   ├── openlane_map_val.pkl
```
