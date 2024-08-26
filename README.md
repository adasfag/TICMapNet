<div align="center">
<h1>TICMapNet</h1>
<h3>A Tightly Coupled Temporal Fusion Pipeline
for End-to-End HD Map Construction</h3>


## Introduction
<div align="center"><h4>TICMapNet is a straightforward and versatile temporal fusion pipeline designed for high-precision vectorized map construction.</h4></div>

![framework](assets/pipeline.png "framework")

High-Definition (HD) map construction is essential for autonomous driving to accurately understand the surrounding environment. In this paper, we propose a **Ti**ghtly **C**oupled temporal
fusion Map Network (TICMapNet). TICMapNet breaks down the fusion process into three sub-problems: PV feature alignment, BEV feature adjustment, and Query feature fusion. By doing 
so, we effectively integrate temporal information at different stages through three plug-and-play modules, using the proposed
tightly coupled strategy. Unlike traditional methods, our approach does not rely on camera extrinsic parameters, offering a new perspective for addressing the visual fusion challenge
in the field of object detection. Experimental results demonstrate that TICMapNet significantly enhances the single-frame baseline and achieves impressive performance across multiple datasets.

## Getting Started
- [Prepare Dataset](prepare_dataset.md)
- [Installization](Install.md)


## Models
### TICMapNet

<div align="center"><h4> Results on the nuScenes validation dataset</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_2 | R50 |GKT |DQ| 24ep |59.0  |[config](config/fusion/nus_tiofuisondq_r50e24.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpi1vwb6Pm9y1B36qn?e=aaOUMx)|


<div align="center"><h4> Results on the OpenLane 300 validation dataset</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_1 | R50 |GKT |VA| 10ep |61.7  |[config](config/fusion/openlane_tiofusionva_r50e24.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpi1lhtUCz53HILIid?e=FXrNrA)|
| ours_2 | R50 |GKT |DQ| 10ep |60.6  |[config](config/fusion/openlane_tiofusiondq_r50e24.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpi1pLXNxq2qTj8jN2?e=piUU7h)|


<div align="center"><h4> Results of TICMapNet_t on the nuScenes validation dataset</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_2 | R50 |GKT |DQ| 24ep |57.4  |[config](config/saved_memroy/nus_maptr_fusion_save_memory.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpjAihTlZliVZ06L7I?e=AqsAfv)|


<div align="center"><h4> Results of TICMapNet_t on the OpenLane 300 validation dataset</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_2 | R50 |GKT |VA| 10ep |59.7  |[config](config/saved_memroy/openlane_maptr_fusion_savememory.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpjAemTymfm13wfMQb?e=0Bez0s)|



<div align="center"><h4> Results on the nuScenes dataset new validation dataset[1]</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_2 | R50 |GKT |DQ| 24ep |28.3  |[config](config/new_split/nus_maptr_fusion_new_split.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpjAnuSD-wRGktdNQL?e=rzeVSc)|


[1]A. Lilja, J. Fu, E. Stenborg, and L. Hammarstrand, "Localization is all you evaluate: Data leakage in online mapping datasets and how to fix it," in CVPR 2024, pp. 22150–22159.




## Qualitative results on nuScenes validation dataset and OpenLane 300 validation dataset
<div align="center"><h4> TICMapNet maintains stable and robust map construction quality in various driving scenes.</h4></div>

![nuScenesVisualization](assets/nuScenes_results.png)  
![openlaneVisualization](assets/OpenLane_results.png)

## Acknowledgements

TICMapNet is based on [MapTR](https://github.com/hustvl/MapTR). It is also greatly inspired by the following outstanding contributions to the open-source community:[BEVFormer](https://github.com/fundamentalvision/BEVFormer),  [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet),[BEVFusion](https://github.com/mit-han-lab/bevfusion),[GKT](https://github.com/hustvl/GKT),[mmdetection3d](https://github.com/open-mmlab/mmdetection3d).
