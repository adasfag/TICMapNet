<div align="center">
<h1>TICMapNet</h1>
<h3>A Tightly Coupled Temporal Fusion Pipeline
for End-to-End HD Map Construction</h3>

## Getting Started
- [Prepare Dataset](prepare_dataset.md)
- [Installization](Install.md)


## Models
### TICMapNet

<div align="center"><h4> Results on the nuScenes validation dataset</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_2 | R50 |GKT |DQ| 24ep |59.0  |[config](config/fusion/nus_tiofuisondq_r50e24.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpi1vwb6Pm9y1B36qn?e=aaOUMx)|


<div align="center"><h4> Results on the openlane dataset</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_1 | R50 |GKT |VA| 10ep |61.7  |[config](config/fusion/openlane_tiofusionva_r50e24.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpi1lhtUCz53HILIid?e=FXrNrA)|
| ours_2 | R50 |GKT |DQ| 10ep |60.6  |[config](config/fusion/openlane_tiofusiondq_r50e24.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpi1pLXNxq2qTj8jN2?e=piUU7h)|


<div align="center"><h4> Results of TICMapNet_t on the nuScenes validation dataset</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_2 | R50 |GKT |DQ| 24ep |57.4  |[config](config/saved_memroy/nus_maptr_fusion_save_memory.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpjAihTlZliVZ06L7I?e=AqsAfv)|


<div align="center"><h4> Results of TICMapNet_t on the openlane dataset</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_2 | R50 |GKT |VA| 10ep |59.7  |[config](config/saved_memroy/openlane_maptr_fusion_savememory.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpjAemTymfm13wfMQb?e=0Bez0s)|



<div align="center"><h4> Results of the nuScenes dataset new split[1]</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_2 | R50 |GKT |DQ| 24ep |28.3  |[config](config/new_split/nus_maptr_fusion_new_split.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpjAnuSD-wRGktdNQL?e=rzeVSc)|


[1]A. Lilja, J. Fu, E. Stenborg, and L. Hammarstrand, "Localization is all you evaluate: Data leakage in online mapping datasets and how to fix it," in CVPR 2024, pp. 22150–22159.




## Qualitative results on nuScenes val split and openlane 300 val split

<div align="center"><h4> TIOFUSION maintains stable and robust map construction quality in various driving scenes.</h4></div>

![nuScenesVisualization](assets/nuScenes_results.png)  
![openlaneVisualization](assets/OpenLane_results.png)
