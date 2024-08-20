<div align="center">
<h1>TICMapNet</h1>
<h3>A Tightly Coupled Temporal Fusion Pipeline
for End-to-End HD Map Construction</h3>

## Getting Started
- [Prepare Dataset](prepare_dataset.md)
- [Installization](Install.md)


## Models
### TIOFUSION

<div align="center"><h4> nuScenes dataset</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_2 | R50 |GKT |DQ| 24ep |59.0  |[config](config/fusion/nus_tiofuisondq_r50e24.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpi1vwb6Pm9y1B36qn?e=aaOUMx)|


<div align="center"><h4> openlane dataset</h4></div>

| Method | Backbone | PV2BEV |BEVDeocder|Lr Schd | mAP| Config | Download |
| :---: | :---: | :---: | :---: |  :---: |:---: | :---:|:---: |
| ours_1 | R50 |GKT |VA| 10ep |61.7  |[config](config/fusion/openlane_tiofusionva_r50e24.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpi1lhtUCz53HILIid?e=FXrNrA)|
| ours_2 | R50 |GKT |DQ| 10ep |60.6  |[config](config/fusion/openlane_tiofusiondq_r50e24.py) |[model](https://1drv.ms/u/s!AklTOiULSSxpi1pLXNxq2qTj8jN2?e=piUU7h)|



## Qualitative results on nuScenes val split and openlane 300 val split

<div align="center"><h4> TIOFUSION maintains stable and robust map construction quality in various driving scenes.</h4></div>

![nuScenesVisualization](assets/nuScenes_results.png)  
![openlaneVisualization](assets/OpenLane_results.png)
