
#### the dataset of [Weighted skip-connection feature fusion: A method for augmenting UAV oriented rice panicle image segmentation](https://doi.org/10.1016/j.compag.2023.107754)

## Data Introduction

<div style="text-align: justify">Two sets of mature rice panicle datasets, FSF1E and FSF2L, were collected from an unmanned farm in South China using a Phantom 4 RTK UAV. The FSF1E dataset represents the early rice in Field I, while the FSF2L dataset represents the late rice in Field II. The datasets have been published on GRAVITI, and a sample image taken by the UAV is shown in Fig. The details of the datasets are displayed in Table.
The images were cropped from left to right and top to bottom to a size of 256 × 256. Over 2,000 images were manually annotated with semantic segmentation using LabelMe.</div>


|   dataset | farmland         |   farmland size (acres) |   type | image amount | altitude(m) | resolution |
|:------------:|:--------------------:|:-------------:|:--------------:|:--------------:|:--------------:|:--------------:|
|           FSF1E | I |           0.46 |       early rice | 6 | 4.3| 5472x3648|
|           FSF1E | II |           0.31 |       late rice | 4 | 2.3|4826x3648|

```Note: FSF1E was taken vertically by UAV at an altitude of 4.3 m for farmland I inJune 2021 with a resolution of 5472 × 3648. FSF2L was taken vertically at an altitude of 2.3 m in late October 2021 for farmland II, with a resolution of 4846× 3648.```



|   dataset | images         |   size |   train | valid | modality  |
|:------------:|:--------------------:|:-------------:|:--------------:|:--------------:|:--------------:|
|           FSF1E | 1746 |           256x256 |       1418 | 328 | RGB| 
|           FSF1E | 991 |           256x256 |       774 | 217 | RGB|

```Note: When cropping large images of FSF1E and FSF2L, the crop method was from left to right and from top to bottom, excluding the margin images with less than 256 pixels and those without rice panicles```

## Cite
If you use this dataset, please cite:
```
@article{XIAO2023107754,
title = {Weighted skip-connection feature fusion: A method for augmenting UAV oriented rice panicle image segmentation},
journal = {Computers and Electronics in Agriculture},
volume = {207},
pages = {107754},
year = {2023},
issn = {0168-1699},
author = {Luolin Xiao and Zhibin Pan and Xiaoyong Du and Wei Chen and Wenpeng Qu and Youda Bai and Tian Xu}
}
```
