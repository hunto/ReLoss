
## Results
**COCO validation set**
|Method|Backbone|Input size|AP|Config|
|:--:|:--:|:--:|:--:|:--:|
|HRNet|HRNet-W32|256 x 192|74.4|[mmpose](https://github.com/open-mmlab/mmpose/blob/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py)|
|HRNet + ReLoss|HRNet-W32|256 x 192|74.8|-|

**COCO test-dev set**
|Method|Backbone|Input size|AP|Config|
|:--:|:--:|:--:|:--:|:--:|
|DARK|HRNet-W48|384 x 288|76.2|[mmpose](https://github.com/open-mmlab/mmpose/blob/master/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_384x288_dark.py)|
|DARK + ReLoss|HRNet-W48|384 x 288|76.2|-|

---

## Training
We train human pose estimation models using [MMPose](https://github.com/open-mmlab/mmpose).

Steps to reimplement our results:
* Install mmpose (see mmpose's official guide)
* Write a custom loss module and register it into mmpose
* Add ReLoss into the configs listed in above tables
* Train & test models


