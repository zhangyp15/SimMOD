# Prerequisites
SimMOD is developed with the these main dependencies:
- Python 3.7
- PyTorch 1.10.0
- CUDA 11.3.1 
- MMCV==1.3.14
- MMDetection==2.14.0
- MMSegmentation==0.14.1
- MMDetection3D==0.17.2

Please follow the instructions from mmdet3d [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md) to install pytorch and other dependencies. Since the latest mmdet3d (>= 1.0) is possibly not supported, an effective practice is to install the dependencies of specific versions above and build the provided mmdet3d: 
```bash
cd mmdetection3d && python setup.py develop
```

In case of any environment issues, please check our detailed dependencies [simmod.yaml](simmod.yaml).