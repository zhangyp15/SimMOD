# Dataset Preparation
Please download the nuScenes V1.0 full dataset data [HERE](https://www.nuscenes.org/download).

## Dataset structure
It is recommended to symlink the dataset root to `$SimMOD/data`.
If your folder structure is different from the following, you may need to change the corresponding paths in config files.

```
SimMOD
├── data
│   ├── nuscenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   ├── nuscenes_infos (generated below)
├── mmdetection3d
├── projects
├── tools
```

## Download and prepare the nuScenes dataset
To prepare data infos with mmdet3d, run the following command:
```bash
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --save_path ./data/nuscenes_infos --extra-tag nuscenes
```

Note that our preparation is slightly different from the official mmdet3d, you may need to regenerate the infos following our implementation.
