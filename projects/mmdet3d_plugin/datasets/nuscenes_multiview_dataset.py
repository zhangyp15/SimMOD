# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
import copy

from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import (
    Box3DMode,
    Coord3DMode,
    LiDARInstance3DBoxes,
    CameraInstance3DBoxes,
)
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from mmdet.datasets.api_wrappers import COCO, COCOeval

import pdb


@DATASETS.register_module()
class NuScenesMultiviewDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool): Whether to use `use_valid_flag` key in the info
            file as mask to filter gt_boxes and gt_names. Defaults to False.
    """
    NameMapping = {
        "movable_object.barrier": "barrier",
        "vehicle.bicycle": "bicycle",
        "vehicle.bus.bendy": "bus",
        "vehicle.bus.rigid": "bus",
        "vehicle.car": "car",
        "vehicle.construction": "construction_vehicle",
        "vehicle.motorcycle": "motorcycle",
        "human.pedestrian.adult": "pedestrian",
        "human.pedestrian.child": "pedestrian",
        "human.pedestrian.construction_worker": "pedestrian",
        "human.pedestrian.police_officer": "pedestrian",
        "movable_object.trafficcone": "traffic_cone",
        "vehicle.trailer": "trailer",
        "vehicle.truck": "truck",
    }
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    AttrMapping = {
        "cycle.with_rider": 0,
        "cycle.without_rider": 1,
        "pedestrian.moving": 2,
        "pedestrian.standing": 3,
        "pedestrian.sitting_lying_down": 4,
        "vehicle.moving": 5,
        "vehicle.parked": 6,
        "vehicle.stopped": 7,
    }
    AttrMapping_rev = [
        "cycle.with_rider",
        "cycle.without_rider",
        "pedestrian.moving",
        "pedestrian.standing",
        "pedestrian.sitting_lying_down",
        "vehicle.moving",
        "vehicle.parked",
        "vehicle.stopped",
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )

    def __init__(
        self,
        ann_file,
        mono_anno_file=None,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        box_type_3d="LiDAR",
        filter_empty_gt=True,
        test_mode=False,
        eval_version="detection_cvpr_2019",
        use_valid_flag=False,
    ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
        )

        self.bbox_code_size = 9
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory

        self.eval_detection_configs = config_factory(self.eval_version)

        if not self.test_mode:
            self.mono_anno_file = mono_anno_file
            self.coco = COCO(self.mono_anno_file)
            self.coco_cat_ids = self.coco.get_cat_ids(
                cat_names=self.CLASSES,
            )  # The order of returned `cat_ids` will not change with the order of the CLASSES

        if self.modality is None:
            self.modality = dict(
                use_camera=True,
                use_lidar=False,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]

        return data_infos

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None

        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and (
            example is None or ~(example["gt_labels_3d"]._data != -1).any() or example["gt_labels_3d"]._data.shape[0] == 0
        ):
            return None
        
        return example

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
        )

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            cam_anno_infos = []

            sensor2lidar_rotation = []
            sensor2lidar_translation = []

            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]

                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T

                # lidar to camera coordinates
                lidar2cam_rts.append(lidar2cam_rt.T)
                # camera coordinates to image pixels
                cam_intrinsics.append(viewpad)
                # lidar to image pixels
                lidar2img_rts.append(lidar2img_rt)

                sensor2lidar_rotation.append(cam_info["sensor2lidar_rotation"])
                sensor2lidar_translation.append(cam_info["sensor2lidar_translation"])

                # mono_infos
                if not self.test_mode:
                    cam_anno_infos.append(
                        self.get_mono_anno_info(
                            cam_type=cam_type,
                            foldername=str(info["token"]),
                            cam_info=cam_info,
                        )
                    )

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam2img=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    sensor2lidar_rotation=sensor2lidar_rotation,
                    sensor2lidar_translation=sensor2lidar_translation,
                )
            )
            
            # load annos for each camera
            if not self.test_mode:
                input_dict["cam_anno_infos"] = cam_anno_infos
        
        # load annos for the entire scene 
        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict["ann_info"] = annos

        return input_dict

    def get_mono_anno_info(self, cam_type, cam_info, foldername=None):
        sample_data_token = cam_info["sample_data_token"]
        mono_ann_info_ids = self.coco.get_ann_ids(img_ids=[sample_data_token])
        mono_ann_info = self.coco.load_anns(mono_ann_info_ids)
        img_info = self.coco.load_imgs([sample_data_token])[0]

        gt_bboxes = []
        gt_labels = []
        attr_labels = []
        gt_bboxes_ignore = []
        gt_bboxes_cam3d = []
        centers2d = []
        depths = []
        gt_tokens = []

        for i, ann in enumerate(mono_ann_info):
            if ann.get("ignore", False):
                continue
            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))

            if inter_w * inter_h == 0:
                continue
            if ann["area"] <= 0 or w < 1 or h < 1:
                continue
            if ann["category_id"] not in self.coco_cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get("iscrowd", False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2id[ann["category_name"]])
                attr_labels.append(ann["attribute_id"])
                # 3D annotations in camera coordinates
                bbox_cam3d = np.array(ann["bbox_cam3d"]).reshape(1, -1)
                velo_cam3d = np.array(ann["velo_cam3d"]).reshape(1, 2)
                nan_mask = np.isnan(velo_cam3d[:, 0])
                velo_cam3d[nan_mask] = [0.0, 0.0]
                bbox_cam3d = np.concatenate([bbox_cam3d, velo_cam3d], axis=-1)
                gt_bboxes_cam3d.append(bbox_cam3d.squeeze())
                # 2.5D annotations in camera coordinates
                center2d = ann["center2d"][:2]
                depth = ann["center2d"][2]
                centers2d.append(center2d)
                depths.append(depth)
                gt_tokens.append(ann["token"])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            attr_labels = np.array(attr_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            attr_labels = np.array([], dtype=np.int64)

        if gt_bboxes_cam3d:
            gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
            centers2d = np.array(centers2d, dtype=np.float32)
            depths = np.array(depths, dtype=np.float32)
            gt_tokens = np.array(gt_tokens)
        else:
            gt_bboxes_cam3d = np.zeros((0, self.bbox_code_size), dtype=np.float32)
            centers2d = np.zeros((0, 2), dtype=np.float32)
            depths = np.zeros((0), dtype=np.float32)
            gt_tokens = np.zeros((0))

        gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d, box_dim=gt_bboxes_cam3d.shape[-1], origin=(0.5, 0.5, 0.5)
        )
        gt_labels_3d = copy.deepcopy(gt_labels)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if len(gt_bboxes_cam3d.tensor) > 0:
            # generate 2D corners and the visibility
            gt_corners_3d = gt_bboxes_cam3d.corners.numpy()
            K = np.array(img_info["cam_intrinsic"]).T[np.newaxis]
            gt_corners_2d = np.matmul(gt_corners_3d, K)
            gt_corners_2d = gt_corners_2d[..., :2] / gt_corners_2d[..., 2:]

            valid_mask_x = (gt_corners_2d[..., 0] >= 0) & (
                gt_corners_2d[..., 0] <= img_info["width"] - 1
            )
            valid_mask_y = (gt_corners_2d[..., 1] >= 0) & (
                gt_corners_2d[..., 1] <= img_info["height"] - 1
            )
            valid_mask = valid_mask_x & valid_mask_y

            gt_corners_2d = np.concatenate(
                (gt_corners_2d, valid_mask[..., np.newaxis]), axis=2
            )
        else:
            gt_corners_2d = np.zeros((0, 8, 3))

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_bboxes_3d=gt_bboxes_cam3d,
            gt_corners_2d=gt_corners_2d,
            gt_labels_3d=gt_labels_3d,
            attr_labels=attr_labels,
            centers2d=centers2d,
            depths=depths,
            bboxes_ignore=gt_bboxes_ignore,
            gt_tokens=gt_tokens,
        )

        return ann

    def visualize(
        self,
        img_metas,
        cam_anno_infos,
    ):
        import os
        import matplotlib.pyplot as plt
        import imageio
        from mmdet3d.core.visualizer.image_vis import (
            plot_rect3d_on_img,
            plot_bbox2d_on_img,
        )

        save_path = "debug_visualize/{}".format("dataset_label")
        os.makedirs(save_path, exist_ok=True)

        img_metas = img_metas._data
        img_filenames = img_metas["filename"]

        imgs = [imageio.imread(img_file) for img_file in img_filenames]
        imgs = [np.ascontiguousarray(img.astype(np.uint8)) for img in imgs]

        gt_bboxes_list = [info["bboxes"]._data.numpy() for info in cam_anno_infos]
        gt_corners_2d_list = [
            info["gt_corners_2d"]._data.numpy() for info in cam_anno_infos
        ]

        num_row = 2
        num_col = 3
        # visualize bbox
        plt.figure(0, figsize=(16, 9))
        for ii in range(num_row):
            for jj in range(num_col):
                index = ii * num_col + jj
                plt.subplot(num_row, num_col, index + 1)
                img_with_bbox = plot_bbox2d_on_img(
                    imgs[index], bboxes=gt_bboxes_list[index]
                )
                plt.imshow(img_with_bbox)
                plt.axis("off")
        plt.savefig("{}/bbox_2D.png".format(save_path))
        plt.close(0)

        plt.figure(0, figsize=(16, 9))
        for ii in range(num_row):
            for jj in range(num_col):
                index = ii * num_col + jj
                plt.subplot(num_row, num_col, index + 1)
                gt_corners_2d = gt_corners_2d_list[index]
                img_with_bbox = plot_rect3d_on_img(
                    imgs[index],
                    num_rects=gt_corners_2d.shape[0],
                    rect_corners=gt_corners_2d,
                )
                plt.imshow(img_with_bbox)
                plt.axis("off")
        plt.savefig("{}/bbox_3D.png".format(save_path))
        plt.close(0)

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0

        gt_bboxes_3d = info["gt_boxes"]
        gt_names_3d = info["gt_names"]
        gt_tokens = np.array(info["gt_tokens"])

        # valid masking
        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_names_3d = gt_names_3d[mask]
        gt_tokens = gt_tokens[mask]
        gt_labels_3d = []

        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)

        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_tokens=gt_tokens,
        )

        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.eval_detection_configs,
                self.eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = NuScenesMultiviewDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = NuScenesMultiviewDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, metric="bbox", result_name="pts_bbox"
    ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True,
        )
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, "metrics_summary.json"))
        detail = dict()
        metric_prefix = f"{result_name}_NuScenes"
        for name in self.CLASSES:
            for k, v in metrics["label_aps"][name].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}_AP_dist_{}".format(metric_prefix, name, k)] = val
            for k, v in metrics["label_tp_errors"][name].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}_{}".format(metric_prefix, name, k)] = val
            for k, v in metrics["tp_errors"].items():
                val = float("{:.4f}".format(v))
                detail["{}/{}".format(metric_prefix, self.ErrNameMapping[k])] = val

        detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
        detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]

        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update({name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print("Evaluating bboxes of {}".format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend="disk"),
            ),
            dict(
                type="LoadPointsFromMultiSweeps",
                sweeps_num=10,
                file_client_args=dict(backend="disk"),
            ),
            dict(
                type="DefaultFormatBundle3D", class_names=self.CLASSES, with_label=False
            ),
            dict(type="Collect3D", keys=["points"]),
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, "Expect out_dir, got none."
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if "pts_bbox" in result.keys():
                result = result["pts_bbox"]
            data_info = self.data_infos[i]
            pts_path = data_info["lidar_path"]
            file_name = osp.split(pts_path)[-1].split(".")[0]
            points = self._extract_data(i, pipeline, "points").numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(
                points, Coord3DMode.LIDAR, Coord3DMode.DEPTH
            )
            inds = result["scores_3d"] > 0.1
            gt_bboxes = self.get_ann_info(i)["gt_bboxes_3d"].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(
                gt_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH
            )
            pred_bboxes = result["boxes_3d"][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(
                pred_bboxes, Box3DMode.LIDAR, Box3DMode.DEPTH
            )
            show_result(
                points, show_gt_bboxes, show_pred_bboxes, out_dir, file_name, show
            )


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info, boxes, classes, eval_configs, eval_version="detection_cvpr_2019"
):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list
