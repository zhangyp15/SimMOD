from multiprocessing.sharedctypes import Value
from mmdet.datasets.builder import PIPELINES

import numpy as np
import mmcv
import pdb
import imageio
import torch
import os
import cv2
import pyquaternion
import matplotlib.pyplot as plt

from PIL import Image, ImageFile
from mmdet3d.core.bbox import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    Box3DMode,
    box_np_ops,
)


@PIPELINES.register_module()
class CustomObjectNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )

        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]

        if "gt_tokens" in input_dict:
            input_dict["gt_tokens"] = input_dict["gt_tokens"][gt_bboxes_mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str
    
    
@PIPELINES.register_module()
class CustomObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range
        if isinstance(
            input_dict["gt_bboxes_3d"], (LiDARInstance3DBoxes, DepthInstance3DBoxes)
        ):
            bev_range = self.pcd_range[[0, 1, 3, 4]]

        elif isinstance(input_dict["gt_bboxes_3d"], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        # mask is a torch tensor but gt_labels_3d is still numpy array
        # using mask to index gt_labels_3d will cause bug when
        # len(gt_labels_3d) == 1, where mask=1 will be interpreted
        # as gt_labels_3d[1] and cause out of index error
        numpy_mask = mask.numpy().astype(np.bool)
        gt_labels_3d = gt_labels_3d[numpy_mask]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d

        if "gt_tokens" in input_dict:
            gt_tokens_3d = input_dict["gt_tokens"][numpy_mask]
            input_dict["gt_tokens"] = gt_tokens_3d
        else:
            gt_tokens_3d = None
        
        if "cam_anno_infos" in input_dict:
            filter_keys = [
                "bboxes",
                "labels",
                "gt_bboxes_3d",
                "gt_corners_2d",
                "gt_labels_3d",
                "attr_labels",
                "centers2d",
                "depths",
                "gt_tokens",
            ]

            for cam_index, anno_info in enumerate(input_dict["cam_anno_infos"]):
                cam_anno_tokens = anno_info["gt_tokens"]
                valid_mask = np.isin(cam_anno_tokens, gt_tokens_3d)

                for filter_key in filter_keys:
                    input_dict["cam_anno_infos"][cam_index][filter_key] = anno_info[
                        filter_key
                    ][valid_mask]

                if valid_mask.sum() > 0:
                    valid_cam_anno_tokens = cam_anno_tokens[valid_mask].reshape(-1, 1)
                    rep_gt_tokens_3d = np.repeat(
                        gt_tokens_3d.reshape(1, -1),
                        valid_cam_anno_tokens.shape[0],
                        axis=0,
                    )
                    valid_cam_anno_tokens = np.repeat(
                        valid_cam_anno_tokens,
                        rep_gt_tokens_3d.shape[1],
                        axis=1,
                    )
                    match_idxs = np.argwhere(valid_cam_anno_tokens == rep_gt_tokens_3d)[
                        :, 1
                    ]
                else:
                    match_idxs = np.zeros(0)

                input_dict["cam_anno_infos"][cam_index]["match_idxs"] = match_idxs

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(point_cloud_range={self.pcd_range.tolist()})"
        return repr_str