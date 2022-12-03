from multiprocessing.sharedctypes import Value
import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
import pdb
import torch
import cv2
import pyquaternion
from PIL import Image, ImageFile

from mmdet3d.core.bbox import (
    CameraInstance3DBoxes,
    DepthInstance3DBoxes,
    LiDARInstance3DBoxes,
    Box3DMode,
    box_np_ops,
)


@PIPELINES.register_module()
class PadMultiViewImage(object):
    """Pad the multi-view image.
    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number. padding 过程发生在图片下方和右方，不会影响 2d centers/keypoints
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",
    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """ Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = [
                mmcv.impad(img, shape=self.size, pad_val=self.pad_val)
                for img in results["img"]
            ]
        elif self.size_divisor is not None:
            padded_img = [
                mmcv.impad_to_multiple(img, self.size_divisor, pad_val=self.pad_val)
                for img in results["img"]
            ]
        
        # padding on the right and bottom borders
        results["img"] = padded_img
        results["img_shape"] = [img.shape for img in padded_img]
        results["pad_shape"] = [img.shape for img in padded_img]
        results["pad_fixed_size"] = self.size
        results["pad_size_divisor"] = self.size_divisor

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        repr_str += f"size_divisor={self.size_divisor}, "
        repr_str += f"pad_val={self.pad_val})"
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        # save raw_imgs for debugging
        results['raw_img'] = torch.stack([torch.from_numpy(x) for x in results["img"]], dim=0)
        
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str


@PIPELINES.register_module()
class CropMultiViewImage(object):
    """Crop the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, size=None):
        self.size = size

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        results["img"] = [
            img[: self.size[0], : self.size[1], ...] for img in results["img"]
        ]
        results["img_shape"] = [img.shape for img in results["img"]]
        results["img_fixed_size"] = self.size
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(size={self.size}, "
        return repr_str

@PIPELINES.register_module()
class HorizontalRandomFlipMultiViewImage(object):
    def __init__(self, flip_ratio=0.5):
        self.flip_ratio = flip_ratio

    def __call__(self, results):
        
        # interested keys in cam_infos
        for cam_index, img in enumerate(results['img']):
            # whether to flip
            if np.random.random() > self.flip_ratio:
                continue
            
            # flip image
            flip_img = mmcv.imflip(img, "horizontal")
            '''
            flip camera-view annotations, including bboxes, centers2d, gt_corners_2d, and gt_bboxes_3d
            '''
            h, w = img.shape[:2]
            flip_cam_anno_infos = results['cam_anno_infos'][cam_index]
            '''
            # check img2cam mapping
            cam2img = results["cam2img"][cam_index]
            cam2img = torch.from_numpy(cam2img).float()
            inv_K = torch.inverse(cam2img).transpose(0, 1)
            centers2d = torch.from_numpy(flip_cam_anno_infos['centers2d']).float()
            depths = torch.from_numpy(flip_cam_anno_infos['depths']).float()[..., None]
            homo_centers2d = torch.cat(
                (centers2d * depths, depths, torch.ones_like(depths)), dim=1
            )
            centers3d = torch.mm(homo_centers2d, inv_K)
            '''

            # 1. flip bboxes
            flip_bboxes = flip_cam_anno_infos['bboxes']
            flip_bboxes[:, [0, 2]] = w - flip_bboxes[:, [0, 2]]
            flip_cam_anno_infos['bboxes'] = flip_bboxes
            # 2. flip centers2d
            flip_centers2d = flip_cam_anno_infos['centers2d']
            flip_centers2d[:, 0] = w - flip_centers2d[:, 0]
            flip_cam_anno_infos['centers2d'] = flip_centers2d
            # 3. flip corners2d
            flip_corners2d = flip_cam_anno_infos['gt_corners_2d']
            flip_corners2d[..., 0] = w - flip_corners2d[..., 0]
            flip_cam_anno_infos['gt_corners_2d'] = flip_corners2d
            # 4. flip gt_bboxes_3d
            cam2img = results["cam2img"][cam_index]
            cam2img = torch.from_numpy(cam2img).float()
            inv_K = torch.inverse(cam2img).transpose(0, 1)
            flip_centers2d = torch.from_numpy(flip_centers2d).float()
            depths = torch.from_numpy(flip_cam_anno_infos['depths']).float()[..., None]
            homo_centers2d = torch.cat(
                (flip_centers2d * depths, depths, torch.ones_like(depths)), dim=1
            )
            flip_centers3d = torch.mm(homo_centers2d, inv_K)
            flip_bboxes_3d = flip_cam_anno_infos['gt_bboxes_3d'].clone()
            flip_bboxes_3d.tensor[:, [0, 2]] = flip_centers3d[:, [0, 2]]
            flip_bboxes_3d.tensor[:, 6] = np.pi - flip_bboxes_3d.tensor[:, 6]
            flip_bboxes_3d.tensor[:, 7] = - flip_bboxes_3d.tensor[:, 7]
            flip_cam_anno_infos['gt_bboxes_3d'] = flip_bboxes_3d

            results['img'][cam_index] = flip_img
            results['cam_anno_infos'][cam_index] = flip_cam_anno_infos
        
        return results

# @PIPELINES.register_module()
# class MergeMultiViewObjects(object):
#     def __init__(self, remove_dumplicate=True, matched_only=False):
#         self.remove_dumplicate = remove_dumplicate
#         self.matched_only = matched_only

#         self.filter_keys = [
#             "bboxes",
#             "labels",
#             "gt_bboxes_3d",
#             "gt_corners_2d",
#             "gt_labels_3d",
#             "attr_labels",
#             "centers2d",
#             "depths",
#             "gt_tokens",
#         ]

#     def __call__(self, results):
#         transform_bboxes_lidar = []
#         gt_labels_3d = []
#         gt_tokens = []

#         input_gt_tokens = results["gt_tokens"].copy()
#         for cam_index, cam_anno_info in enumerate(results['cam_anno_infos']):
#             sensor2lidar_rotation = results['sensor2lidar_rotation'][cam_index]
#             sensor2lidar_translation = results['sensor2lidar_translation'][cam_index]

#             sensor_bboxes = cam_anno_info['gt_bboxes_3d'].clone()
#             # transform velocity manually
#             velos = sensor_bboxes.tensor[:, 7:]
#             homo_velos = torch.cat((velos[:, :1], torch.zeros(velos.shape[0], 1), velos[:, 1:]), dim=1)
#             # cam_velo to lidar_velo
#             transform_velos = homo_velos @ sensor2lidar_rotation.T
#             transform_velos = transform_velos[:, :2]

#             sensor_bboxes.rotate(sensor2lidar_rotation.T)
#             sensor_bboxes.translate(sensor2lidar_translation)
#             sensor_bboxes.tensor[:, 7:] = transform_velos

#             # filtering
#             if self.matched_only:
#                 valid_mask = np.isin(cam_anno_info["gt_tokens"], input_gt_tokens)
#                 for filter_key in self.filter_keys:
#                     cam_anno_info[filter_key] = cam_anno_info[filter_key][valid_mask]
                
#                 results['cam_anno_infos'][cam_index] = cam_anno_info
#                 sensor_bboxes = sensor_bboxes[valid_mask]

#             transform_bboxes_lidar.append(sensor_bboxes)
#             gt_tokens.append(cam_anno_info["gt_tokens"])
#             gt_labels_3d.append(cam_anno_info["gt_labels_3d"])
        
#         gt_tokens = np.concatenate(gt_tokens)
#         gt_labels_3d = np.concatenate(gt_labels_3d)

#         transform_bboxes_lidar = CameraInstance3DBoxes.cat(transform_bboxes_lidar)
#         # convert xyz size
#         arr = transform_bboxes_lidar.tensor.clone()
#         x_size, y_size, z_size = arr[..., 3:4], arr[..., 4:5], arr[..., 5:6]
#         xyz_size = torch.cat([z_size, x_size, y_size], dim=-1)
#         transform_bboxes_lidar.tensor[:, 3:6] = xyz_size
#         transform_bboxes_lidar.tensor[:, 6] -= np.pi / 2
#         transform_bboxes_lidar = LiDARInstance3DBoxes(
#             tensor=transform_bboxes_lidar.tensor,
#             box_dim=transform_bboxes_lidar.box_dim,
#             with_yaw=transform_bboxes_lidar.with_yaw,
#         )

#         # transform_bboxes_lidar = transform_bboxes_lidar.convert_to(Box3DMode.LIDAR)
#         results["gt_bboxes_3d"].limit_yaw(offset=0.5, period=2 * np.pi)
#         transform_bboxes_lidar.limit_yaw(offset=0.5, period=2 * np.pi)
        
#         # # velocity 转换有问题
#         # for i in range(gt_tokens.shape[0]):
#         #     sample_token = gt_tokens[i]
#         #     sample_index = np.where(np.array(results['gt_tokens']) == sample_token)[0]
#         #     sample_bbox_3d = results["gt_bboxes_3d"].tensor[sample_index]
#         #     print('GT: ', sample_bbox_3d)
#         #     print('CAM_RT: ', transform_bboxes_lidar.tensor[i])

#         if self.remove_dumplicate:
#             unique_tokens, unique_token_idxs = np.unique(
#                 gt_tokens,
#                 return_index=True,
#             )
#             results["gt_tokens"] = unique_tokens
#             results["gt_labels_3d"] = gt_labels_3d[unique_token_idxs]
#             results["gt_bboxes_3d"] = transform_bboxes_lidar[unique_token_idxs]
#         else:
#             # generate match_idxs
#             total = gt_tokens.shape[0]
#             current = 0
#             indices = np.arange(total)
#             for cam_index, cam_anno_info in enumerate(results['cam_anno_infos']):
#                 num_gt = cam_anno_info['bboxes'].shape[0]
#                 cam_anno_info['match_idxs'] = indices[current : current + num_gt]
#                 current += num_gt

#         return results