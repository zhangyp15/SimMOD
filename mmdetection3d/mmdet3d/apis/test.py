# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import time
from mmcv.image import tensor2imgs
from os import path as osp

from mmdet3d.models import (
    Base3DDetector,
    Base3DSegmentor,
    SingleStageMono3DDetector,
)

import pdb


def single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))

    forward_time = []

    for i, data in enumerate(data_loader):
        if show:
            gt_bboxes_3d = data.pop('gt_bboxes_3d', None)
            gt_labels_3d = data.pop('gt_labels_3d', None)
            cam_anno_infos = data.pop('cam_anno_infos', None)
            points = data.pop('points', None)
            
            # data['img'] = [data['img']]
            # data['img_metas'] = [data['img_metas']]

        with torch.no_grad():
            torch.cuda.synchronize()
            s1 = time.time()

            result = model(return_loss=False, rescale=True, **data)

            torch.cuda.synchronize()
            s2 = time.time()

            forward_time.append(s2 - s1)

        if show:
            # Visualize the results of MMDetection3D model
            # 'show_results' is MMdetection3D visualization API
            models_3d = (
                Base3DDetector,
                Base3DSegmentor,
                SingleStageMono3DDetector,
            )
            
            data['gt_bboxes_3d'] = gt_bboxes_3d
            data['gt_labels_3d'] = gt_labels_3d
            data['cam_anno_infos'] = cam_anno_infos
            data['points'] = points

            if isinstance(model.module, models_3d) or hasattr(
                model.module, "mono3d_visualize"
            ):
                this_out_dir = osp.join(out_dir, 'sample_{:04d}'.format(i))
                model.module.show_results(data, result, out_dir=this_out_dir)

            # Visualize the results of MMDetection model
            # 'show_result' is MMdetection visualization API
            else:
                batch_size = len(result)
                if batch_size == 1 and isinstance(data["img"][0], torch.Tensor):
                    img_tensor = data["img"][0]
                else:
                    img_tensor = data["img"][0].data[0]
                img_metas = data["img_metas"][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]["img_norm_cfg"])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta["img_shape"]
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta["ori_shape"][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta["ori_filename"])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=show_score_thr,
                    )
        
        results.extend(result)
        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()
        
        if i > 20:
            times = forward_time[20:]
            avg_latency = sum(times) / len(times)
            print(
                ", average forward time = {:.2f}, fps = {:.2f}".format(
                    avg_latency,
                    1 / avg_latency,
                )
            )

    return results
