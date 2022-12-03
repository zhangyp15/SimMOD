_base_ = [
    "./simmod_r50.py",
]

fp16 = dict(loss_scale='dynamic')
load_from = None

model = dict(
    img_backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        pretrained="open-mmlab://detectron2/resnet101_caffe",
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
)