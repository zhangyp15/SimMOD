_base_ = [
    "../../../mmdetection3d/configs/_base_/datasets/nus-3d.py",
    "../../../mmdetection3d/configs/_base_/default_runtime.py",
]

plugin = True
plugin_dir = "projects/mmdet3d_plugin/"

find_unused_parameters = False
sync_bn = True
fp16 = dict(loss_scale='dynamic')
load_from = "ckpts/cascade_mask_rcnn_r101_fpn_1x_nuim_process.pth"

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

input_modality = dict(
    use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False,
)

num_cam = 6
num_proposal = 600
num_proposal_per_image = num_proposal // num_cam
generate_proposal_per_image = False

model = dict(
    type="SimMOD",
    use_grid_mask=True,
    filter_gt_with_proposals=True,
    img_backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    img_roi_head=dict(
        type="FCOSMono3D_ProposalHead",
        objectness_max_pooling=True,
        depth_with_uncertainty=True,
        num_classes=10,
        cls_agnostic=False,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        strides=[8, 16, 32, 64],
        dcn_on_last_conv=True,
        conv_bias=True,
        use_direction_classifier=True,
        diff_rad_by_sin=True,
        dir_offset=0.7854,  # pi/4
        regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 1e8)),
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        # proposals
        num_proposal=num_proposal,
        generate_proposal_per_image=generate_proposal_per_image,
        objectness_with_centerness=True,
        random_objectness_with_teacher=0.5,
        random_proposal_drop=False,
        random_proposal_drop_upper_bound=1.0,
        random_proposal_drop_lower_bound=0.7,
        proposal_filtering=True,
        proposal_score_thresh=0.1,
        minimal_proposal_number=100,
        # main branches
        cls_branch=(256,),
        reg_keys=(
            "offset",
            "depth",
            "size",
            "rot",
            "bbox2d",
            "corners",
            "velo",
        ),
        reg_branch=(
            (256, 2),  # offset
            (256, 2),  # depth
            (256, 3),  # size
            (256, 1),  # rot
            (256, 4),  # bbox2d
            (256, 16),  # corners
            (2,),  # velo
        ),
        reg_weights=(
            1.0,  # offset
            0.5,  # depth
            1.0,  # size
            1.0,  # rot
            1.0,  # bbox2d
            0.2,  # corners
            0.05,  # velo
        ),
        dir_branch=(256,),
        # auxiliary branches
        attr_branch=(256,),
        pred_attrs=True,
        # loss functions
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0,
        ),
        loss_bbox=dict(type="SmoothL1Loss", beta=1.0 / 9.0, loss_weight=1.0),
        loss_dir=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        loss_attr=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
        loss_bbox2d=dict(
            type="IOULoss", loss_type="giou", return_iou=False
        ),  # IoU loss for 2D detection
        loss_centerness=dict(
            type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0
        ),
        centerness_on_reg=True,
        train_cfg=dict(
            allowed_border=0,
            pos_weight=-1,
            debug=False,
        ),
        test_cfg=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_thr=0.8,
            score_thr=0.05,
            min_bbox_size=0,
            max_per_img=200,
        ),
    ),
    pts_bbox_head=dict(
        type="SimMODHead",
        num_query=num_proposal,
        num_classes=10,
        in_channels=256,
        input_proposal_channel=512,
        num_input_proj=2,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        detach_proposal_positions=False,
        # positional embeddings
        using_pos_embeddings=True,
        use_cam_level_embeddings=True,
        proposal_level_embeddings=True,
        proposal_cam_embeddings=True,
        proposal_embeddings_additive=False,
        transformer=dict(
            type="TwoStageDetr3DTransformer",
            decoder=dict(
                type="TwoStageDetr3DTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                iterative_pos_encoding=False,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1,
                        ),
                        dict(
                            type="TwoStageDetr3DCrossAtten",
                            pc_range=point_cloud_range,
                            num_points=1,
                            embed_dims=256,
                        ),
                    ],
                    feedforward_channels=512,
                    ffn_dropout=0.1,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        bbox_coder=dict(
            type="NMSFreeCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10,
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, normalize=True, offset=-0.5,
        ),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=0.25),
        loss_iou=dict(type="GIoULoss", loss_weight=0.0),
    ),
    # model training and testing settings
    train_cfg=dict(
        two_stage_loss_weights=[
            1.0,
            1.0,
        ],  # balancing the loss weights of two-stage training
        pts=dict(
            grid_size=[512, 512, 1],
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type="HungarianAssigner3D",
                cls_cost=dict(type="FocalLossCost", weight=2.0),
                reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                iou_cost=dict(
                    type="IoUCost", weight=0.0
                ),  # Fake cost. This is just to make it compatible with DETR head.
                pc_range=point_cloud_range,
            ),
        ),
    ),
)

dataset_type = "NuScenesMultiviewDataset"
data_root = "/mnt/cfs/algorithm/junjie.huang/data"

train_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="PhotoMetricDistortionMultiViewImage"),
    dict(
        type="LoadAnnotations3D",
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_tokens=True,
    ),
    dict(type="CustomObjectNameFilter", classes=class_names),
    dict(type="CustomObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    # random scale ==> padding
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=[
            "gt_bboxes_3d",
            "gt_labels_3d",
            "img",
            "cam_anno_infos",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadMultiViewImageFromFiles", to_float32=True),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="DefaultFormatBundle3D", class_names=class_names, with_label=False
            ),
            dict(
                type="Collect3D",
                keys=[
                    "img",
                    "raw_img",
                ],
            ),
        ],
    ),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="data/nuscenes_infos/nuscenes_infos_train.pkl",
        mono_anno_file="data/nuscenes_infos/nuscenes_infos_train_mono3d.coco.json",
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d="LiDAR",
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="data/nuscenes_infos/nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="data/nuscenes_infos/nuscenes_infos_val.pkl",
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
    ),
)

optimizer = dict(
    type="AdamW",
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            "img_backbone": dict(lr_mult=0.1),
        }
    ),
    weight_decay=0.01,
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[20, 23],
)

total_epochs = 24
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
evaluation = dict(interval=8, pipeline=test_pipeline)