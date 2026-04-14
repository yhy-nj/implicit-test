"""
FRNet + Explicit Constraint + Implicit Constraint for SemanticKITTI.

在 frnet-explicit-semantickitti_seg.py 基础上的改动:
1. type: FRNetExplicit -> FRNetExplicitImplicit (segmentor)
2. data_preprocessor.type: ExplicitConstraintPreprocessor -> ExplicitImplicitPreprocessor
3. backbone.type: FRNetExplicitBackbone -> FRNetExplicitImplicitBackbone (+ 隐式约束参数)
4. 新增 implicit_loss_weight 参数

其余所有参数(voxel_encoder, decode_head, auxiliary_head, pipeline)
全部照抄 frnet-explicit-semantickitti_seg.py, 不做任何修改。

放置路径: MY-WORK(3)/configs/frnet/frnet-explicit-implicit-semantickitti_seg.py
"""

_base_ = [
    '../_base_/datasets/semantickitti_seg.py',
    '../_base_/models/frnet.py',
    '../_base_/schedules/onecycle-50k.py',
    '../_base_/default_runtime.py'
]

custom_imports = dict(
    imports=['frnet.datasets', 'frnet.datasets.transforms', 'frnet.models'],
    allow_failed_imports=False)

# ============================================================
# Model: 在 explicit 基础上新增 implicit 分支
# ============================================================
model = dict(
    # 改动1: FRNetExplicit -> FRNetExplicitImplicit (segmentor)
    type='FRNetExplicitImplicit',

    # 改动2: ExplicitConstraintPreprocessor -> ExplicitImplicitPreprocessor
    data_preprocessor=dict(
        type='ExplicitImplicitPreprocessor',
        H=64, W=512, fov_up=3.0, fov_down=-25.0, ignore_index=19,
        image_size=(370, 1226),
        image_stride=8),

    # 改动3: FRNetExplicitBackbone -> FRNetExplicitImplicitBackbone
    backbone=dict(
        type='FRNetExplicitImplicitBackbone',
        output_shape=(64, 512),
        # 显式约束参数 (与 explicit 版本一致)
        enable_explicit=True,
        explicit_image_channels=128,
        explicit_align_channels=128,
        explicit_out_channels=128,
        explicit_num_samples=9,
        # 隐式约束参数 (新增)
        enable_implicit=True,
        implicit_feat_channels=128,     # 必须与图像backbone输出通道数一致
        implicit_hidden_channels=128,
        implicit_num_mlp_layers=2,
        implicit_alpha_min=0.01),

    # 新增: 隐式约束损失权重 λ_imp
    implicit_loss_weight=1.0,

    # decode_head: 只覆盖 num_classes 和 ignore_index
    decode_head=dict(num_classes=20, ignore_index=19),

    # auxiliary_head: 与 explicit 版本完全一致
    auxiliary_head=[
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=2),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=3),
        dict(
            type='FrustumHead',
            channels=128,
            num_classes=20,
            dropout_ratio=0,
            loss_ce=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                class_weight=None,
                loss_weight=1.0),
            loss_lovasz=dict(
                type='LovaszLoss', loss_weight=1.5, reduction='none'),
            loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
            conv_seg_kernel_size=1,
            ignore_index=19,
            indices=4),
    ],
)

# ============================================================
# Pipeline: 与 explicit 版本完全一致
# ============================================================
backend_args = None

pre_transform = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False,
         with_seg_3d=True, seg_3d_dtype='np.int32', seg_offset=2**16,
         dataset_type='semantickitti', backend_args=backend_args),
    dict(type='LoadCalibration'),
    dict(type='LoadImageFromFile'),
    dict(type='PointSegClassMapping'),
    dict(type='RandomFlip3D', sync_2d=False,
         flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-3.1415926, 3.1415926],
         scale_ratio_range=[0.95, 1.05], translation_std=[0.1, 0.1, 0.1]),
]

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False,
         with_seg_3d=True, seg_3d_dtype='np.int32', seg_offset=2**16,
         dataset_type='semantickitti', backend_args=backend_args),
    dict(type='LoadCalibration'),
    dict(type='LoadImageFromFile'),
    dict(type='PointSegClassMapping'),
    dict(type='RandomFlip3D', sync_2d=False,
         flip_ratio_bev_horizontal=0.5, flip_ratio_bev_vertical=0.5),
    dict(type='GlobalRotScaleTrans', rot_range=[-3.1415926, 3.1415926],
         scale_ratio_range=[0.95, 1.05], translation_std=[0.1, 0.1, 0.1]),
    dict(type='FrustumMix', H=64, W=512, fov_up=3.0, fov_down=-25.0,
         num_areas=[3, 4, 5, 6], pre_transform=pre_transform, prob=1.0),
    dict(type='InstanceCopy',
         instance_classes=[1, 2, 3, 4, 5, 6, 7, 11, 15, 17, 18],
         pre_transform=pre_transform, prob=1.0),
    dict(type='RangeInterpolation', H=64, W=2048,
         fov_up=3.0, fov_down=-25.0, ignore_index=19),
    dict(type='Pack3DDetInputs',
         keys=['points', 'pts_semantic_mask',],
         meta_keys=['num_points', 'lidar2img', 'img_path', 'img']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR',
         load_dim=4, use_dim=4, backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=False, with_label_3d=False,
         with_seg_3d=True, seg_3d_dtype='np.int32', seg_offset=2**16,
         dataset_type='semantickitti', backend_args=backend_args),
    dict(type='LoadCalibration'),
    dict(type='LoadImageFromFile'),
    dict(type='PointSegClassMapping'),
    dict(type='RangeInterpolation', H=64, W=2048,
         fov_up=3.0, fov_down=-25.0, ignore_index=19),
    dict(type='Pack3DDetInputs',
         keys=['points'],
         meta_keys=['num_points', 'lidar2img', 'img_path', 'img']),
]

# 用新pipeline覆盖base里的dataloader配置
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
