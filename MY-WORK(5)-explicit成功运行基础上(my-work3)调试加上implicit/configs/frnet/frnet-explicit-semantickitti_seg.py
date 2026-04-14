"""
FRNet + Explicit Constraint for SemanticKITTI.

和原始 frnet-semantickitti_seg.py 的区别只有3处:
1. type: FRNet -> FRNetExplicit
2. data_preprocessor.type: FrustumRangePreprocessor -> ExplicitConstraintPreprocessor
3. backbone.type: FRNetBackbone -> FRNetExplicitBackbone (+ 显式约束参数)

其余所有参数(voxel_encoder, decode_head, auxiliary_head)
全部照抄 _base_/models/frnet.py, 不做任何修改。

放置路径: MY-WORK(3)/configs/frnet/frnet-explicit-semantickitti_seg.py
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
# Model: 只覆盖3个type, 其余参数全部从 _base_/models/frnet.py 继承
# ============================================================
model = dict(
    # 改动1: FRNet -> FRNetExplicit (segmentor)
    type='FRNetExplicit',

    # 改动2: FrustumRangePreprocessor -> ExplicitConstraintPreprocessor
    data_preprocessor=dict(
        type='ExplicitConstraintPreprocessor',
        H=64, W=512, fov_up=3.0, fov_down=-25.0, ignore_index=19,
        image_size=(370, 1226),
        image_stride=8),

    # 改动3: FRNetBackbone -> FRNetExplicitBackbone + 显式约束参数
    # 其余backbone参数(in_channels=16, point_in_channels=384, depth=34等)
    # 全部从 _base_/models/frnet.py 继承, 不需要重写
    backbone=dict(
        type='FRNetExplicitBackbone',
        output_shape=(64, 512),
        enable_explicit=True,
        explicit_image_channels=128,
        explicit_align_channels=128,
        explicit_out_channels=128,
        explicit_num_samples=9),

    # decode_head: 只覆盖 num_classes 和 ignore_index, 其余继承base
    decode_head=dict(num_classes=20, ignore_index=19),

    # auxiliary_head: 和原始 frnet-semantickitti_seg.py 完全一致
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
# Pipeline: 覆盖 _base_/datasets/semantickitti_seg.py 中的 pipeline
# 加入 LoadCalibration + LoadImageFromFile, Pack3DDetInputs 加 imgs
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
         # keys=['points', 'pts_semantic_mask', 'imgs'],
         keys=['points', 'pts_semantic_mask',],
         meta_keys=['num_points', 'lidar2img','img_path','img']),
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
         # keys=['points', 'imgs'],
         # meta_keys=['num_points', 'lidar2img']),
         meta_keys=['num_points', 'lidar2img','img_path','img']),
]

# 用新pipeline覆盖base里的dataloader配置
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))


# """
# Config for FRNet with Explicit Constraint Branch (SemanticKITTI).
#
# This config replaces the original frnet-semantickitti_seg.py to enable
# the explicit constraint branch. Key differences from original FRNet config:
#
# 1. Model: FRNetExplicit (not FRNet)
# 2. Backbone: FRNetExplicitBackbone (not FRNetBackbone)
# 3. DataPreprocessor: ExplicitConstraintPreprocessor (not FrustumRangePreprocessor)
# 4. Pipeline: adds LoadCalibration + LoadImageFromFile
# 5. Pack3DDetInputs: keys includes 'imgs', meta_keys includes 'lidar2img'
#
# Usage:
#     python tools/train.py configs/frnet/frnet-explicit-semantickitti_seg.py
#
# 放置路径:
#     MY-WORK(3)/configs/frnet/frnet-explicit-semantickitti_seg.py
# """
#
# _base_ = [
#     '../_base_/schedules/onecycle-50k.py',
#     '../_base_/default_runtime.py'
# ]
#
# custom_imports = dict(
#     imports=[
#         'frnet.datasets',
#         'frnet.datasets.transforms',
#         'frnet.models',
#     ],
#     allow_failed_imports=False)
#
# # ============================================================
# # Dataset config (原本在 _base_/datasets/semantickitti_seg.py,
# # 但因为 pipeline 需要大幅修改, 这里直接内联写完整版)
# # ============================================================
#
# dataset_type = 'SemanticKITTIDataset'
# data_root = 'data/semantickitti/'
#
# class_names = [
#     'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person',
#     'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
#     'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
#     'pole', 'traffic-sign', 'unknown'
# ]
#
# labels_map = {
#     0: 19, 1: 19, 10: 0, 11: 1, 13: 4, 15: 2, 16: 4, 18: 3, 20: 4,
#     30: 5, 31: 6, 32: 7, 40: 8, 44: 9, 48: 10, 49: 11, 50: 12,
#     51: 13, 52: 19, 60: 8, 70: 14, 71: 15, 72: 16, 80: 17, 81: 18,
#     99: 19, 252: 0, 253: 6, 254: 5, 255: 7, 256: 4, 257: 4, 258: 3,
#     259: 4
# }
#
# metainfo = dict(
#     classes=class_names, seg_label_mapping=labels_map, max_label=259)
#
# # ---- 关键修改: use_camera=True ----
# input_modality = dict(use_lidar=True, use_camera=True)
#
# backend_args = None
#
# # ---- pre_transform: 加入 LoadCalibration 和 LoadImageFromFile ----
# pre_transform = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(
#         type='LoadAnnotations3D',
#         with_bbox_3d=False,
#         with_label_3d=False,
#         with_seg_3d=True,
#         seg_3d_dtype='np.int32',
#         seg_offset=2**16,
#         dataset_type='semantickitti',
#         backend_args=backend_args),
#     dict(type='LoadCalibration'),
#     dict(type='LoadImageFromFile'),
#     dict(type='PointSegClassMapping'),
#     dict(
#         type='RandomFlip3D',
#         sync_2d=False,
#         flip_ratio_bev_horizontal=0.5,
#         flip_ratio_bev_vertical=0.5),
#     dict(
#         type='GlobalRotScaleTrans',
#         rot_range=[-3.1415926, 3.1415926],
#         scale_ratio_range=[0.95, 1.05],
#         translation_std=[0.1, 0.1, 0.1]),
# ]
#
# # ---- train_pipeline: 加入图像加载, Pack 加入 imgs ----
# train_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(
#         type='LoadAnnotations3D',
#         with_bbox_3d=False,
#         with_label_3d=False,
#         with_seg_3d=True,
#         seg_3d_dtype='np.int32',
#         seg_offset=2**16,
#         dataset_type='semantickitti',
#         backend_args=backend_args),
#     dict(type='LoadCalibration'),
#     dict(type='LoadImageFromFile'),
#     dict(type='PointSegClassMapping'),
#     dict(
#         type='RandomFlip3D',
#         sync_2d=False,
#         flip_ratio_bev_horizontal=0.5,
#         flip_ratio_bev_vertical=0.5),
#     dict(
#         type='GlobalRotScaleTrans',
#         rot_range=[-3.1415926, 3.1415926],
#         scale_ratio_range=[0.95, 1.05],
#         translation_std=[0.1, 0.1, 0.1]),
#     dict(
#         type='FrustumMix',
#         H=64,
#         W=512,
#         fov_up=3.0,
#         fov_down=-25.0,
#         num_areas=[3, 4, 5, 6],
#         pre_transform=pre_transform,
#         prob=1.0),
#     dict(
#         type='InstanceCopy',
#         instance_classes=[1, 2, 3, 4, 5, 6, 7, 11, 15, 17, 18],
#         pre_transform=pre_transform,
#         prob=1.0),
#     dict(
#         type='RangeInterpolation',
#         H=64,
#         W=2048,
#         fov_up=3.0,
#         fov_down=-25.0,
#         ignore_index=19),
#     # ---- 关键修改: keys 加入 'imgs', meta_keys 加入 'lidar2img' ----
#     dict(
#         type='Pack3DDetInputs',
#         keys=['points', 'pts_semantic_mask', 'imgs'],
#         meta_keys=['num_points', 'lidar2img']),
# ]
#
# # ---- test_pipeline ----
# test_pipeline = [
#     dict(
#         type='LoadPointsFromFile',
#         coord_type='LIDAR',
#         load_dim=4,
#         use_dim=4,
#         backend_args=backend_args),
#     dict(
#         type='LoadAnnotations3D',
#         with_bbox_3d=False,
#         with_label_3d=False,
#         with_seg_3d=True,
#         seg_3d_dtype='np.int32',
#         seg_offset=2**16,
#         dataset_type='semantickitti',
#         backend_args=backend_args),
#     dict(type='LoadCalibration'),
#     dict(type='LoadImageFromFile'),
#     dict(type='PointSegClassMapping'),
#     dict(
#         type='RangeInterpolation',
#         H=64,
#         W=2048,
#         fov_up=3.0,
#         fov_down=-25.0,
#         ignore_index=19),
#     dict(
#         type='Pack3DDetInputs',
#         keys=['points', 'imgs'],
#         meta_keys=['num_points', 'lidar2img']),
# ]
#
# train_dataloader = dict(
#     batch_size=4,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='InfiniteSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='semantickitti_infos_train.pkl',
#         pipeline=train_pipeline,
#         metainfo=metainfo,
#         modality=input_modality,
#         ignore_index=19,
#         backend_args=backend_args))
#
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=1,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file='semantickitti_infos_val.pkl',
#         pipeline=test_pipeline,
#         metainfo=metainfo,
#         modality=input_modality,
#         ignore_index=19,
#         test_mode=True,
#         backend_args=backend_args))
#
# test_dataloader = val_dataloader
#
# val_evaluator = dict(type='SegMetric')
# test_evaluator = val_evaluator
#
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
#
# tta_model = dict(type='Seg3DTTAModel')
#
# # ============================================================
# # Model config
# # ============================================================
# model = dict(
#     # ---- 关键: 使用 FRNetExplicit 而不是 FRNet ----
#     type='FRNetExplicit',
#
#     # ---- 关键: 使用 ExplicitConstraintPreprocessor 而不是 FrustumRangePreprocessor ----
#     data_preprocessor=dict(
#         type='ExplicitConstraintPreprocessor',
#         H=64,
#         W=512,
#         fov_up=3.0,
#         fov_down=-25.0,
#         ignore_index=19,
#         image_size=(370, 1226),
#         image_stride=8),
#
#     voxel_encoder=dict(
#         type='FrustumFeatureEncoder',
#         # type='FRNetVoxelEncoder',
#         in_channels=4,
#         feat_channels=(64, 128),
#         with_distance=False,
#         voxel_size=(1, 1, 1),
#         point_cloud_range=(-50, -50, -4, 50, 50, 2)),
#
#     # ---- 关键: 使用 FRNetExplicitBackbone 而不是 FRNetBackbone ----
#     backbone=dict(
#         type='FRNetExplicitBackbone',
#         in_channels=128,
#         point_in_channels=128,
#         output_shape=(64, 512),
#         depth=34,
#         stem_channels=128,
#         num_stages=4,
#         out_channels=(128, 128, 128, 128),
#         strides=(1, 2, 2, 2),
#         dilations=(1, 1, 1, 1),
#         fuse_channels=(256, 128),
#         # 显式约束参数
#         enable_explicit=True,
#         explicit_image_channels=128,
#         explicit_align_channels=128,
#         explicit_out_channels=128,
#         explicit_num_samples=9),
#
#     decode_head=dict(
#         # type='FRNetHead',
#         type='FRHead',
#         in_channels = 128,
#         middle_channels = (128,128), #--------
#         channels=128,
#         num_classes=20,
#         dropout_ratio=0,
#         loss_ce=dict(
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=False,
#             class_weight=None,
#             loss_weight=1.0),
#         # loss_lovasz=dict(
#         #     type='LovaszLoss', loss_weight=1.5, reduction='none'),
#         # loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
#         conv_seg_kernel_size=1,
#         ignore_index=19),
#
#     auxiliary_head=[
#         dict(
#             type='FrustumHead',
#             channels=128,
#             num_classes=20,
#             dropout_ratio=0,
#             loss_ce=dict(
#                 type='mmdet.CrossEntropyLoss',
#                 use_sigmoid=False,
#                 class_weight=None,
#                 loss_weight=1.0),
#             loss_lovasz=dict(
#                 type='LovaszLoss', loss_weight=1.5, reduction='none'),
#             loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
#             conv_seg_kernel_size=1,
#             ignore_index=19),
#         dict(
#             type='FrustumHead',
#             channels=128,
#             num_classes=20,
#             dropout_ratio=0,
#             loss_ce=dict(
#                 type='mmdet.CrossEntropyLoss',
#                 use_sigmoid=False,
#                 class_weight=None,
#                 loss_weight=1.0),
#             loss_lovasz=dict(
#                 type='LovaszLoss', loss_weight=1.5, reduction='none'),
#             loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
#             conv_seg_kernel_size=1,
#             ignore_index=19,
#             indices=2),
#         dict(
#             type='FrustumHead',
#             channels=128,
#             num_classes=20,
#             dropout_ratio=0,
#             loss_ce=dict(
#                 type='mmdet.CrossEntropyLoss',
#                 use_sigmoid=False,
#                 class_weight=None,
#                 loss_weight=1.0),
#             loss_lovasz=dict(
#                 type='LovaszLoss', loss_weight=1.5, reduction='none'),
#             loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
#             conv_seg_kernel_size=1,
#             ignore_index=19,
#             indices=3),
#         dict(
#             type='FrustumHead',
#             channels=128,
#             num_classes=20,
#             dropout_ratio=0,
#             loss_ce=dict(
#                 type='mmdet.CrossEntropyLoss',
#                 use_sigmoid=False,
#                 class_weight=None,
#                 loss_weight=1.0),
#             loss_lovasz=dict(
#                 type='LovaszLoss', loss_weight=1.5, reduction='none'),
#             loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
#             conv_seg_kernel_size=1,
#             ignore_index=19,
#             indices=4),
#     ],
# )







# """
# Config for FRNet with Explicit Constraint Branch (ONLY).
#
# 关键改动（相对于原版 frnet-semantickitti_seg.py）：
# ============================================================
# 1. model.type:           'FRNet'           → 'FRNetExplicit'
# 2. data_preprocessor:    FrustumRangePreprocessor → ExplicitConstraintPreprocessor
# 3. backbone.type:        'FRNetBackbone'   → 'FRNetExplicitBackbone'
# 4. backbone 新增参数:     enable_explicit, explicit_* 系列参数
# 5. Pack3DDetInputs.keys: 加入 'imgs' 以传递图像数据
# 6. Pipeline 新增:         LoadCalibration, LoadImageFromFile
#
# 损失函数:
#     L = L_p + λ·L_f + λ_c·Loss_VI
#     - L_p:      point-level cross-entropy (decode_head)
#     - L_f:      frustum-level CE + Lovász + Boundary (auxiliary_head × 4)
#     - Loss_VI:  contrastive alignment loss（显式约束，自动加入，无需配置）
#
# 注意：本配置使用完整的 model dict（不依赖 _base_/models/frnet.py），
# 以避免因 _base_ 继承而导致 model.type 被覆盖回 'FRNet' 的问题。
#
# Usage:
#     python tools/train.py configs/frnet/frnet-explicit-semantickitti_seg.py
# """
#
# # ============================================================
# # 继承数据集、训练策略、默认运行时配置
# # 注意：不继承 _base_/models/frnet.py，避免 type 被覆盖
# # ============================================================
# _base_ = [
#     '../_base_/datasets/semantickitti_seg.py',
#     '../_base_/schedules/onecycle-50k.py',
#     '../_base_/default_runtime.py',
# ]
#
# # ============================================================
# # 自定义导入：确保所有自定义模块被注册
# # ============================================================
# custom_imports = dict(
#     imports=[
#         'frnet.datasets',
#         'frnet.datasets.transforms',
#         'frnet.models',
#     ],
#     allow_failed_imports=False)
#
# # ============================================================
# # 完整的模型配置（不依赖 _base_/models/frnet.py）
# # ============================================================
# model = dict(
#     # ---- Segmentor: FRNetExplicit（核心改动 1）----
#     # FRNetExplicit 继承 EncoderDecoder3D，新增：
#     #   - contrastive_loss: 对比对齐损失（Loss_VI）
#     #   - loss() 方法中自动计算 loss_contrastive
#     type='FRNetExplicit',
#
#     # ---- Data Preprocessor（核心改动 2）----
#     # ExplicitConstraintPreprocessor 继承 BaseDataPreprocessor，新增：
#     #   - 加载并预处理相机图像
#     #   - 计算 3D→2D 投影坐标（lidar2img）
#     #   - 将 images, proj_coords, has_images 存入 voxel_dict
#     data_preprocessor=dict(
#         type='ExplicitConstraintPreprocessor',
#         H=64,
#         W=512,
#         fov_up=3.0,
#         fov_down=-25.0,
#         ignore_index=19,
#         image_size=(370, 1226),      # KITTI 图像尺寸
#         mean=[123.675, 116.28, 103.53],
#         std=[58.395, 57.12, 57.375],
#         image_stride=8),
#
#     # ---- Voxel Encoder（与原版完全一致）----
#     voxel_encoder=dict(
#         type='FrustumFeatureEncoder',
#         in_channels=4,
#         feat_channels=(64, 128, 256, 256),
#         with_distance=True,
#         with_cluster_center=True,
#         norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
#         with_pre_norm=True,
#         feat_compression=16),
#
#     # ---- Backbone（核心改动 3）----
#     # FRNetExplicitBackbone 继承 BaseModule，在原版 FRNetBackbone 基础上新增：
#     #   - image_backbone: 轻量级图像特征提取网络
#     #   - explicit_branch: ExplicitConstraintBranch（偏移预测+特征校正+跨模态融合）
#     #   - explicit_merge: 将显式分支输出与原始骨干特征合并
#     #   - proj_head_voxel / proj_head_image: 对比学习投影头（φ_V / φ_I）
#     #   - forward() 中：
#     #     1. 原版 FRNet backbone 特征提取
#     #     2. 如果 has_images=True → 运行显式约束分支
#     #     3. 训练时生成 z_voxel, z_image 用于 Loss_VI
#     backbone=dict(
#         type='FRNetExplicitBackbone',
#         # ---- 原版 FRNetBackbone 参数（保持不变）----
#         in_channels=16,
#         point_in_channels=384,
#         output_shape=(64, 512),
#         depth=34,
#         stem_channels=128,
#         num_stages=4,
#         out_channels=(128, 128, 128, 128),
#         strides=(1, 2, 2, 2),
#         dilations=(1, 1, 1, 1),
#         fuse_channels=(256, 128),
#         norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
#         point_norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
#         act_cfg=dict(type='HSwish', inplace=True),
#         # ---- 显式约束分支参数（新增）----
#         enable_explicit=True,
#         explicit_image_channels=128,   # 图像 backbone 输出通道数
#         explicit_align_channels=128,   # 对齐后的特征通道数
#         explicit_out_channels=128,     # 显式分支输出通道数
#         explicit_num_samples=9,        # 可变形采样点数 K
#     ),
#
#     # ---- Decode Head（与原版完全一致）----
#     decode_head=dict(
#         type='FRHead',
#         in_channels=128,
#         middle_channels=(128, 256, 128, 64),
#         norm_cfg=dict(type='SyncBN', eps=1e-3, momentum=0.01),
#         channels=64,
#         num_classes=20,
#         dropout_ratio=0,
#         loss_ce=dict(
#             type='mmdet.CrossEntropyLoss',
#             use_sigmoid=False,
#             class_weight=None,
#             loss_weight=1.0),
#         conv_seg_kernel_size=1,
#         ignore_index=19),
#
#     # ---- Auxiliary Heads（与原版完全一致）----
#     # 4 个 FrustumHead 用于 frustum-level 辅助监督
#     auxiliary_head=[
#         dict(
#             type='FrustumHead',
#             channels=128,
#             num_classes=20,
#             dropout_ratio=0,
#             loss_ce=dict(
#                 type='mmdet.CrossEntropyLoss',
#                 use_sigmoid=False,
#                 class_weight=None,
#                 loss_weight=1.0),
#             loss_lovasz=dict(
#                 type='LovaszLoss', loss_weight=1.5, reduction='none'),
#             loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
#             conv_seg_kernel_size=1,
#             ignore_index=19),
#         dict(
#             type='FrustumHead',
#             channels=128,
#             num_classes=20,
#             dropout_ratio=0,
#             loss_ce=dict(
#                 type='mmdet.CrossEntropyLoss',
#                 use_sigmoid=False,
#                 class_weight=None,
#                 loss_weight=1.0),
#             loss_lovasz=dict(
#                 type='LovaszLoss', loss_weight=1.5, reduction='none'),
#             loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
#             conv_seg_kernel_size=1,
#             ignore_index=19,
#             indices=2),
#         dict(
#             type='FrustumHead',
#             channels=128,
#             num_classes=20,
#             dropout_ratio=0,
#             loss_ce=dict(
#                 type='mmdet.CrossEntropyLoss',
#                 use_sigmoid=False,
#                 class_weight=None,
#                 loss_weight=1.0),
#             loss_lovasz=dict(
#                 type='LovaszLoss', loss_weight=1.5, reduction='none'),
#             loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
#             conv_seg_kernel_size=1,
#             ignore_index=19,
#             indices=3),
#         dict(
#             type='FrustumHead',
#             channels=128,
#             num_classes=20,
#             dropout_ratio=0,
#             loss_ce=dict(
#                 type='mmdet.CrossEntropyLoss',
#                 use_sigmoid=False,
#                 class_weight=None,
#                 loss_weight=1.0),
#             loss_lovasz=dict(
#                 type='LovaszLoss', loss_weight=1.5, reduction='none'),
#             loss_boundary=dict(type='BoundaryLoss', loss_weight=1.0),
#             conv_seg_kernel_size=1,
#             ignore_index=19,
#             indices=4),
#     ],
# )
#
# # ============================================================
# # 训练/测试/验证目录
# # ============================================================
# work_dir = './work_dirs/frnet_explicit'
#
# # ============================================================
# # 数据 pipeline 覆盖：加入图像加载和标定读取
# # ============================================================
# # 训练 pipeline 需要加入 LoadCalibration 和 LoadImageFromFile
# # 同时 Pack3DDetInputs.keys 必须包含 'imgs'
# #
# # 注意：如果 _base_/datasets/semantickitti_seg.py 中的 train_pipeline
# # 已经包含了 LoadCalibration 和 LoadImageFromFile，则此处无需重复覆盖。
# # 但必须确保 Pack3DDetInputs 的 keys 包含 'imgs'。
# #
# # 如果你的 base dataset config 中 pipeline 没有图像相关 transform，
# # 请取消下面的注释并按需修改：
#
# # train_dataloader = dict(
# #     dataset=dict(
# #         pipeline=[
# #             dict(type='LoadPointsFromFile', coord_type='LIDAR',
# #                  load_dim=4, use_dim=4),
# #             dict(type='LoadAnnotations3D', with_bbox_3d=False,
# #                  with_label_3d=False, with_seg_3d=True,
# #                  seg_3d_dtype='np.int32', seg_offset=65536,
# #                  dataset_type='semantickitti'),
# #             dict(type='LoadCalibration'),          # ← 新增：加载标定参数
# #             dict(type='LoadImageFromFile'),         # ← 新增：加载相机图像
# #             dict(type='PointSegClassMapping'),
# #             dict(type='RandomFlip3D', sync_2d=False,
# #                  flip_ratio_bev_horizontal=0.5,
# #                  flip_ratio_bev_vertical=0.5),
# #             dict(type='GlobalRotScaleTrans',
# #                  rot_range=[-3.1415926, 3.1415926],
# #                  scale_ratio_range=[0.95, 1.05],
# #                  translation_std=[0.1, 0.1, 0.1]),
# #             dict(type='FrustumMix', H=64, W=512, fov_up=3.0,
# #                  fov_down=-25.0, num_areas=[3, 4, 5, 6],
# #                  pre_transform=pre_transform, prob=1.0),
# #             dict(type='InstanceCopy',
# #                  instance_classes=[1,2,3,4,5,6,7,11,15,17,18],
# #                  pre_transform=pre_transform, prob=1.0),
# #             dict(type='RangeInterpolation', H=64, W=2048,
# #                  fov_up=3.0, fov_down=-25.0, ignore_index=19),
# #             dict(type='Pack3DDetInputs',
# #                  keys=['points', 'pts_semantic_mask', 'imgs']),  # ← 加入 imgs
# #         ]))
