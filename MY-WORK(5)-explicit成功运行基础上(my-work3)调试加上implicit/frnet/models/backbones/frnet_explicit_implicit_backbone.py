"""
FRNet Backbone with Explicit + Implicit Constraint Branches.

Extends FRNetExplicitBackbone by adding the implicit constraint branch
based on 3D Gaussian Splatting. The implicit branch:

1. Takes the explicit-enhanced point features F_exp and point coordinates
2. Maps them to 3D Gaussian parameters via a lightweight MLP
3. Renders an implicit feature map via differentiable Gaussian splatting
4. The rendered feature map is stored for loss computation against image features

The implicit branch is ONLY active during training. At inference it is
removed, but its regularization persists in the learned backbone weights.

Architecture flow:
    Point Cloud → FRNet Backbone → F_u
    F_u + Image → Explicit Branch → F_exp
    F_exp + xyz → Implicit Branch (MLP → 3D Gaussians → Render) → F̂_img
    Loss: L = L_seg + λ_exp·L_exp + λ_imp·||F̂_img - F_img||_1
"""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from torch import Tensor

from .frnet_explicit_backbone import FRNetExplicitBackbone
from .implicit_constraint import ImplicitConstraintBranch


@MODELS.register_module()
class FRNetExplicitImplicitBackbone(FRNetExplicitBackbone):
    """FRNet Backbone with both Explicit and Implicit Constraint Branches.

    Inherits all functionality from FRNetExplicitBackbone and adds the
    implicit constraint branch based on 3D Gaussian Splatting.

    Additional Args:
        enable_implicit (bool): Whether to enable implicit constraint. Default: True.
        implicit_feat_channels (int): Feature dim for Gaussian splatting
            (must match image encoder intermediate feature dim). Default: 128.
        implicit_hidden_channels (int): MLP hidden layer dim. Default: 128.
        implicit_num_mlp_layers (int): Number of MLP hidden layers. Default: 2.
        implicit_alpha_min (float): Minimum opacity threshold. Default: 0.01.
    """

    def __init__(self,
                 in_channels: int,
                 point_in_channels: int,
                 output_shape: Sequence[int],
                 depth: int,
                 stem_channels: int = 128,
                 num_stages: int = 4,
                 out_channels: Sequence[int] = (128, 128, 128, 128),
                 strides: Sequence[int] = (1, 2, 2, 2),
                 dilations: Sequence[int] = (1, 1, 1, 1),
                 fuse_channels: Sequence[int] = (256, 128),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: ConfigType = dict(type='BN'),
                 point_norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='LeakyReLU'),
                 # Explicit constraint parameters (passed to parent)
                 image_backbone_cfg: Optional[dict] = None,
                 explicit_voxel_channels: int = 128,
                 explicit_image_channels: int = 128,
                 explicit_align_channels: int = 128,
                 explicit_out_channels: int = 128,
                 explicit_num_samples: int = 9,
                 enable_explicit: bool = True,
                 # Implicit constraint parameters (NEW)
                 enable_implicit: bool = True,
                 implicit_feat_channels: int = 128,
                 implicit_hidden_channels: int = 128,
                 implicit_num_mlp_layers: int = 2,
                 implicit_alpha_min: float = 0.01,
                 init_cfg: OptMultiConfig = None) -> None:

        # Initialize parent (FRNetExplicitBackbone with all explicit params)
        super().__init__(
            in_channels=in_channels,
            point_in_channels=point_in_channels,
            output_shape=output_shape,
            depth=depth,
            stem_channels=stem_channels,
            num_stages=num_stages,
            out_channels=out_channels,
            strides=strides,
            dilations=dilations,
            fuse_channels=fuse_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            point_norm_cfg=point_norm_cfg,
            act_cfg=act_cfg,
            image_backbone_cfg=image_backbone_cfg,
            explicit_voxel_channels=explicit_voxel_channels,
            explicit_image_channels=explicit_image_channels,
            explicit_align_channels=explicit_align_channels,
            explicit_out_channels=explicit_out_channels,
            explicit_num_samples=explicit_num_samples,
            enable_explicit=enable_explicit,
            init_cfg=init_cfg,
        )

        self.enable_implicit = enable_implicit

        if enable_implicit:
            # The point feature dimension after explicit fusion = fuse_channels[-1]
            point_feat_dim = fuse_channels[-1]

            self.implicit_branch = ImplicitConstraintBranch(
                point_feat_channels=point_feat_dim,
                image_feat_channels=implicit_feat_channels,
                hidden_channels=implicit_hidden_channels,
                num_mlp_layers=implicit_num_mlp_layers,
                alpha_min=implicit_alpha_min,
            )

    def _build_projection_matrices(self, meta, device):
        """Build view and projection matrices from calibration data.

        For projecting 3D Gaussians onto the camera image plane.

        Args:
            meta: metainfo dict with calibration parameters.
            device: target device.

        Returns:
            viewmatrix: (4, 4) world-to-camera transform.
            projmatrix: (4, 4) full projection (world-to-pixel).
        """
        if 'lidar2img' in meta:
            lidar2img = torch.tensor(
                meta['lidar2img'], device=device, dtype=torch.float32)
            if lidar2img.shape == (3, 4):
                # Pad to 4x4
                pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
                lidar2img = torch.cat([lidar2img, pad], dim=0)

            # For the GaussianCaR-style rasterizer, viewmatrix is the
            # world-to-camera (extrinsic), and projmatrix is the full
            # world-to-pixel projection.
            # Since lidar2img = cam2img @ lidar2cam, we use:
            #   viewmatrix = identity (pass 3D coords directly)
            #   projmatrix = lidar2img (handles both extrinsic + intrinsic)
            viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
            projmatrix = lidar2img

        elif 'lidar2cam' in meta and 'cam2img' in meta:
            lidar2cam = torch.tensor(
                meta['lidar2cam'], device=device, dtype=torch.float32)
            cam2img = torch.tensor(
                meta['cam2img'], device=device, dtype=torch.float32)

            if cam2img.shape == (3, 3):
                cam2img_4x4 = torch.eye(4, device=device, dtype=torch.float32)
                cam2img_4x4[:3, :3] = cam2img
                cam2img = cam2img_4x4
            elif cam2img.shape == (3, 4):
                pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
                cam2img = torch.cat([cam2img, pad], dim=0)

            viewmatrix = lidar2cam
            projmatrix = cam2img @ lidar2cam
        else:
            # Fallback: identity (no proper projection)
            viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
            projmatrix = torch.eye(4, device=device, dtype=torch.float32)

        return viewmatrix, projmatrix

    def forward(self, voxel_dict: dict) -> dict:
        """Forward pass with explicit + implicit constraint branches.

        Calls the parent's forward (which includes explicit constraint),
        then if training and implicit is enabled, runs the implicit branch.
        """
        # ---- Run parent forward (FRNet backbone + explicit constraint) ----
        voxel_dict = super().forward(voxel_dict)

        # ---- Implicit Constraint Branch (training only) ----
        if (self.training and self.enable_implicit
                and voxel_dict.get('has_images', False)):

            # Get the enhanced point features after explicit fusion
            point_feats = voxel_dict['point_feats_backbone'][0]  # (N, C_fuse)

            # Get point 3D coordinates
            # In FRNet, after frustum_region_group, voxel_dict['voxels'] = (N, 4+)
            # where columns are [x, y, z, intensity, ...]
            voxels = voxel_dict.get('voxels', None)
            if voxels is not None and voxels.dim() == 2 and voxels.shape[-1] >= 3:
                xyz = voxels[:, :3].clone()
            else:
                # Fallback: cannot run implicit branch without coordinates
                return voxel_dict

            pts_coors = voxel_dict['coors']  # (N, 3) [batch_idx, y, x]

            # Get image feature map from explicit branch
            # The image backbone was already run in the parent's forward
            # We need to re-extract image features (or we can store them)
            # Re-run image backbone (it's lightweight and results are needed)
            images = voxel_dict['images']  # (B, 3, H_img, W_img)
            image_feat_map = self.image_backbone(images)  # (B, C_img, H', W')

            # Store image_feat_map for loss computation
            voxel_dict['image_feat_map'] = image_feat_map

            _, C_img, H_feat, W_feat = image_feat_map.shape
            batch_size = int(pts_coors[:, 0].max().item()) + 1

            rendered_feats_list = []
            image_feats_list = []

            for b in range(batch_size):
                batch_mask = (pts_coors[:, 0] == b)
                xyz_b = xyz[batch_mask]            # (N_b, 3)
                feats_b = point_feats[batch_mask]  # (N_b, C)

                if xyz_b.shape[0] == 0:
                    continue

                # ---- Build projection matrices ----
                viewmatrix_b, projmatrix_b = self._get_proj_matrices_for_batch(
                    voxel_dict, b, xyz_b.device)

                # ---- Compute tanfovx / tanfovy from scaled lidar2img ----
                lidar2img_list = voxel_dict.get('lidar2img_matrices', None)
                if lidar2img_list is not None and b < len(lidar2img_list):
                    P = lidar2img_list[b].to(xyz_b.device).float()
                    # P[0,0] = fx_scaled (fx / image_stride)
                    # P[1,1] = fy_scaled (fy / image_stride)
                    fx_s = P[0, 0].item()
                    fy_s = P[1, 1].item()
                    tanfovx = W_feat / (2.0 * fx_s) if fx_s > 0 else 1.0
                    tanfovy = H_feat / (2.0 * fy_s) if fy_s > 0 else 1.0
                else:
                    tanfovx = 1.0
                    tanfovy = 1.0

                # ---- Compute campos = camera position in world space ----
                # campos is the translation part of the inverse of viewmatrix
                try:
                    campos = viewmatrix_b.inverse()[:3, 3]
                except Exception:
                    campos = torch.zeros(3, device=xyz_b.device,
                                        dtype=torch.float32)

                # ---- Render implicit feature map ----
                rendered_feat = self.implicit_branch(
                    xyz=xyz_b,
                    point_feats=feats_b,
                    viewmatrix=viewmatrix_b,
                    projmatrix=projmatrix_b,
                    image_height=H_feat,
                    image_width=W_feat,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    campos=campos,
                )  # (C_feat, H_feat, W_feat)

                rendered_feats_list.append(rendered_feat)
                image_feats_list.append(image_feat_map[b])

            if len(rendered_feats_list) > 0:
                rendered_feats = torch.stack(rendered_feats_list, dim=0)
                image_feats = torch.stack(image_feats_list, dim=0)
                voxel_dict['rendered_implicit_feat'] = rendered_feats
                voxel_dict['image_feat_for_implicit'] = image_feats

        return voxel_dict

    def _get_proj_matrices_for_batch(self, voxel_dict, batch_idx, device):
        """Get view/projection matrices for a specific batch sample.

        Tries to use calibration info stored in voxel_dict by the
        data preprocessor. Falls back to simple perspective projection.

        Args:
            voxel_dict: The voxel dictionary.
            batch_idx: Batch index.
            device: Target device.

        Returns:
            viewmatrix: (4, 4) tensor.
            projmatrix: (4, 4) tensor.
        """
        # Try to get from stored calibration
        calib_list = voxel_dict.get('calib_matrices', None)
        if calib_list is not None and batch_idx < len(calib_list):
            calib = calib_list[batch_idx]
            viewmatrix = calib['viewmatrix'].to(device).float().contiguous()
            projmatrix = calib['projmatrix'].to(device).float().contiguous()
            return viewmatrix, projmatrix

        # Try to construct from lidar2img stored by preprocessor
        lidar2img_list = voxel_dict.get('lidar2img_matrices', None)
        if lidar2img_list is not None and batch_idx < len(lidar2img_list):
            lidar2img = lidar2img_list[batch_idx].to(device).float()
            if lidar2img.shape == (3, 4):
                pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
                lidar2img = torch.cat([lidar2img, pad], dim=0)
            viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
            projmatrix = lidar2img.contiguous()
            return viewmatrix, projmatrix

        # Fallback: construct a simple perspective projection
        # Use image dimensions to approximate
        images = voxel_dict.get('images', None)
        if images is not None:
            _, _, H_img, W_img = images.shape
            fx = W_img / 2.0
            fy = H_img / 2.0
            cx = W_img / 2.0
            cy = H_img / 2.0
        else:
            fx, fy, cx, cy = 500.0, 500.0, 613.0, 185.0

        # Simple perspective: project (x,y,z) -> (fx*x/z+cx, fy*y/z+cy)
        # This is a fallback and won't be geometrically accurate
        viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
        projmatrix = torch.tensor([
            [fx,  0,  cx, 0],
            [ 0, fy,  cy, 0],
            [ 0,  0,   1, 0],
            [ 0,  0,   1, 0],
        ], device=device, dtype=torch.float32)

        return viewmatrix.contiguous(), projmatrix.contiguous()

# """
# FRNet Backbone with Explicit + Implicit Constraint Branches.
#
# Extends FRNetExplicitBackbone by adding the implicit constraint branch
# based on 3D Gaussian Splatting. The implicit branch:
#
# 1. Takes the explicit-enhanced point features F_exp and point coordinates
# 2. Maps them to 3D Gaussian parameters via a lightweight MLP
# 3. Renders an implicit feature map via differentiable Gaussian splatting
# 4. The rendered feature map is stored for loss computation against image features
#
# The implicit branch is ONLY active during training. At inference it is
# removed, but its regularization persists in the learned backbone weights.
#
# Architecture flow:
#     Point Cloud → FRNet Backbone → F_u
#     F_u + Image → Explicit Branch → F_exp
#     F_exp + xyz → Implicit Branch (MLP → 3D Gaussians → Render) → F̂_img
#     Loss: L = L_seg + λ_exp·L_exp + λ_imp·||F̂_img - F_img||_1
# """
#
# from typing import Optional, Sequence
#
# import numpy as np
# import torch
# import torch.nn as nn
# from mmdet3d.registry import MODELS
# from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
# from mmengine.model import BaseModule
# from torch import Tensor
#
# from .frnet_explicit_backbone import FRNetExplicitBackbone
# from .implicit_constraint import ImplicitConstraintBranch
#
#
# @MODELS.register_module()
# class FRNetExplicitImplicitBackbone(FRNetExplicitBackbone):
#     """FRNet Backbone with both Explicit and Implicit Constraint Branches.
#
#     Inherits all functionality from FRNetExplicitBackbone and adds the
#     implicit constraint branch based on 3D Gaussian Splatting.
#
#     Additional Args:
#         enable_implicit (bool): Whether to enable implicit constraint. Default: True.
#         implicit_feat_channels (int): Feature dim for Gaussian splatting
#             (must match image encoder intermediate feature dim). Default: 128.
#         implicit_hidden_channels (int): MLP hidden layer dim. Default: 128.
#         implicit_num_mlp_layers (int): Number of MLP hidden layers. Default: 2.
#         implicit_alpha_min (float): Minimum opacity threshold. Default: 0.01.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  point_in_channels: int,
#                  output_shape: Sequence[int],
#                  depth: int,
#                  stem_channels: int = 128,
#                  num_stages: int = 4,
#                  out_channels: Sequence[int] = (128, 128, 128, 128),
#                  strides: Sequence[int] = (1, 2, 2, 2),
#                  dilations: Sequence[int] = (1, 1, 1, 1),
#                  fuse_channels: Sequence[int] = (256, 128),
#                  conv_cfg: OptConfigType = None,
#                  norm_cfg: ConfigType = dict(type='BN'),
#                  point_norm_cfg: ConfigType = dict(type='BN1d'),
#                  act_cfg: ConfigType = dict(type='LeakyReLU'),
#                  # Explicit constraint parameters (passed to parent)
#                  image_backbone_cfg: Optional[dict] = None,
#                  explicit_voxel_channels: int = 128,
#                  explicit_image_channels: int = 128,
#                  explicit_align_channels: int = 128,
#                  explicit_out_channels: int = 128,
#                  explicit_num_samples: int = 9,
#                  enable_explicit: bool = True,
#                  # Implicit constraint parameters (NEW)
#                  enable_implicit: bool = True,
#                  implicit_feat_channels: int = 128,
#                  implicit_hidden_channels: int = 128,
#                  implicit_num_mlp_layers: int = 2,
#                  implicit_alpha_min: float = 0.01,
#                  init_cfg: OptMultiConfig = None) -> None:
#
#         # Initialize parent (FRNetExplicitBackbone with all explicit params)
#         super().__init__(
#             in_channels=in_channels,
#             point_in_channels=point_in_channels,
#             output_shape=output_shape,
#             depth=depth,
#             stem_channels=stem_channels,
#             num_stages=num_stages,
#             out_channels=out_channels,
#             strides=strides,
#             dilations=dilations,
#             fuse_channels=fuse_channels,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             point_norm_cfg=point_norm_cfg,
#             act_cfg=act_cfg,
#             image_backbone_cfg=image_backbone_cfg,
#             explicit_voxel_channels=explicit_voxel_channels,
#             explicit_image_channels=explicit_image_channels,
#             explicit_align_channels=explicit_align_channels,
#             explicit_out_channels=explicit_out_channels,
#             explicit_num_samples=explicit_num_samples,
#             enable_explicit=enable_explicit,
#             init_cfg=init_cfg,
#         )
#
#         self.enable_implicit = enable_implicit
#
#         if enable_implicit:
#             # The point feature dimension after explicit fusion = fuse_channels[-1]
#             point_feat_dim = fuse_channels[-1]
#
#             self.implicit_branch = ImplicitConstraintBranch(
#                 point_feat_channels=point_feat_dim,
#                 image_feat_channels=implicit_feat_channels,
#                 hidden_channels=implicit_hidden_channels,
#                 num_mlp_layers=implicit_num_mlp_layers,
#                 alpha_min=implicit_alpha_min,
#             )
#
#     def _build_projection_matrices(self, meta, device):
#         """Build view and projection matrices from calibration data.
#
#         For projecting 3D Gaussians onto the camera image plane.
#
#         Args:
#             meta: metainfo dict with calibration parameters.
#             device: target device.
#
#         Returns:
#             viewmatrix: (4, 4) world-to-camera transform.
#             projmatrix: (4, 4) full projection (world-to-pixel).
#         """
#         if 'lidar2img' in meta:
#             lidar2img = torch.tensor(
#                 meta['lidar2img'], device=device, dtype=torch.float32)
#             if lidar2img.shape == (3, 4):
#                 # Pad to 4x4
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 lidar2img = torch.cat([lidar2img, pad], dim=0)
#
#             # For the GaussianCaR-style rasterizer, viewmatrix is the
#             # world-to-camera (extrinsic), and projmatrix is the full
#             # world-to-pixel projection.
#             # Since lidar2img = cam2img @ lidar2cam, we use:
#             #   viewmatrix = identity (pass 3D coords directly)
#             #   projmatrix = lidar2img (handles both extrinsic + intrinsic)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = lidar2img
#
#         elif 'lidar2cam' in meta and 'cam2img' in meta:
#             lidar2cam = torch.tensor(
#                 meta['lidar2cam'], device=device, dtype=torch.float32)
#             cam2img = torch.tensor(
#                 meta['cam2img'], device=device, dtype=torch.float32)
#
#             if cam2img.shape == (3, 3):
#                 cam2img_4x4 = torch.eye(4, device=device, dtype=torch.float32)
#                 cam2img_4x4[:3, :3] = cam2img
#                 cam2img = cam2img_4x4
#             elif cam2img.shape == (3, 4):
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 cam2img = torch.cat([cam2img, pad], dim=0)
#
#             viewmatrix = lidar2cam
#             projmatrix = cam2img @ lidar2cam
#         else:
#             # Fallback: identity (no proper projection)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = torch.eye(4, device=device, dtype=torch.float32)
#
#         return viewmatrix, projmatrix
#
#     def forward(self, voxel_dict: dict) -> dict:
#         """Forward pass with explicit + implicit constraint branches.
#
#         Calls the parent's forward (which includes explicit constraint),
#         then if training and implicit is enabled, runs the implicit branch.
#         """
#         # ---- Run parent forward (FRNet backbone + explicit constraint) ----
#         voxel_dict = super().forward(voxel_dict)
#
#         # ---- Implicit Constraint Branch (training only) ----
#         if (self.training and self.enable_implicit
#                 and voxel_dict.get('has_images', False)):
#
#             # Get the enhanced point features after explicit fusion
#             point_feats = voxel_dict['point_feats_backbone'][0]  # (N, C_fuse)
#
#             # Get point 3D coordinates
#             # In FRNet, after frustum_region_group, voxel_dict['voxels'] = (N, 4+)
#             # where columns are [x, y, z, intensity, ...]
#             voxels = voxel_dict.get('voxels', None)
#             if voxels is not None and voxels.dim() == 2 and voxels.shape[-1] >= 3:
#                 xyz = voxels[:, :3].clone()
#             else:
#                 # Fallback: cannot run implicit branch without coordinates
#                 return voxel_dict
#
#             pts_coors = voxel_dict['coors']  # (N, 3) [batch_idx, y, x]
#
#             # Get image feature map from explicit branch
#             # The image backbone was already run in the parent's forward
#             # We need to re-extract image features (or we can store them)
#             # Re-run image backbone (it's lightweight and results are needed)
#             images = voxel_dict['images']  # (B, 3, H_img, W_img)
#             image_feat_map = self.image_backbone(images)  # (B, C_img, H', W')
#
#             # Store image_feat_map for loss computation
#             voxel_dict['image_feat_map'] = image_feat_map
#
#             _, C_img, H_feat, W_feat = image_feat_map.shape
#             batch_size = int(pts_coors[:, 0].max().item()) + 1
#
#             # Get calibration info from data_samples (stored in voxel_dict)
#             # We need to retrieve projection matrices per sample
#             # These are passed through the data_samples metainfo
#
#             rendered_feats_list = []
#             image_feats_list = []
#
#             for b in range(batch_size):"""
# FRNet Backbone with Explicit + Implicit Constraint Branches.
#
# Extends FRNetExplicitBackbone by adding the implicit constraint branch
# based on 3D Gaussian Splatting. The implicit branch:
#
# 1. Takes the explicit-enhanced point features F_exp and point coordinates
# 2. Maps them to 3D Gaussian parameters via a lightweight MLP
# 3. Renders an implicit feature map via differentiable Gaussian splatting
# 4. The rendered feature map is stored for loss computation against image features
#
# The implicit branch is ONLY active during training. At inference it is
# removed, but its regularization persists in the learned backbone weights.
#
# Architecture flow:
#     Point Cloud → FRNet Backbone → F_u
#     F_u + Image → Explicit Branch → F_exp
#     F_exp + xyz → Implicit Branch (MLP → 3D Gaussians → Render) → F̂_img
#     Loss: L = L_seg + λ_exp·L_exp + λ_imp·||F̂_img - F_img||_1
# """
#
# from typing import Optional, Sequence
#
# import numpy as np
# import torch
# import torch.nn as nn
# from mmdet3d.registry import MODELS
# from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
# from mmengine.model import BaseModule
# from torch import Tensor
#
# from .frnet_explicit_backbone import FRNetExplicitBackbone
# from .implicit_constraint import ImplicitConstraintBranch
#
#
# @MODELS.register_module()
# class FRNetExplicitImplicitBackbone(FRNetExplicitBackbone):
#     """FRNet Backbone with both Explicit and Implicit Constraint Branches.
#
#     Inherits all functionality from FRNetExplicitBackbone and adds the
#     implicit constraint branch based on 3D Gaussian Splatting.
#
#     Additional Args:
#         enable_implicit (bool): Whether to enable implicit constraint. Default: True.
#         implicit_feat_channels (int): Feature dim for Gaussian splatting
#             (must match image encoder intermediate feature dim). Default: 128.
#         implicit_hidden_channels (int): MLP hidden layer dim. Default: 128.
#         implicit_num_mlp_layers (int): Number of MLP hidden layers. Default: 2.
#         implicit_alpha_min (float): Minimum opacity threshold. Default: 0.01.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  point_in_channels: int,
#                  output_shape: Sequence[int],
#                  depth: int,
#                  stem_channels: int = 128,
#                  num_stages: int = 4,
#                  out_channels: Sequence[int] = (128, 128, 128, 128),
#                  strides: Sequence[int] = (1, 2, 2, 2),
#                  dilations: Sequence[int] = (1, 1, 1, 1),
#                  fuse_channels: Sequence[int] = (256, 128),
#                  conv_cfg: OptConfigType = None,
#                  norm_cfg: ConfigType = dict(type='BN'),
#                  point_norm_cfg: ConfigType = dict(type='BN1d'),
#                  act_cfg: ConfigType = dict(type='LeakyReLU'),
#                  # Explicit constraint parameters (passed to parent)
#                  image_backbone_cfg: Optional[dict] = None,
#                  explicit_voxel_channels: int = 128,
#                  explicit_image_channels: int = 128,
#                  explicit_align_channels: int = 128,
#                  explicit_out_channels: int = 128,
#                  explicit_num_samples: int = 9,
#                  enable_explicit: bool = True,
#                  # Implicit constraint parameters (NEW)
#                  enable_implicit: bool = True,
#                  implicit_feat_channels: int = 128,
#                  implicit_hidden_channels: int = 128,
#                  implicit_num_mlp_layers: int = 2,
#                  implicit_alpha_min: float = 0.01,
#                  init_cfg: OptMultiConfig = None) -> None:
#
#         # Initialize parent (FRNetExplicitBackbone with all explicit params)
#         super().__init__(
#             in_channels=in_channels,
#             point_in_channels=point_in_channels,
#             output_shape=output_shape,
#             depth=depth,
#             stem_channels=stem_channels,
#             num_stages=num_stages,
#             out_channels=out_channels,
#             strides=strides,
#             dilations=dilations,
#             fuse_channels=fuse_channels,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             point_norm_cfg=point_norm_cfg,
#             act_cfg=act_cfg,
#             image_backbone_cfg=image_backbone_cfg,
#             explicit_voxel_channels=explicit_voxel_channels,
#             explicit_image_channels=explicit_image_channels,
#             explicit_align_channels=explicit_align_channels,
#             explicit_out_channels=explicit_out_channels,
#             explicit_num_samples=explicit_num_samples,
#             enable_explicit=enable_explicit,
#             init_cfg=init_cfg,
#         )
#
#         self.enable_implicit = enable_implicit
#
#         if enable_implicit:
#             # The point feature dimension after explicit fusion = fuse_channels[-1]
#             point_feat_dim = fuse_channels[-1]
#
#             self.implicit_branch = ImplicitConstraintBranch(
#                 point_feat_channels=point_feat_dim,
#                 image_feat_channels=implicit_feat_channels,
#                 hidden_channels=implicit_hidden_channels,
#                 num_mlp_layers=implicit_num_mlp_layers,
#                 alpha_min=implicit_alpha_min,
#             )
#
#     def _build_projection_matrices(self, meta, device):
#         """Build view and projection matrices from calibration data.
#
#         For projecting 3D Gaussians onto the camera image plane.
#
#         Args:
#             meta: metainfo dict with calibration parameters.
#             device: target device.
#
#         Returns:
#             viewmatrix: (4, 4) world-to-camera transform.
#             projmatrix: (4, 4) full projection (world-to-pixel).
#         """
#         if 'lidar2img' in meta:
#             lidar2img = torch.tensor(
#                 meta['lidar2img'], device=device, dtype=torch.float32)
#             if lidar2img.shape == (3, 4):
#                 # Pad to 4x4
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 lidar2img = torch.cat([lidar2img, pad], dim=0)
#
#             # For the GaussianCaR-style rasterizer, viewmatrix is the
#             # world-to-camera (extrinsic), and projmatrix is the full
#             # world-to-pixel projection.
#             # Since lidar2img = cam2img @ lidar2cam, we use:
#             #   viewmatrix = identity (pass 3D coords directly)
#             #   projmatrix = lidar2img (handles both extrinsic + intrinsic)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = lidar2img
#
#         elif 'lidar2cam' in meta and 'cam2img' in meta:
#             lidar2cam = torch.tensor(
#                 meta['lidar2cam'], device=device, dtype=torch.float32)
#             cam2img = torch.tensor(
#                 meta['cam2img'], device=device, dtype=torch.float32)
#
#             if cam2img.shape == (3, 3):
#                 cam2img_4x4 = torch.eye(4, device=device, dtype=torch.float32)
#                 cam2img_4x4[:3, :3] = cam2img
#                 cam2img = cam2img_4x4
#             elif cam2img.shape == (3, 4):
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 cam2img = torch.cat([cam2img, pad], dim=0)
#
#             viewmatrix = lidar2cam
#             projmatrix = cam2img @ lidar2cam
#         else:
#             # Fallback: identity (no proper projection)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = torch.eye(4, device=device, dtype=torch.float32)
#
#         return viewmatrix, projmatrix
#
#     def forward(self, voxel_dict: dict) -> dict:
#         """Forward pass with explicit + implicit constraint branches.
#
#         Calls the parent's forward (which includes explicit constraint),
#         then if training and implicit is enabled, runs the implicit branch.
#         """
#         # ---- Run parent forward (FRNet backbone + explicit constraint) ----
#         voxel_dict = super().forward(voxel_dict)
#
#         # ---- Implicit Constraint Branch (training only) ----
#         if (self.training and self.enable_implicit
#                 and voxel_dict.get('has_images', False)):
#
#             # Get the enhanced point features after explicit fusion
#             point_feats = voxel_dict['point_feats_backbone'][0]  # (N, C_fuse)
#
#             # Get point 3D coordinates
#             # In FRNet, after frustum_region_group, voxel_dict['voxels'] = (N, 4+)
#             # where columns are [x, y, z, intensity, ...]
#             voxels = voxel_dict.get('voxels', None)
#             if voxels is not None and voxels.dim() == 2 and voxels.shape[-1] >= 3:
#                 xyz = voxels[:, :3].clone()
#             else:
#                 # Fallback: cannot run implicit branch without coordinates
#                 return voxel_dict
#
#             pts_coors = voxel_dict['coors']  # (N, 3) [batch_idx, y, x]
#
#             # Get image feature map from explicit branch
#             # The image backbone was already run in the parent's forward
#             # We need to re-extract image features (or we can store them)
#             # Re-run image backbone (it's lightweight and results are needed)
#             images = voxel_dict['images']  # (B, 3, H_img, W_img)
#             image_feat_map = self.image_backbone(images)  # (B, C_img, H', W')
#
#             # Store image_feat_map for loss computation
#             voxel_dict['image_feat_map'] = image_feat_map
#
#             _, C_img, H_feat, W_feat = image_feat_map.shape
#             batch_size = int(pts_coors[:, 0].max().item()) + 1
#
#             rendered_feats_list = []
#             image_feats_list = []
#
#             for b in range(batch_size):
#                 batch_mask = (pts_coors[:, 0] == b)
#                 xyz_b = xyz[batch_mask]            # (N_b, 3)
#                 feats_b = point_feats[batch_mask]  # (N_b, C)
#
#                 if xyz_b.shape[0] == 0:
#                     continue
#
#                 # ---- Build projection matrices ----
#                 viewmatrix_b, projmatrix_b = self._get_proj_matrices_for_batch(
#                     voxel_dict, b, xyz_b.device)
#
#                 # ---- Compute tanfovx / tanfovy from scaled lidar2img ----
#                 lidar2img_list = voxel_dict.get('lidar2img_matrices', None)
#                 if lidar2img_list is not None and b < len(lidar2img_list):
#                     P = lidar2img_list[b].to(xyz_b.device).float()
#                     # P[0,0] = fx_scaled (fx / image_stride)
#                     # P[1,1] = fy_scaled (fy / image_stride)
#                     fx_s = P[0, 0].item()
#                     fy_s = P[1, 1].item()
#                     tanfovx = W_feat / (2.0 * fx_s) if fx_s > 0 else 1.0
#                     tanfovy = H_feat / (2.0 * fy_s) if fy_s > 0 else 1.0
#                 else:
#                     tanfovx = 1.0
#                     tanfovy = 1.0
#
#                 # ---- Compute campos = camera position in world space ----
#                 # campos is the translation part of the inverse of viewmatrix
#                 try:
#                     campos = viewmatrix_b.inverse()[:3, 3]
#                 except Exception:
#                     campos = torch.zeros(3, device=xyz_b.device,
#                                         dtype=torch.float32)
#
#                 # ---- Render implicit feature map ----
#                 rendered_feat = self.implicit_branch(
#                     xyz=xyz_b,
#                     point_feats=feats_b,
#                     viewmatrix=viewmatrix_b,
#                     projmatrix=projmatrix_b,
#                     image_height=H_feat,
#                     image_width=W_feat,
#                     tanfovx=tanfovx,
#                     tanfovy=tanfovy,
#                     campos=campos,
#                 )  # (C_feat, H_feat, W_feat)
#
#                 rendered_feats_list.append(rendered_feat)
#                 image_feats_list.append(image_feat_map[b])
#
#             if len(rendered_feats_list) > 0:
#                 rendered_feats = torch.stack(rendered_feats_list, dim=0)
#                 image_feats = torch.stack(image_feats_list, dim=0)
#                 voxel_dict['rendered_implicit_feat'] = rendered_feats
#                 voxel_dict['image_feat_for_implicit'] = image_feats
#
#         return voxel_dict
#
#     def _get_proj_matrices_for_batch(self, voxel_dict, batch_idx, device):
#         """Get view/projection matrices for a specific batch sample.
#
#         Tries to use calibration info stored in voxel_dict by the
#         data preprocessor. Falls back to simple perspective projection.
#
#         Args:
#             voxel_dict: The voxel dictionary.
#             batch_idx: Batch index.
#             device: Target device.
#
#         Returns:
#             viewmatrix: (4, 4) tensor.
#             projmatrix: (4, 4) tensor.
#         """
#         # Try to get from stored calibration
#         calib_list = voxel_dict.get('calib_matrices', None)
#         if calib_list is not None and batch_idx < len(calib_list):
#             calib = calib_list[batch_idx]
#             viewmatrix = calib['viewmatrix'].to(device).float().contiguous()
#             projmatrix = calib['projmatrix'].to(device).float().contiguous()
#             return viewmatrix, projmatrix
#
#         # Try to construct from lidar2img stored by preprocessor
#         lidar2img_list = voxel_dict.get('lidar2img_matrices', None)
#         if lidar2img_list is not None and batch_idx < len(lidar2img_list):
#             lidar2img = lidar2img_list[batch_idx].to(device).float()
#             if lidar2img.shape == (3, 4):
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 lidar2img = torch.cat([lidar2img, pad], dim=0)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = lidar2img.contiguous()
#             return viewmatrix, projmatrix
#
#         # Fallback: construct a simple perspective projection
#         # Use image dimensions to approximate
#         images = voxel_dict.get('images', None)
#         if images is not None:
#             _, _, H_img, W_img = images.shape
#             fx = W_img / 2.0"""
# FRNet Backbone with Explicit + Implicit Constraint Branches.
#
# Extends FRNetExplicitBackbone by adding the implicit constraint branch
# based on 3D Gaussian Splatting. The implicit branch:
#
# 1. Takes the explicit-enhanced point features F_exp and point coordinates
# 2. Maps them to 3D Gaussian parameters via a lightweight MLP
# 3. Renders an implicit feature map via differentiable Gaussian splatting
# 4. The rendered feature map is stored for loss computation against image features
#
# The implicit branch is ONLY active during training. At inference it is
# removed, but its regularization persists in the learned backbone weights.
#
# Architecture flow:
#     Point Cloud → FRNet Backbone → F_u
#     F_u + Image → Explicit Branch → F_exp
#     F_exp + xyz → Implicit Branch (MLP → 3D Gaussians → Render) → F̂_img
#     Loss: L = L_seg + λ_exp·L_exp + λ_imp·||F̂_img - F_img||_1
# """
#
# from typing import Optional, Sequence
#
# import numpy as np
# import torch
# import torch.nn as nn
# from mmdet3d.registry import MODELS
# from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
# from mmengine.model import BaseModule
# from torch import Tensor
#
# from .frnet_explicit_backbone import FRNetExplicitBackbone
# from .implicit_constraint import ImplicitConstraintBranch
#
#
# @MODELS.register_module()
# class FRNetExplicitImplicitBackbone(FRNetExplicitBackbone):
#     """FRNet Backbone with both Explicit and Implicit Constraint Branches.
#
#     Inherits all functionality from FRNetExplicitBackbone and adds the
#     implicit constraint branch based on 3D Gaussian Splatting.
#
#     Additional Args:
#         enable_implicit (bool): Whether to enable implicit constraint. Default: True.
#         implicit_feat_channels (int): Feature dim for Gaussian splatting
#             (must match image encoder intermediate feature dim). Default: 128.
#         implicit_hidden_channels (int): MLP hidden layer dim. Default: 128.
#         implicit_num_mlp_layers (int): Number of MLP hidden layers. Default: 2.
#         implicit_alpha_min (float): Minimum opacity threshold. Default: 0.01.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  point_in_channels: int,
#                  output_shape: Sequence[int],
#                  depth: int,
#                  stem_channels: int = 128,
#                  num_stages: int = 4,
#                  out_channels: Sequence[int] = (128, 128, 128, 128),
#                  strides: Sequence[int] = (1, 2, 2, 2),
#                  dilations: Sequence[int] = (1, 1, 1, 1),
#                  fuse_channels: Sequence[int] = (256, 128),
#                  conv_cfg: OptConfigType = None,
#                  norm_cfg: ConfigType = dict(type='BN'),
#                  point_norm_cfg: ConfigType = dict(type='BN1d'),
#                  act_cfg: ConfigType = dict(type='LeakyReLU'),
#                  # Explicit constraint parameters (passed to parent)
#                  image_backbone_cfg: Optional[dict] = None,
#                  explicit_voxel_channels: int = 128,
#                  explicit_image_channels: int = 128,
#                  explicit_align_channels: int = 128,
#                  explicit_out_channels: int = 128,
#                  explicit_num_samples: int = 9,
#                  enable_explicit: bool = True,
#                  # Implicit constraint parameters (NEW)
#                  enable_implicit: bool = True,
#                  implicit_feat_channels: int = 128,
#                  implicit_hidden_channels: int = 128,
#                  implicit_num_mlp_layers: int = 2,
#                  implicit_alpha_min: float = 0.01,
#                  init_cfg: OptMultiConfig = None) -> None:
#
#         # Initialize parent (FRNetExplicitBackbone with all explicit params)
#         super().__init__(
#             in_channels=in_channels,
#             point_in_channels=point_in_channels,
#             output_shape=output_shape,
#             depth=depth,
#             stem_channels=stem_channels,
#             num_stages=num_stages,
#             out_channels=out_channels,
#             strides=strides,
#             dilations=dilations,
#             fuse_channels=fuse_channels,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             point_norm_cfg=point_norm_cfg,
#             act_cfg=act_cfg,
#             image_backbone_cfg=image_backbone_cfg,
#             explicit_voxel_channels=explicit_voxel_channels,
#             explicit_image_channels=explicit_image_channels,
#             explicit_align_channels=explicit_align_channels,
#             explicit_out_channels=explicit_out_channels,
#             explicit_num_samples=explicit_num_samples,
#             enable_explicit=enable_explicit,
#             init_cfg=init_cfg,
#         )
#
#         self.enable_implicit = enable_implicit
#
#         if enable_implicit:
#             # The point feature dimension after explicit fusion = fuse_channels[-1]
#             point_feat_dim = fuse_channels[-1]
#
#             self.implicit_branch = ImplicitConstraintBranch(
#                 point_feat_channels=point_feat_dim,
#                 image_feat_channels=implicit_feat_channels,
#                 hidden_channels=implicit_hidden_channels,
#                 num_mlp_layers=implicit_num_mlp_layers,
#                 alpha_min=implicit_alpha_min,
#             )
#
#     def _build_projection_matrices(self, meta, device):
#         """Build view and projection matrices from calibration data.
#
#         For projecting 3D Gaussians onto the camera image plane.
#
#         Args:
#             meta: metainfo dict with calibration parameters.
#             device: target device.
#
#         Returns:
#             viewmatrix: (4, 4) world-to-camera transform.
#             projmatrix: (4, 4) full projection (world-to-pixel).
#         """
#         if 'lidar2img' in meta:
#             lidar2img = torch.tensor(
#                 meta['lidar2img'], device=device, dtype=torch.float32)
#             if lidar2img.shape == (3, 4):
#                 # Pad to 4x4
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 lidar2img = torch.cat([lidar2img, pad], dim=0)
#
#             # For the GaussianCaR-style rasterizer, viewmatrix is the
#             # world-to-camera (extrinsic), and projmatrix is the full
#             # world-to-pixel projection.
#             # Since lidar2img = cam2img @ lidar2cam, we use:
#             #   viewmatrix = identity (pass 3D coords directly)
#             #   projmatrix = lidar2img (handles both extrinsic + intrinsic)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = lidar2img
#
#         elif 'lidar2cam' in meta and 'cam2img' in meta:
#             lidar2cam = torch.tensor(
#                 meta['lidar2cam'], device=device, dtype=torch.float32)
#             cam2img = torch.tensor(
#                 meta['cam2img'], device=device, dtype=torch.float32)
#
#             if cam2img.shape == (3, 3):
#                 cam2img_4x4 = torch.eye(4, device=device, dtype=torch.float32)
#                 cam2img_4x4[:3, :3] = cam2img
#                 cam2img = cam2img_4x4
#             elif cam2img.shape == (3, 4):
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 cam2img = torch.cat([cam2img, pad], dim=0)
#
#             viewmatrix = lidar2cam
#             projmatrix = cam2img @ lidar2cam
#         else:
#             # Fallback: identity (no proper projection)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = torch.eye(4, device=device, dtype=torch.float32)
#
#         return viewmatrix, projmatrix
#
#     def forward(self, voxel_dict: dict) -> dict:
#         """Forward pass with explicit + implicit constraint branches.
#
#         Calls the parent's forward (which includes explicit constraint),
#         then if training and implicit is enabled, runs the implicit branch.
#         """
#         # ---- Run parent forward (FRNet backbone + explicit constraint) ----
#         voxel_dict = super().forward(voxel_dict)
#
#         # ---- Implicit Constraint Branch (training only) ----
#         if (self.training and self.enable_implicit
#                 and voxel_dict.get('has_images', False)):
#
#             # Get the enhanced point features after explicit fusion
#             point_feats = voxel_dict['point_feats_backbone'][0]  # (N, C_fuse)
#
#             # Get point 3D coordinates
#             # In FRNet, after frustum_region_group, voxel_dict['voxels'] = (N, 4+)
#             # where columns are [x, y, z, intensity, ...]
#             voxels = voxel_dict.get('voxels', None)
#             if voxels is not None and voxels.dim() == 2 and voxels.shape[-1] >= 3:
#                 xyz = voxels[:, :3].clone()
#             else:
#                 # Fallback: cannot run implicit branch without coordinates
#                 return voxel_dict
#
#             pts_coors = voxel_dict['coors']  # (N, 3) [batch_idx, y, x]
#
#             # Get image feature map from explicit branch
#             # The image backbone was already run in the parent's forward
#             # We need to re-extract image features (or we can store them)
#             # Re-run image backbone (it's lightweight and results are needed)
#             images = voxel_dict['images']  # (B, 3, H_img, W_img)
#             image_feat_map = self.image_backbone(images)  # (B, C_img, H', W')
#
#             # Store image_feat_map for loss computation
#             voxel_dict['image_feat_map'] = image_feat_map
#
#             _, C_img, H_feat, W_feat = image_feat_map.shape
#             batch_size = int(pts_coors[:, 0].max().item()) + 1
#
#             rendered_feats_list = []
#             image_feats_list = []
#
#             for b in range(batch_size):
#                 batch_mask = (pts_coors[:, 0] == b)
#                 xyz_b = xyz[batch_mask]            # (N_b, 3)
#                 feats_b = point_feats[batch_mask]  # (N_b, C)
#
#                 if xyz_b.shape[0] == 0:
#                     continue
#
#                 # ---- Build projection matrices ----
#                 viewmatrix_b, projmatrix_b = self._get_proj_matrices_for_batch(
#                     voxel_dict, b, xyz_b.device)
#
#                 # ---- Compute tanfovx / tanfovy from scaled lidar2img ----
#                 lidar2img_list = voxel_dict.get('lidar2img_matrices', None)
#                 if lidar2img_list is not None and b < len(lidar2img_list):
#                     P = lidar2img_list[b].to(xyz_b.device).float()
#                     # P[0,0] = fx_scaled (fx / image_stride)
#                     # P[1,1] = fy_scaled (fy / image_stride)
#                     fx_s = P[0, 0].item()
#                     fy_s = P[1, 1].item()
#                     tanfovx = W_feat / (2.0 * fx_s) if fx_s > 0 else 1.0
#                     tanfovy = H_feat / (2.0 * fy_s) if fy_s > 0 else 1.0
#                 else:
#                     tanfovx = 1.0
#                     tanfovy = 1.0
#
#                 # ---- Compute campos = camera position in world space ----
#                 # campos is the translation part of the inverse of viewmatrix
#                 try:
#                     campos = viewmatrix_b.inverse()[:3, 3]
#                 except Exception:
#                     campos = torch.zeros(3, device=xyz_b.device,
#                                         dtype=torch.float32)
#
#                 # ---- Render implicit feature map ----
#                 rendered_feat = self.implicit_branch(
#                     xyz=xyz_b,
#                     point_feats=feats_b,
#                     viewmatrix=viewmatrix_b,
#                     projmatrix=projmatrix_b,
#                     image_height=H_feat,
#                     image_width=W_feat,
#                     tanfovx=tanfovx,
#                     tanfovy=tanfovy,
#                     campos=campos,
#                 )  # (C_feat, H_feat, W_feat)
#
#                 rendered_feats_list.append(rendered_feat)
#                 image_feats_list.append(image_feat_map[b])
#
#             if len(rendered_feats_list) > 0:
#                 rendered_feats = torch.stack(rendered_feats_list, dim=0)
#                 image_feats = torch.stack(image_feats_list, dim=0)
#                 voxel_dict['rendered_implicit_feat'] = rendered_feats
#                 voxel_dict['image_feat_for_implicit'] = image_feats
#
#         return voxel_dict
#
#     def _get_proj_matrices_for_batch(self, voxel_dict, batch_idx, device):
#         """Get view/projection matrices for a specific batch sample.
#
#         Tries to use calibration info stored in voxel_dict by the
#         data preprocessor. Falls back to simple perspective projection.
#
#         Args:
#             voxel_dict: The voxel dictionary.
#             batch_idx: Batch index.
#             device: Target device.
#
#         Returns:
#             viewmatrix: (4, 4) tensor.
#             projmatrix: (4, 4) tensor.
#         """
#         # Try to get from stored calibration
#         calib_list = voxel_dict.get('calib_matrices', None)
#         if calib_list is not None and batch_idx < len(calib_list):
#             calib = calib_list[batch_idx]
#             viewmatrix = calib['viewmatrix'].to(device).float().contiguous()
#             projmatrix = calib['projmatrix'].to(device).float().contiguous()
#             return viewmatrix, projmatrix
#
#         # Try to construct from lidar2img stored by preprocessor
#         lidar2img_list = voxel_dict.get('lidar2img_matrices', None)
#         if lidar2img_list is not None and batch_idx < len(lidar2img_list):
#             lidar2img = lidar2img_list[batch_idx].to(device).float()
#             if lidar2img.shape == (3, 4):
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 lidar2img = torch.cat([lidar2img, pad], dim=0)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = lidar2img.contiguous()
#             return viewmatrix, projmatrix
#
#         # Fallback: construct a simple perspective projection
#         # Use image dimensions to approximate
#         images = voxel_dict.get('images', None)
#         if images is not None:
#             _, _, H_img, W_img = images.shape"""
# FRNet Backbone with Explicit + Implicit Constraint Branches.
#
# Extends FRNetExplicitBackbone by adding the implicit constraint branch
# based on 3D Gaussian Splatting. The implicit branch:
#
# 1. Takes the explicit-enhanced point features F_exp and point coordinates
# 2. Maps them to 3D Gaussian parameters via a lightweight MLP
# 3. Renders an implicit feature map via differentiable Gaussian splatting
# 4. The rendered feature map is stored for loss computation against image features
#
# The implicit branch is ONLY active during training. At inference it is
# removed, but its regularization persists in the learned backbone weights.
#
# Architecture flow:
#     Point Cloud → FRNet Backbone → F_u
#     F_u + Image → Explicit Branch → F_exp
#     F_exp + xyz → Implicit Branch (MLP → 3D Gaussians → Render) → F̂_img
#     Loss: L = L_seg + λ_exp·L_exp + λ_imp·||F̂_img - F_img||_1
# """
#
# from typing import Optional, Sequence
#
# import numpy as np
# import torch
# import torch.nn as nn
# from mmdet3d.registry import MODELS
# from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
# from mmengine.model import BaseModule
# from torch import Tensor
#
# from .frnet_explicit_backbone import FRNetExplicitBackbone
# from .implicit_constraint import ImplicitConstraintBranch
#
#
# @MODELS.register_module()
# class FRNetExplicitImplicitBackbone(FRNetExplicitBackbone):
#     """FRNet Backbone with both Explicit and Implicit Constraint Branches.
#
#     Inherits all functionality from FRNetExplicitBackbone and adds the
#     implicit constraint branch based on 3D Gaussian Splatting.
#
#     Additional Args:
#         enable_implicit (bool): Whether to enable implicit constraint. Default: True.
#         implicit_feat_channels (int): Feature dim for Gaussian splatting
#             (must match image encoder intermediate feature dim). Default: 128.
#         implicit_hidden_channels (int): MLP hidden layer dim. Default: 128.
#         implicit_num_mlp_layers (int): Number of MLP hidden layers. Default: 2.
#         implicit_alpha_min (float): Minimum opacity threshold. Default: 0.01.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  point_in_channels: int,
#                  output_shape: Sequence[int],
#                  depth: int,
#                  stem_channels: int = 128,
#                  num_stages: int = 4,
#                  out_channels: Sequence[int] = (128, 128, 128, 128),
#                  strides: Sequence[int] = (1, 2, 2, 2),
#                  dilations: Sequence[int] = (1, 1, 1, 1),
#                  fuse_channels: Sequence[int] = (256, 128),
#                  conv_cfg: OptConfigType = None,
#                  norm_cfg: ConfigType = dict(type='BN'),
#                  point_norm_cfg: ConfigType = dict(type='BN1d'),
#                  act_cfg: ConfigType = dict(type='LeakyReLU'),
#                  # Explicit constraint parameters (passed to parent)
#                  image_backbone_cfg: Optional[dict] = None,
#                  explicit_voxel_channels: int = 128,
#                  explicit_image_channels: int = 128,
#                  explicit_align_channels: int = 128,
#                  explicit_out_channels: int = 128,
#                  explicit_num_samples: int = 9,
#                  enable_explicit: bool = True,
#                  # Implicit constraint parameters (NEW)
#                  enable_implicit: bool = True,
#                  implicit_feat_channels: int = 128,
#                  implicit_hidden_channels: int = 128,
#                  implicit_num_mlp_layers: int = 2,
#                  implicit_alpha_min: float = 0.01,
#                  init_cfg: OptMultiConfig = None) -> None:
#
#         # Initialize parent (FRNetExplicitBackbone with all explicit params)
#         super().__init__(
#             in_channels=in_channels,
#             point_in_channels=point_in_channels,
#             output_shape=output_shape,
#             depth=depth,
#             stem_channels=stem_channels,
#             num_stages=num_stages,
#             out_channels=out_channels,
#             strides=strides,
#             dilations=dilations,
#             fuse_channels=fuse_channels,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             point_norm_cfg=point_norm_cfg,
#             act_cfg=act_cfg,
#             image_backbone_cfg=image_backbone_cfg,
#             explicit_voxel_channels=explicit_voxel_channels,
#             explicit_image_channels=explicit_image_channels,
#             explicit_align_channels=explicit_align_channels,
#             explicit_out_channels=explicit_out_channels,
#             explicit_num_samples=explicit_num_samples,
#             enable_explicit=enable_explicit,
#             init_cfg=init_cfg,
#         )
#
#         self.enable_implicit = enable_implicit
#
#         if enable_implicit:
#             # The point feature dimension after explicit fusion = fuse_channels[-1]
#             point_feat_dim = fuse_channels[-1]
#
#             self.implicit_branch = ImplicitConstraintBranch(
#                 point_feat_channels=point_feat_dim,
#                 image_feat_channels=implicit_feat_channels,
#                 hidden_channels=implicit_hidden_channels,
#                 num_mlp_layers=implicit_num_mlp_layers,
#                 alpha_min=implicit_alpha_min,
#             )
#
#     def _build_projection_matrices(self, meta, device):
#         """Build view and projection matrices from calibration data.
#
#         For projecting 3D Gaussians onto the camera image plane.
#
#         Args:
#             meta: metainfo dict with calibration parameters.
#             device: target device.
#
#         Returns:
#             viewmatrix: (4, 4) world-to-camera transform.
#             projmatrix: (4, 4) full projection (world-to-pixel).
#         """
#         if 'lidar2img' in meta:
#             lidar2img = torch.tensor(
#                 meta['lidar2img'], device=device, dtype=torch.float32)
#             if lidar2img.shape == (3, 4):
#                 # Pad to 4x4
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 lidar2img = torch.cat([lidar2img, pad], dim=0)
#
#             # For the GaussianCaR-style rasterizer, viewmatrix is the
#             # world-to-camera (extrinsic), and projmatrix is the full
#             # world-to-pixel projection.
#             # Since lidar2img = cam2img @ lidar2cam, we use:
#             #   viewmatrix = identity (pass 3D coords directly)
#             #   projmatrix = lidar2img (handles both extrinsic + intrinsic)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = lidar2img
#
#         elif 'lidar2cam' in meta and 'cam2img' in meta:
#             lidar2cam = torch.tensor(
#                 meta['lidar2cam'], device=device, dtype=torch.float32)
#             cam2img = torch.tensor(
#                 meta['cam2img'], device=device, dtype=torch.float32)
#
#             if cam2img.shape == (3, 3):
#                 cam2img_4x4 = torch.eye(4, device=device, dtype=torch.float32)
#                 cam2img_4x4[:3, :3] = cam2img
#                 cam2img = cam2img_4x4
#             elif cam2img.shape == (3, 4):
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 cam2img = torch.cat([cam2img, pad], dim=0)
#
#             viewmatrix = lidar2cam
#             projmatrix = cam2img @ lidar2cam
#         else:
#             # Fallback: identity (no proper projection)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = torch.eye(4, device=device, dtype=torch.float32)
#
#         return viewmatrix, projmatrix
#
#     def forward(self, voxel_dict: dict) -> dict:
#         """Forward pass with explicit + implicit constraint branches.
#
#         Calls the parent's forward (which includes explicit constraint),
#         then if training and implicit is enabled, runs the implicit branch.
#         """
#         # ---- Run parent forward (FRNet backbone + explicit constraint) ----
#         voxel_dict = super().forward(voxel_dict)
#
#         # ---- Implicit Constraint Branch (training only) ----
#         if (self.training and self.enable_implicit
#                 and voxel_dict.get('has_images', False)):
#
#             # Get the enhanced point features after explicit fusion
#             point_feats = voxel_dict['point_feats_backbone'][0]  # (N, C_fuse)
#
#             # Get point 3D coordinates
#             # In FRNet, after frustum_region_group, voxel_dict['voxels'] = (N, 4+)
#             # where columns are [x, y, z, intensity, ...]
#             voxels = voxel_dict.get('voxels', None)
#             if voxels is not None and voxels.dim() == 2 and voxels.shape[-1] >= 3:
#                 xyz = voxels[:, :3].clone()
#             else:
#                 # Fallback: cannot run implicit branch without coordinates
#                 return voxel_dict
#
#             pts_coors = voxel_dict['coors']  # (N, 3) [batch_idx, y, x]
#
#             # Get image feature map from explicit branch
#             # The image backbone was already run in the parent's forward
#             # We need to re-extract image features (or we can store them)
#             # Re-run image backbone (it's lightweight and results are needed)
#             images = voxel_dict['images']  # (B, 3, H_img, W_img)
#             image_feat_map = self.image_backbone(images)  # (B, C_img, H', W')
#
#             # Store image_feat_map for loss computation
#             voxel_dict['image_feat_map'] = image_feat_map
#
#             _, C_img, H_feat, W_feat = image_feat_map.shape
#             batch_size = int(pts_coors[:, 0].max().item()) + 1
#
#             rendered_feats_list = []
#             image_feats_list = []
#
#             for b in range(batch_size):
#                 batch_mask = (pts_coors[:, 0] == b)
#                 xyz_b = xyz[batch_mask]            # (N_b, 3)
#                 feats_b = point_feats[batch_mask]  # (N_b, C)
#
#                 if xyz_b.shape[0] == 0:
#                     continue
#
#                 # ---- Build projection matrices ----
#                 viewmatrix_b, projmatrix_b = self._get_proj_matrices_for_batch(
#                     voxel_dict, b, xyz_b.device)
#
#                 # ---- Compute tanfovx / tanfovy from scaled lidar2img ----
#                 lidar2img_list = voxel_dict.get('lidar2img_matrices', None)
#                 if lidar2img_list is not None and b < len(lidar2img_list):
#                     P = lidar2img_list[b].to(xyz_b.device).float()
#                     # P[0,0] = fx_scaled (fx / image_stride)
#                     # P[1,1] = fy_scaled (fy / image_stride)
#                     fx_s = P[0, 0].item()
#                     fy_s = P[1, 1].item()
#                     tanfovx = W_feat / (2.0 * fx_s) if fx_s > 0 else 1.0
#                     tanfovy = H_feat / (2.0 * fy_s) if fy_s > 0 else 1.0
#                 else:
#                     tanfovx = 1.0
#                     tanfovy = 1.0
#
#                 # ---- Compute campos = camera position in world space ----
#                 # campos is the translation part of the inverse of viewmatrix
#                 try:
#                     campos = viewmatrix_b.inverse()[:3, 3]
#                 except Exception:
#                     campos = torch.zeros(3, device=xyz_b.device,
#                                         dtype=torch.float32)
#
#                 # ---- Render implicit feature map ----
#                 rendered_feat = self.implicit_branch(
#                     xyz=xyz_b,
#                     point_feats=feats_b,
#                     viewmatrix=viewmatrix_b,
#                     projmatrix=projmatrix_b,
#                     image_height=H_feat,
#                     image_width=W_feat,
#                     tanfovx=tanfovx,
#                     tanfovy=tanfovy,
#                     campos=campos,
#                 )  # (C_feat, H_feat, W_feat)
#
#                 rendered_feats_list.append(rendered_feat)
#                 image_feats_list.append(image_feat_map[b])
#
#             if len(rendered_feats_list) > 0:
#                 rendered_feats = torch.stack(rendered_feats_list, dim=0)
#                 image_feats = torch.stack(image_feats_list, dim=0)
#                 voxel_dict['rendered_implicit_feat'] = rendered_feats
#                 voxel_dict['image_feat_for_implicit'] = image_feats
#
#         return voxel_dict
#
#     def _get_proj_matrices_for_batch(self, voxel_dict, batch_idx, device):
#         """Get view/projection matrices for a specific batch sample.
#
#         Tries to use calibration info stored in voxel_dict by the
#         data preprocessor. Falls back to simple perspective projection.
#
#         Args:
#             voxel_dict: The voxel dictionary.
#             batch_idx: Batch index.
#             device: Target device.
#
#         Returns:
#             viewmatrix: (4, 4) tensor.
#             projmatrix: (4, 4) tensor.
#         """
#         # Try to get from stored calibration
#         calib_list = voxel_dict.get('calib_matrices', None)
#         if calib_list is not None and batch_idx < len(calib_list):
#             calib = calib_list[batch_idx]
#             viewmatrix = calib['viewmatrix'].to(device).float().contiguous()
#             projmatrix = calib['projmatrix'].to(device).float().contiguous()
#             return viewmatrix, projmatrix
#
#         # Try to construct from lidar2img stored by preprocessor
#         lidar2img_list = voxel_dict.get('lidar2img_matrices', None)
#         if lidar2img_list is not None and batch_idx < len(lidar2img_list):
#             lidar2img = lidar2img_list[batch_idx].to(device).float()
#             if lidar2img.shape == (3, 4):
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 lidar2img = torch.cat([lidar2img, pad], dim=0)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = lidar2img.contiguous()
#             return viewmatrix, projmatrix
#
#         # Fallback: construct a simple perspective projection
#         # Use image dimensions to approximate
#         images = voxel_dict.get('images', None)
#         if images is not None:
#             _, _, H_img, W_img = images.shape
#             fx = W_img / 2.0
#             fy = H_img / 2.0
#             cx = W_img / 2.0
#             cy = H_img / 2.0
#         else:
#             fx, fy, cx, cy = 500.0, 500.0, 613.0, 185.0
#
#         # Simple perspective: project (x,y,z) -> (fx*x/z+cx, fy*y/z+cy)
#         # This is a fallback and won't be geometrically accurate
#         viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#         projmatrix = torch.tensor([
#             [fx,  0,  cx, 0],
#             [ 0, fy,  cy, 0],
#             [ 0,  0,   1, 0],
#             [ 0,  0,   1, 0],
#         ], device=device, dtype=torch.float32)
#
#         return viewmatrix.contiguous(), projmatrix.contiguous()
#             fx = W_img / 2.0
#             fy = H_img / 2.0
#             cx = W_img / 2.0
#             cy = H_img / 2.0
#         else:
#             fx, fy, cx, cy = 500.0, 500.0, 613.0, 185.0
#
#         # Simple perspective: project (x,y,z) -> (fx*x/z+cx, fy*y/z+cy)
#         # This is a fallback and won't be geometrically accurate
#         viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#         projmatrix = torch.tensor([
#             [fx,  0,  cx, 0],
#             [ 0, fy,  cy, 0],
#             [ 0,  0,   1, 0],
#             [ 0,  0,   1, 0],
#         ], device=device, dtype=torch.float32)
#
#         return viewmatrix.contiguous(), projmatrix.contiguous()
#             fy = H_img / 2.0
#             cx = W_img / 2.0
#             cy = H_img / 2.0
#         else:
#             fx, fy, cx, cy = 500.0, 500.0, 613.0, 185.0
#
#         # Simple perspective: project (x,y,z) -> (fx*x/z+cx, fy*y/z+cy)
#         # This is a fallback and won't be geometrically accurate
#         viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#         projmatrix = torch.tensor([
#             [fx,  0,  cx, 0],
#             [ 0, fy,  cy, 0],
#             [ 0,  0,   1, 0],
#             [ 0,  0,   1, 0],
#         ], device=device, dtype=torch.float32)
#
#         return viewmatrix.contiguous(), projmatrix.contiguous()
#                 batch_mask = (pts_coors[:, 0] == b)
#                 xyz_b = xyz[batch_mask]           # (N_b, 3)
#                 feats_b = point_feats[batch_mask]  # (N_b, C)
#
#                 if xyz_b.shape[0] == 0:
#                     continue
#
#                 # Build projection matrices
#                 # These are stored by the data preprocessor in voxel_dict
#                 viewmatrix_b, projmatrix_b = self._get_proj_matrices_for_batch(
#                     voxel_dict, b, xyz_b.device)
#
#                 # Render implicit feature map for this sample
#                 rendered_feat = self.implicit_branch(
#                     xyz=xyz_b,
#                     point_feats=feats_b,
#                     viewmatrix=viewmatrix_b,
#                     projmatrix=projmatrix_b,
#                     image_height=H_feat,
#                     image_width=W_feat,
#                 )  # (C_feat, H_feat, W_feat)
#
#                 rendered_feats_list.append(rendered_feat)
#                 image_feats_list.append(image_feat_map[b])
#
#             if len(rendered_feats_list) > 0:
#                 rendered_feats = torch.stack(rendered_feats_list, dim=0)
#                 image_feats = torch.stack(image_feats_list, dim=0)
#                 voxel_dict['rendered_implicit_feat'] = rendered_feats
#                 voxel_dict['image_feat_for_implicit'] = image_feats
#
#         return voxel_dict
#
#     def _get_proj_matrices_for_batch(self, voxel_dict, batch_idx, device):
#         """Get view/projection matrices for a specific batch sample.
#
#         Tries to use calibration info stored in voxel_dict by the
#         data preprocessor. Falls back to simple perspective projection.
#
#         Args:
#             voxel_dict: The voxel dictionary.
#             batch_idx: Batch index.
#             device: Target device.
#
#         Returns:
#             viewmatrix: (4, 4) tensor.
#             projmatrix: (4, 4) tensor.
#         """
#         # Try to get from stored calibration
#         calib_list = voxel_dict.get('calib_matrices', None)
#         if calib_list is not None and batch_idx < len(calib_list):
#             calib = calib_list[batch_idx]
#             viewmatrix = calib['viewmatrix'].to(device).float().contiguous()
#             projmatrix = calib['projmatrix'].to(device).float().contiguous()
#             return viewmatrix, projmatrix
#
#         # Try to construct from lidar2img stored by preprocessor
#         lidar2img_list = voxel_dict.get('lidar2img_matrices', None)
#         if lidar2img_list is not None and batch_idx < len(lidar2img_list):
#             lidar2img = lidar2img_list[batch_idx].to(device).float()
#             if lidar2img.shape == (3, 4):
#                 pad = torch.tensor([[0, 0, 0, 1]], device=device, dtype=torch.float32)
#                 lidar2img = torch.cat([lidar2img, pad], dim=0)
#             viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#             projmatrix = lidar2img.contiguous()
#             return viewmatrix, projmatrix
#
#         # Fallback: construct a simple perspective projection
#         # Use image dimensions to approximate
#         images = voxel_dict.get('images', None)
#         if images is not None:
#             _, _, H_img, W_img = images.shape
#             fx = W_img / 2.0
#             fy = H_img / 2.0
#             cx = W_img / 2.0
#             cy = H_img / 2.0
#         else:
#             fx, fy, cx, cy = 500.0, 500.0, 613.0, 185.0
#
#         # Simple perspective: project (x,y,z) -> (fx*x/z+cx, fy*y/z+cy)
#         # This is a fallback and won't be geometrically accurate
#         viewmatrix = torch.eye(4, device=device, dtype=torch.float32)
#         projmatrix = torch.tensor([
#             [fx,  0,  cx, 0],
#             [ 0, fy,  cy, 0],
#             [ 0,  0,   1, 0],
#             [ 0,  0,   1, 0],
#         ], device=device, dtype=torch.float32)
#
#         return viewmatrix.contiguous(), projmatrix.contiguous()
