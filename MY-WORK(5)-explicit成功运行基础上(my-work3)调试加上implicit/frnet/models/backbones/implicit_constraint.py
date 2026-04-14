"""
Implicit Constraint Branch based on 3D Gaussian Splatting.

This module implements the implicit constraint branch that:
1. Maps each point's coordinates + explicit-enhanced features to 3D Gaussian parameters
   via a lightweight MLP (Points-to-Gaussians).
2. Performs differentiable Gaussian Splatting to render an implicit feature map.
3. The rendered feature map is supervised against image encoder features (L_imp).

The branch is ONLY active during training. At inference, it is removed,
but the regularization it imposed on the backbone persists in the learned weights.

Key components:
- PointsToGaussiansMLP: Predicts (μ_offset, Σ_compact, α, e) per point.
- ImplicitConstraintBranch: Orchestrates MLP + differentiable rendering.
- ImplicitConstraintLoss: L1 loss between rendered and image features.

IMPORTANT - diff-gaussian-rasterization installation:
    Uses the ORIGINAL 3DGS diff-gaussian-rasterization repo
    (https://github.com/graphdeco-inria/diff-gaussian-rasterization)
    with NUM_CHANNELS modified to 128 in cuda_rasterizer/config.h:
        #define NUM_CHANNELS 128
    Then rebuild:  pip install . (or python setup.py install)

    This allows single-pass rendering of 128-dimensional feature maps
    using perspective projection (the original 3DGS projection pipeline).

References:
- GaussianCaR (Montiel-Marín et al., 2025): Points-to-Gaussians module,
  coarse-to-fine positioning, compact covariance representation.
- 3D Gaussian Splatting (Kerbl et al., 2023): Differentiable rasterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch import Tensor
from typing import Optional, Tuple

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    HAS_DIFF_GAUSSIAN_RASTERIZATION = True
except ImportError:
    HAS_DIFF_GAUSSIAN_RASTERIZATION = False


class PointsToGaussiansMLP(BaseModule):
    """MLP that maps point coordinates + features to 3D Gaussian parameters.

    Input:  concat(xyz_i, F_exp_i)  ->  shape (N, 3 + C_in)
    Output: (offset_3d, cov_compact_6d, opacity_1d, implicit_feat)

    Following GaussianCaR's Points-to-Gaussians design:
    - Position: coarse-to-fine, mu_i = xyz_i + delta_p_i (MLP predicts offset only)
    - Covariance: compact 6D via Cholesky L*L^T -> guaranteed positive definite
    - Opacity: sigmoid -> [0, 1], with alpha_min threshold
    - Implicit feature: e_i in R^{C_feat}

    Args:
        in_channels (int): Input feature dimension (3 + C_exp).
        hidden_channels (int): Hidden layer dimension. Default: 128.
        feat_channels (int): Output implicit feature dimension. Default: 128.
        num_layers (int): Number of hidden layers. Default: 2.
        alpha_min (float): Minimum opacity threshold. Default: 0.01.
    """

    def __init__(self,
                 in_channels: int,
                 hidden_channels: int = 128,
                 feat_channels: int = 128,
                 num_layers: int = 2,
                 alpha_min: float = 0.01) -> None:
        super().__init__()
        self.alpha_min = alpha_min
        self.feat_channels = feat_channels

        # Shared trunk
        layers = []
        ch = in_channels
        for _ in range(num_layers):
            layers.append(nn.Linear(ch, hidden_channels))
            layers.append(nn.ReLU(inplace=True))
            ch = hidden_channels
        self.trunk = nn.Sequential(*layers)

        # Separate heads for each Gaussian parameter
        self.offset_head = nn.Linear(hidden_channels, 3)       # delta_p (position offset)
        self.cov_head = nn.Linear(hidden_channels, 6)           # Cholesky factors
        self.opacity_head = nn.Linear(hidden_channels, 1)       # alpha (pre-sigmoid)
        self.feat_head = nn.Linear(hidden_channels, feat_channels)  # e_i

        self._init_weights()

    def _init_weights(self):
        """Initialize for stable coarse-to-fine training."""
        nn.init.zeros_(self.offset_head.weight)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.zeros_(self.opacity_head.bias)

    def forward(self, xyz: Tensor, features: Tensor) -> dict:
        """
        Args:
            xyz (Tensor): Point coordinates, shape (N, 3).
            features (Tensor): Enhanced point features F_exp, shape (N, C).

        Returns:
            dict with keys:
                means3D: (N, 3) Gaussian positions mu_i = xyz_i + delta_p_i
                cov3D_precomp: (N, 6) upper triangle of covariance Sigma = L*L^T
                opacities: (N, 1) opacity values after sigmoid
                features: (N, C_feat) implicit feature vectors
                mask: (N,) boolean mask for valid Gaussians (opacity >= alpha_min)
        """
        inp = torch.cat([xyz, features], dim=-1)  # (N, 3+C)
        h = self.trunk(inp)  # (N, hidden)

        # Position: coarse-to-fine offset
        offset = self.offset_head(h)  # (N, 3)
        means3D = xyz + offset  # mu_i = xyz_i + delta_p_i

        # Covariance via Cholesky decomposition: Sigma = L*L^T (guaranteed PD)
        # L = [[l0, 0,  0 ],
        #      [l1, l2, 0 ],
        #      [l3, l4, l5]]
        # Diagonal must be positive -> softplus
        cov_raw = self.cov_head(h)  # (N, 6)
        l0 = F.softplus(cov_raw[:, 0])
        l1 = cov_raw[:, 1]
        l2 = F.softplus(cov_raw[:, 2])
        l3 = cov_raw[:, 3]
        l4 = cov_raw[:, 4]
        l5 = F.softplus(cov_raw[:, 5])

        # Sigma = L*L^T upper triangle: [xx, xy, xz, yy, yz, zz]
        cov_xx = l0 * l0
        cov_xy = l0 * l1
        cov_xz = l0 * l3
        cov_yy = l1 * l1 + l2 * l2
        cov_yz = l1 * l3 + l2 * l4
        cov_zz = l3 * l3 + l4 * l4 + l5 * l5
        cov_compact = torch.stack(
            [cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz], dim=-1)  # (N, 6)

        # Opacity: sigmoid with min threshold
        alpha_raw = self.opacity_head(h)  # (N, 1)
        opacities = torch.sigmoid(alpha_raw)  # (N, 1)

        # Mask: prune Gaussians with negligible opacity
        mask = (opacities.squeeze(-1) >= self.alpha_min)

        # Implicit features
        impl_feats = self.feat_head(h)  # (N, C_feat)

        return {
            'means3D': means3D,
            'cov3D_precomp': cov_compact,
            'opacities': opacities,
            'features': impl_feats,
            'mask': mask,
        }


class ImplicitConstraintBranch(BaseModule):
    """Implicit Constraint Branch using 3D Gaussian Splatting.

    Single-pass rendering using the ORIGINAL 3DGS diff-gaussian-rasterization
    compiled with NUM_CHANNELS=128 in config.h. This uses perspective projection
    (transformPoint4x4 + computeCov2D), so tanfovx/tanfovy/campos are required.

    Args:
        point_feat_channels (int): Dimension of F_exp (point features).
        image_feat_channels (int): Dimension of image encoder features.
            Must equal NUM_CHANNELS in config.h (128). Default: 128.
        hidden_channels (int): MLP hidden layer dimension. Default: 128.
        num_mlp_layers (int): Number of MLP hidden layers. Default: 2.
        alpha_min (float): Minimum opacity threshold. Default: 0.01.
    """

    def __init__(self,
                 point_feat_channels: int,
                 image_feat_channels: int = 128,
                 hidden_channels: int = 128,
                 num_mlp_layers: int = 2,
                 alpha_min: float = 0.01) -> None:
        super().__init__()

        assert HAS_DIFF_GAUSSIAN_RASTERIZATION, (
            "diff_gaussian_rasterization is not installed. "
            "Please install from original 3DGS repo with NUM_CHANNELS=128: "
            "1) git clone https://github.com/graphdeco-inria/diff-gaussian-rasterization\n"
            "2) Edit cuda_rasterizer/config.h: #define NUM_CHANNELS 128\n"
            "3) pip install ."
        )

        self.image_feat_channels = image_feat_channels

        # MLP: (3 + point_feat_channels) -> Gaussian params
        self.mlp = PointsToGaussiansMLP(
            in_channels=3 + point_feat_channels,
            hidden_channels=hidden_channels,
            feat_channels=image_feat_channels,
            num_layers=num_mlp_layers,
            alpha_min=alpha_min,
        )

    def forward(self,
                xyz: Tensor,
                point_feats: Tensor,
                viewmatrix: Tensor,
                projmatrix: Tensor,
                campos: Tensor,
                tanfovx: float,
                tanfovy: float,
                image_height: int,
                image_width: int) -> Tensor:
        """Render implicit feature map via single-pass Gaussian Splatting.

        Uses original 3DGS rasterizer compiled with NUM_CHANNELS=128,
        so all 128 feature channels are rendered in one CUDA kernel call.
        Uses perspective projection (requires real tanfov and campos).

        Args:
            xyz (Tensor): Point coordinates, shape (N, 3).
            point_feats (Tensor): Enhanced point features F_exp, shape (N, C).
            viewmatrix (Tensor): World-to-camera matrix, shape (4, 4).
            projmatrix (Tensor): Full projection matrix (view * proj), shape (4, 4).
            campos (Tensor): Camera position in world coordinates, shape (3,).
            tanfovx (float): tan(FoV_x / 2).
            tanfovy (float): tan(FoV_y / 2).
            image_height (int): Target rendering height (at feature map scale).
            image_width (int): Target rendering width (at feature map scale).

        Returns:
            Tensor: Rendered implicit feature map F_hat_img, shape (C_feat, H, W).
        """
        # Step 1: MLP predicts Gaussian parameters
        gaussian_params = self.mlp(xyz, point_feats)

        means3D = gaussian_params['means3D']
        cov3D = gaussian_params['cov3D_precomp']
        opacities = gaussian_params['opacities']
        impl_feats = gaussian_params['features']
        mask = gaussian_params['mask']

        # Apply opacity mask - prune low-opacity Gaussians
        if mask.sum() < means3D.shape[0]:
            means3D = means3D[mask]
            cov3D = cov3D[mask]
            opacities = opacities[mask]
            impl_feats = impl_feats[mask]

        N = means3D.shape[0]
        if N == 0:
            return torch.zeros(
                self.image_feat_channels, image_height, image_width,
                device=xyz.device, dtype=xyz.dtype)

        # Step 2: Set up rasterization settings
        # Original 3DGS: raster_settings is passed to GaussianRasterizer constructor
        bg_color = torch.zeros(
            self.image_feat_channels, device=xyz.device, dtype=torch.float32)

        raster_settings = GaussianRasterizationSettings(
            image_height=image_height,
            image_width=image_width,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmatrix.float().contiguous(),
            projmatrix=projmatrix.float().contiguous(),
            sh_degree=0,
            campos=campos.float().contiguous(),
            prefiltered=False,
            debug=False,
        )

        # Original 3DGS: GaussianRasterizer takes raster_settings in constructor
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # means2D: screen-space positions (computed internally by rasterizer,
        # but we provide a tensor for gradient tracking)
        means2D = torch.zeros(
            N, 2, device=xyz.device, dtype=torch.float32, requires_grad=True)

        # Step 3: Single-pass rasterization
        # With NUM_CHANNELS=128, this renders all 128 feature channels at once.
        # Original 3DGS rasterizer forward signature:
        #   rasterizer(means3D, means2D, shs, colors_precomp,
        #              opacities, scales, rotations, cov3D_precomp)
        rendered_feat, _ = rasterizer(
            means3D=means3D.float().contiguous(),
            means2D=means2D,
            shs=None,
            colors_precomp=impl_feats.float().contiguous(),
            opacities=opacities.float().contiguous(),
            scales=None,
            rotations=None,
            cov3D_precomp=cov3D.float().contiguous(),
        )  # (128, H, W)

        return rendered_feat


class ImplicitConstraintLoss(nn.Module):
    """Implicit constraint loss: L1 distance between rendered and image features.

    L_imp = ||F_hat_img - F_img||_1

    Args:
        loss_weight (float): Weight lambda_imp for the implicit loss. Default: 1.0.
    """

    def __init__(self, loss_weight: float = 1.0) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, rendered_feat: Tensor, image_feat: Tensor) -> Tensor:
        """
        Args:
            rendered_feat: F_hat_img from Gaussian splatting, (B, C, H, W) or (C, H, W).
            image_feat: F_img from image encoder, same shape.

        Returns:
            Tensor: Scalar loss value.
        """
        # Ensure same spatial resolution
        if rendered_feat.shape[-2:] != image_feat.shape[-2:]:
            if rendered_feat.dim() == 3:
                rendered_feat = rendered_feat.unsqueeze(0)
            rendered_feat = F.interpolate(
                rendered_feat,
                size=image_feat.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            if rendered_feat.dim() == 4 and image_feat.dim() == 3:
                rendered_feat = rendered_feat.squeeze(0)

        loss = F.l1_loss(rendered_feat, image_feat)
        return self.loss_weight * loss


# """
# Implicit Constraint Branch based on 3D Gaussian Splatting.
#
# This module implements the implicit constraint branch that:
# 1. Maps each point's coordinates + explicit-enhanced features to 3D Gaussian parameters
#    via a lightweight MLP (Points-to-Gaussians).
# 2. Performs differentiable Gaussian Splatting to render an implicit feature map.
# 3. The rendered feature map is supervised against image encoder features (L_imp).
#
# The branch is ONLY active during training. At inference, it is removed,
# but the regularization it imposed on the backbone persists in the learned weights.
#
# Key components:
# - PointsToGaussiansMLP: Predicts (μ_offset, Σ_compact, α, e) per point.
# - ImplicitConstraintBranch: Orchestrates MLP + differentiable rendering.
# - ImplicitConstraintLoss: L1 loss between rendered and image features.
#
# IMPORTANT - diff-gaussian-rasterization installation:
#     Must install from the GaussianCaR repo, which compiles with NUM_CHANNELS=128
#     (not the original 3DGS repo which uses NUM_CHANNELS=3).
#     The config.h in GaussianCaR sets:
#         #define NUM_CHANNELS 128
#         #define BLOCK_X 8
#         #define BLOCK_Y 8
#     This allows single-pass rendering of 128-dimensional feature maps.
#
# References:
# - GaussianCaR (Montiel-Marín et al., 2025): Points-to-Gaussians module,
#   coarse-to-fine positioning, compact covariance representation.
# - 3D Gaussian Splatting (Kerbl et al., 2023): Differentiable rasterization.
# """
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmengine.model import BaseModule
# from torch import Tensor
# from typing import Optional, Tuple
#
# try:
#     from diff_gaussian_rasterization import (
#         GaussianRasterizationSettings,
#         GaussianRasterizer,
#     )
#     HAS_DIFF_GAUSSIAN_RASTERIZATION = True
# except ImportError:
#     HAS_DIFF_GAUSSIAN_RASTERIZATION = False
#
#
# class PointsToGaussiansMLP(BaseModule):
#     """MLP that maps point coordinates + features to 3D Gaussian parameters.
#
#     Input:  concat(xyz_i, F_exp_i)  ->  shape (N, 3 + C_in)
#     Output: (offset_3d, cov_compact_6d, opacity_1d, implicit_feat)
#
#     Following GaussianCaR's Points-to-Gaussians design:
#     - Position: coarse-to-fine, mu_i = xyz_i + delta_p_i (MLP predicts offset only)
#     - Covariance: compact 6D via Cholesky L*L^T -> guaranteed positive definite
#     - Opacity: sigmoid -> [0, 1], with alpha_min threshold
#     - Implicit feature: e_i in R^{C_feat}
#
#     Args:
#         in_channels (int): Input feature dimension (3 + C_exp).
#         hidden_channels (int): Hidden layer dimension. Default: 128.
#         feat_channels (int): Output implicit feature dimension. Default: 128.
#         num_layers (int): Number of hidden layers. Default: 2.
#         alpha_min (float): Minimum opacity threshold. Default: 0.01.
#     """
#
#     def __init__(self,
#                  in_channels: int,
#                  hidden_channels: int = 128,
#                  feat_channels: int = 128,
#                  num_layers: int = 2,
#                  alpha_min: float = 0.01) -> None:
#         super().__init__()
#         self.alpha_min = alpha_min
#         self.feat_channels = feat_channels
#
#         # Shared trunk
#         layers = []
#         ch = in_channels
#         for _ in range(num_layers):
#             layers.append(nn.Linear(ch, hidden_channels))
#             layers.append(nn.ReLU(inplace=True))
#             ch = hidden_channels
#         self.trunk = nn.Sequential(*layers)
#
#         # Separate heads for each Gaussian parameter
#         self.offset_head = nn.Linear(hidden_channels, 3)       # delta_p (position offset)
#         self.cov_head = nn.Linear(hidden_channels, 6)           # Cholesky factors
#         self.opacity_head = nn.Linear(hidden_channels, 1)       # alpha (pre-sigmoid)
#         self.feat_head = nn.Linear(hidden_channels, feat_channels)  # e_i
#
#         self._init_weights()
#
#     def _init_weights(self):
#         """Initialize for stable coarse-to-fine training."""
#         nn.init.zeros_(self.offset_head.weight)
#         nn.init.zeros_(self.offset_head.bias)
#         nn.init.zeros_(self.opacity_head.bias)
#
#     def forward(self, xyz: Tensor, features: Tensor) -> dict:
#         """
#         Args:
#             xyz (Tensor): Point coordinates, shape (N, 3).
#             features (Tensor): Enhanced point features F_exp, shape (N, C).
#
#         Returns:
#             dict with keys:
#                 means3D: (N, 3) Gaussian positions mu_i = xyz_i + delta_p_i
#                 cov3D_precomp: (N, 6) upper triangle of covariance Sigma = L*L^T
#                 opacities: (N, 1) opacity values after sigmoid
#                 features: (N, C_feat) implicit feature vectors
#                 mask: (N,) boolean mask for valid Gaussians (opacity >= alpha_min)
#         """
#         inp = torch.cat([xyz, features], dim=-1)  # (N, 3+C)
#         h = self.trunk(inp)  # (N, hidden)
#
#         # Position: coarse-to-fine offset
#         offset = self.offset_head(h)  # (N, 3)
#         means3D = xyz + offset  # mu_i = xyz_i + delta_p_i
#
#         # Covariance via Cholesky decomposition: Sigma = L*L^T (guaranteed PD)
#         # L = [[l0, 0,  0 ],
#         #      [l1, l2, 0 ],
#         #      [l3, l4, l5]]
#         # Diagonal must be positive -> softplus
#         cov_raw = self.cov_head(h)  # (N, 6)
#         l0 = F.softplus(cov_raw[:, 0])
#         l1 = cov_raw[:, 1]
#         l2 = F.softplus(cov_raw[:, 2])
#         l3 = cov_raw[:, 3]
#         l4 = cov_raw[:, 4]
#         l5 = F.softplus(cov_raw[:, 5])
#
#         # Sigma = L*L^T upper triangle: [xx, xy, xz, yy, yz, zz]
#         cov_xx = l0 * l0
#         cov_xy = l0 * l1
#         cov_xz = l0 * l3
#         cov_yy = l1 * l1 + l2 * l2
#         cov_yz = l1 * l3 + l2 * l4
#         cov_zz = l3 * l3 + l4 * l4 + l5 * l5
#         cov_compact = torch.stack(
#             [cov_xx, cov_xy, cov_xz, cov_yy, cov_yz, cov_zz], dim=-1)  # (N, 6)
#
#         # Opacity: sigmoid with min threshold
#         alpha_raw = self.opacity_head(h)  # (N, 1)
#         opacities = torch.sigmoid(alpha_raw)  # (N, 1)
#
#         # Mask: prune Gaussians with negligible opacity
#         mask = (opacities.squeeze(-1) >= self.alpha_min)
#
#         # Implicit features
#         impl_feats = self.feat_head(h)  # (N, C_feat)
#
#         return {
#             'means3D': means3D,
#             'cov3D_precomp': cov_compact,
#             'opacities': opacities,
#             'features': impl_feats,
#             'mask': mask,
#         }
#
#
# class ImplicitConstraintBranch(BaseModule):
#     """Implicit Constraint Branch using 3D Gaussian Splatting.
#
#     Single-pass rendering: uses the GaussianCaR version of
#     diff-gaussian-rasterization which is compiled with NUM_CHANNELS=128.
#     This renders all 128 feature channels in a single rasterization pass.
#
#     Args:
#         point_feat_channels (int): Dimension of F_exp (point features).
#         image_feat_channels (int): Dimension of image encoder features.
#             Must equal NUM_CHANNELS in config.h (128). Default: 128.
#         hidden_channels (int): MLP hidden layer dimension. Default: 128.
#         num_mlp_layers (int): Number of MLP hidden layers. Default: 2.
#         alpha_min (float): Minimum opacity threshold. Default: 0.01.
#     """
#
#     def __init__(self,
#                  point_feat_channels: int,
#                  image_feat_channels: int = 128,
#                  hidden_channels: int = 128,
#                  num_mlp_layers: int = 2,
#                  alpha_min: float = 0.01) -> None:
#         super().__init__()
#
#         assert HAS_DIFF_GAUSSIAN_RASTERIZATION, (
#             "diff_gaussian_rasterization is not installed. "
#             "Please install from GaussianCaR repo (NUM_CHANNELS=128): "
#             "cd gaussiancar/ops/diff-gaussian-rasterization && "
#             "python setup.py install"
#         )
#
#         self.image_feat_channels = image_feat_channels
#
#         # MLP: (3 + point_feat_channels) -> Gaussian params
#         self.mlp = PointsToGaussiansMLP(
#             in_channels=3 + point_feat_channels,
#             hidden_channels=hidden_channels,
#             feat_channels=image_feat_channels,
#             num_layers=num_mlp_layers,
#             alpha_min=alpha_min,
#         )
#
#         # Rasterizer instance
#         self.rasterizer = GaussianRasterizer()
#
#     def forward(self,
#                 xyz: Tensor,
#                 point_feats: Tensor,
#                 viewmatrix: Tensor,
#                 projmatrix: Tensor,
#                 image_height: int,
#                 image_width: int) -> Tensor:
#         """Render implicit feature map via single-pass Gaussian Splatting.
#
#         Uses GaussianCaR's rasterizer compiled with NUM_CHANNELS=128,
#         so all 128 feature channels are rendered in one CUDA kernel call.
#
#         Args:
#             xyz (Tensor): Point coordinates, shape (N, 3).
#             point_feats (Tensor): Enhanced point features F_exp, shape (N, C).
#             viewmatrix (Tensor): World-to-camera matrix, shape (4, 4).
#             projmatrix (Tensor): Full projection matrix, shape (4, 4).
#             image_height (int): Target rendering height (at feature map scale).
#             image_width (int): Target rendering width (at feature map scale).
#
#         Returns:
#             Tensor: Rendered implicit feature map F_hat_img, shape (C_feat, H, W).
#         """
#         # Step 1: MLP predicts Gaussian parameters
#         gaussian_params = self.mlp(xyz, point_feats)
#
#         means3D = gaussian_params['means3D']
#         cov3D = gaussian_params['cov3D_precomp']
#         opacities = gaussian_params['opacities']
#         impl_feats = gaussian_params['features']
#         mask = gaussian_params['mask']
#
#         # Apply opacity mask - prune low-opacity Gaussians
#         if mask.sum() < means3D.shape[0]:
#             means3D = means3D[mask]
#             cov3D = cov3D[mask]
#             opacities = opacities[mask]
#             impl_feats = impl_feats[mask]
#
#         N = means3D.shape[0]
#         if N == 0:
#             return torch.zeros(
#                 self.image_feat_channels, image_height, image_width,
#                 device=xyz.device, dtype=xyz.dtype)
#
#         # Step 2: Set up rasterization settings
#         bg_color = torch.zeros(
#             self.image_feat_channels, device=xyz.device, dtype=torch.float32)
#
#         raster_settings = GaussianRasterizationSettings(
#             image_height=image_height,
#             image_width=image_width,
#             tanfovx=1.0,   # not used by GaussianCaR's modified rasterizer
#             tanfovy=1.0,   # not used by GaussianCaR's modified rasterizer
#             bg=bg_color,
#             scale_modifier=1.0,
#             viewmatrix=viewmatrix.float().contiguous(),
#             projmatrix=projmatrix.float().contiguous(),
#             sh_degree=0,
#             campos=torch.zeros(3, device=xyz.device),  # not used
#             prefiltered=False,
#             debug=False,
#         )
#         self.rasterizer.set_raster_settings(raster_settings)
#
#         # means2D: screen-space positions (computed internally by rasterizer,
#         # but we provide a tensor for gradient tracking)
#         means2D = torch.zeros(
#             N, 2, device=xyz.device, dtype=torch.float32, requires_grad=True)
#
#         # Step 3: Single-pass rasterization
#         # With NUM_CHANNELS=128, this renders all 128 feature channels at once
#         rendered_feat, _ = self.rasterizer(
#             means3D=means3D.float().contiguous(),
#             means2D=means2D,
#             opacities=opacities.float().contiguous(),
#             colors_precomp=impl_feats.float().contiguous(),
#             cov3D_precomp=cov3D.float().contiguous(),
#         )  # (128, H, W)
#
#         return rendered_feat
#
#
# class ImplicitConstraintLoss(nn.Module):
#     """Implicit constraint loss: L1 distance between rendered and image features.
#
#     L_imp = ||F_hat_img - F_img||_1
#
#     Args:
#         loss_weight (float): Weight lambda_imp for the implicit loss. Default: 1.0.
#     """
#
#     def __init__(self, loss_weight: float = 1.0) -> None:
#         super().__init__()
#         self.loss_weight = loss_weight
#
#     def forward(self, rendered_feat: Tensor, image_feat: Tensor) -> Tensor:
#         """
#         Args:
#             rendered_feat: F_hat_img from Gaussian splatting, (B, C, H, W) or (C, H, W).
#             image_feat: F_img from image encoder, same shape.
#
#         Returns:
#             Tensor: Scalar loss value.
#         """
#         # Ensure same spatial resolution
#         if rendered_feat.shape[-2:] != image_feat.shape[-2:]:
#             if rendered_feat.dim() == 3:
#                 rendered_feat = rendered_feat.unsqueeze(0)
#             rendered_feat = F.interpolate(
#                 rendered_feat,
#                 size=image_feat.shape[-2:],
#                 mode='bilinear',
#                 align_corners=False,
#             )
#             if rendered_feat.dim() == 4 and image_feat.dim() == 3:
#                 rendered_feat = rendered_feat.squeeze(0)
#
#         loss = F.l1_loss(rendered_feat, image_feat)
#         return self.loss_weight * loss