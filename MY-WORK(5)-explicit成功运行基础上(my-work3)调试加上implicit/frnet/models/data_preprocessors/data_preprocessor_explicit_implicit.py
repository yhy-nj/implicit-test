"""
Data Preprocessor for Explicit + Implicit Constraint Branches.

Extends ExplicitConstraintPreprocessor to additionally store camera
calibration matrices (lidar2img) in voxel_dict for use by the implicit
constraint branch's Gaussian Splatting renderer.
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from torch import Tensor

from .data_preprocessor_explicit import ExplicitConstraintPreprocessor


@MODELS.register_module()
class ExplicitImplicitPreprocessor(ExplicitConstraintPreprocessor):
    """Preprocessor for explicit + implicit constraint branches.

    In addition to everything ExplicitConstraintPreprocessor does, this
    preprocessor also stores the lidar2img calibration matrices in
    voxel_dict['lidar2img_matrices'] so that the implicit branch can
    construct view/projection matrices for Gaussian Splatting rendering.

    Args:
        Same as ExplicitConstraintPreprocessor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, data: dict, training: bool = False) -> dict:
        """Process data and additionally store calibration matrices."""
        # Run parent forward (frustum grouping + image processing + projection)
        result = super().forward(data, training)

        # Additionally store lidar2img matrices for implicit branch
        data_samples = result['data_samples']
        voxel_dict = result['inputs']['voxels']

        if voxel_dict.get('has_images', False) and data_samples is not None:
            lidar2img_matrices = []
            for ds in data_samples:
                meta = ds.metainfo
                if 'lidar2img' in meta:
                    lidar2img = meta['lidar2img']
                    if not isinstance(lidar2img, torch.Tensor):
                        lidar2img = torch.tensor(
                            lidar2img, dtype=torch.float32)
                    # Ensure 4x4
                    if lidar2img.shape == (3, 4):
                        pad = torch.tensor(
                            [[0, 0, 0, 1]], dtype=torch.float32)
                        lidar2img = torch.cat([lidar2img, pad], dim=0)

                    # Scale the projection to match feature map resolution
                    # The image backbone outputs features at image_stride
                    # So we need to scale the intrinsic part
                    lidar2img_scaled = lidar2img.clone()
                    lidar2img_scaled[0, :] = lidar2img[0, :] / self.image_stride
                    lidar2img_scaled[1, :] = lidar2img[1, :] / self.image_stride

                    lidar2img_matrices.append(lidar2img_scaled)
                else:
                    # Fallback identity
                    lidar2img_matrices.append(
                        torch.eye(4, dtype=torch.float32))

            voxel_dict['lidar2img_matrices'] = lidar2img_matrices

        return result
