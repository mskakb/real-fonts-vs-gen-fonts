"""Signed Distance Field (SDF) utilities for binary font images."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def binary_image_to_sdf(
    image: np.ndarray | torch.Tensor,
    max_dist: float = 10.0,
) -> np.ndarray | torch.Tensor:
    """Convert a binary font image to an SDF image.

    Args:
        image: Binary image with stroke=0, background=1. Shape (H, W), (1,H,W), or (3,H,W).
               Accepts both numpy arrays and torch Tensors.
        max_dist: Clipping distance in pixels for normalization. Values beyond ±max_dist
                  are clipped to ±1 before scaling to [0, 1].

    Returns:
        SDF image in [0, 1] with the same shape and type as the input:
          - stroke interior (far from contour): → 1.0
          - background (far from contour):      → 0.0
          - contour (boundary):                 ≈ 0.5
    """
    is_tensor = isinstance(image, torch.Tensor)
    arr: np.ndarray = image.numpy() if is_tensor else np.asarray(image, dtype=np.float32)

    # Handle channel dimensions: (C,H,W) or (H,W) → (H,W)
    squeezed = arr.ndim == 3
    if squeezed:
        if arr.shape[0] == 1:
            arr2d = arr[0]
        else:
            # multichannel: average channels then binarize
            arr2d = arr.mean(axis=0)
    else:
        arr2d = arr

    stroke_mask = arr2d < 0.5  # True = stroke pixel

    # All-white or all-black image: SDF is constant
    if stroke_mask.all():
        sdf_01 = np.ones_like(arr2d, dtype=np.float32)
    elif (~stroke_mask).all():
        sdf_01 = np.zeros_like(arr2d, dtype=np.float32)
    else:
        dist_in = distance_transform_edt(stroke_mask).astype(np.float32)
        dist_out = distance_transform_edt(~stroke_mask).astype(np.float32)
        sdf_raw = dist_in - dist_out
        sdf_01 = (np.clip(sdf_raw / max_dist, -1.0, 1.0) + 1.0) / 2.0

    # Restore channel dimension
    if squeezed:
        sdf_01 = sdf_01[np.newaxis]  # (1,H,W)

    if is_tensor:
        return torch.from_numpy(sdf_01)
    return sdf_01


def sdf_to_smoothstep(
    sdf: torch.Tensor,
    filter_width_px: float = 1.0,
    max_dist: float = 10.0,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Convert an SDF image to antialiased grayscale using smoothstep.

    Unlike ``sdf_to_binary`` (hard threshold → {0, 1}), returns continuous
    [0, 1] values with smooth edge transitions of width ``filter_width_px``
    pixels. Based on the SDF text rendering technique by Green/Valve (2007).

    Args:
        sdf: SDF image in [0, 1] as produced by ``binary_image_to_sdf``.
        filter_width_px: Transition half-width in pixels. 1 px is typical for
            64 px glyphs; increase slightly for higher resolutions.
        max_dist: Normalisation distance used in ``binary_image_to_sdf``.
            Must match the ``max_dist`` used when the SDF was created.
        threshold: SDF value corresponding to the contour (default 0.5).

    Returns:
        Antialiased image in [0, 1] with stroke=0, background=1.
        Values are 0.0/1.0 beyond ±filter_width_px from the contour;
        only the edge band is interpolated.
    """
    # Normalised half-width in [0, 1] SDF space
    w = max(filter_width_px / (2.0 * max_dist), 1e-6)
    d = sdf - threshold
    t = torch.clamp((d + w) / (2.0 * w), 0.0, 1.0)
    stroke_alpha = t * t * (3.0 - 2.0 * t)   # smoothstep: 0=background, 1=stroke
    return 1.0 - stroke_alpha                  # stroke=0, background=1


def sdf_to_binary(
    sdf: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
    sharpness: float | None = None,
) -> np.ndarray | torch.Tensor:
    """Threshold an SDF image to recover a binary image.

    Args:
        sdf: SDF image in [0, 1] as produced by ``binary_image_to_sdf``.
        threshold: Pixels above this value become stroke (0), others become background (1).
                   The default 0.5 corresponds to the zero-crossing of the raw SDF.
        sharpness: If given, applies soft sigmoid conversion ``sigmoid((threshold - sdf) * sharpness)``
                   instead of a hard threshold. Values 20–50 give near-binary output with smooth edges.
                   Only supported for torch.Tensor inputs; ndarray always uses the hard threshold.

    Returns:
        Image in [0, 1] with the same shape and type as the input.
        Hard threshold (sharpness=None): binary {0.0, 1.0}.
        Soft conversion: continuous [0, 1] approximating the binary result.
    """
    if isinstance(sdf, torch.Tensor):
        if sharpness is not None:
            return torch.sigmoid((threshold - sdf) * sharpness)
        return (sdf <= threshold).float()
    sdf_arr = np.asarray(sdf, dtype=np.float32)
    return (sdf_arr <= threshold).astype(np.float32)
