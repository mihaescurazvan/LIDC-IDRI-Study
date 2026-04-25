import logging
from typing import Tuple

import numpy as np
import scipy.ndimage

logger = logging.getLogger(__name__)


def resample_volume(
    hu_volume: np.ndarray,
    orig_spacing: np.ndarray,
    target_spacing: float = 1.0,
    order: int = 1,
    mode: str = "nearest",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    resampling al volumului 3D la spacing de 1mm
    """
    orig_spacing = np.array(orig_spacing, dtype=np.float64)

    # factorii de zoom per axa: > 1 = upsample (z rar de obicei), < 1 = downsample
    zoom_factors = orig_spacing / target_spacing

    logger.debug(
        f"Resample: shape {hu_volume.shape} * zoom {zoom_factors.round(3)} "
        f"shape estimat: {_compute_new_shape(hu_volume.shape, zoom_factors)}"
    )

    resampled = scipy.ndimage.zoom(
        hu_volume,
        zoom=zoom_factors,
        order=order,
        mode=mode,
        prefilter=(order > 1),
    )

    # spacing-ul efectiv: orig_spacing / zoom_factori_efectivi
    # zoom-ul efectiv = raportul dimensiunilor reale
    effective_zoom = np.array(resampled.shape) / np.array(hu_volume.shape)
    new_spacing = orig_spacing / effective_zoom

    logger.debug(
        f"Resampling complet: {hu_volume.shape} -> {resampled.shape}, spacing efectiv: {new_spacing.round(3)} mm"
    )

    return resampled.astype(np.float32), new_spacing


def _compute_new_shape(orig_shape, zoom_factors) -> np.ndarray:
    return np.round(np.array(orig_shape) * zoom_factors).astype(int)


def compute_new_shape(
    orig_shape: Tuple,
    orig_spacing: np.ndarray,
    target_spacing: float = 1.0,
) -> np.ndarray:
    zoom_factors = np.array(orig_spacing) / target_spacing
    return _compute_new_shape(orig_shape, zoom_factors)


def map_coordinate_to_resampled(
    coord_orig: np.ndarray,
    orig_spacing: np.ndarray,
    target_spacing: float = 1.0,
) -> np.ndarray:
    """
    mapeaza coordonatele voxelilor din spatiul original in spatiul resampled.
    coord_resampled[i] = coord_orig[i] * orig_spacing[i] / target_spacing
    """
    coord_orig   = np.array(coord_orig, dtype=np.float64)
    orig_spacing = np.array(orig_spacing, dtype=np.float64)
    return coord_orig * orig_spacing / target_spacing
