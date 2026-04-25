import logging

import numpy as np
import scipy.ndimage
from sklearn.cluster import KMeans
from skimage import measure, morphology

logger = logging.getLogger(__name__)


def segment_lungs(
    hu_volume: np.ndarray,
    dilation_radius: int = 10,
    closing_radius: int = 5,
    min_component_size: int = 1000,
) -> np.ndarray:
    """
    segmenteaza plamanii dintr-un volum CT 3D si returneaza masca
    """
    num_slices = hu_volume.shape[0]
    slice_masks = []

    for z in range(num_slices):
        mask_2d = _segment_slice_2d(
            hu_volume[z],
            dilation_radius=dilation_radius,
            min_component_size=min_component_size,
        )
        slice_masks.append(mask_2d)

    # stack 2D masks -> 3D mask
    mask_3d = np.stack(slice_masks, axis=0)  # (Z, H, W)

    # cleanup 3D: umple goluri intre slice-uri consecutive
    if closing_radius > 0:
        ball = morphology.ball(closing_radius)
        mask_3d = morphology.closing(mask_3d, ball)

    # fill holes 3D complete
    mask_3d = scipy.ndimage.binary_fill_holes(mask_3d)

    logger.debug(
        f"Segmentare: {mask_3d.sum() / mask_3d.size:.1%} din volum clasificat ca plaman"
    )

    return mask_3d.astype(bool)


def _segment_slice_2d(
    slice_hu: np.ndarray,
    dilation_radius: int = 10,
    min_component_size: int = 1000,
) -> np.ndarray:
    h, w = slice_hu.shape

    # normalizare pentru K-means
    center = slice_hu[100:400, 100:400]
    mean = center.mean()
    std  = max(center.std(), 1e-6)  # evita impartirea la 0
    normalized = (slice_hu - mean) / std

    # K-means pe zona centrala
    flat_center = normalized[100:400, 100:400].reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, n_init=3, random_state=42)
    kmeans.fit(flat_center)
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)

    # masca de aer: valori sub prag (intunecate = aer)
    air_mask = (normalized < threshold).astype(np.uint8)

    # eliminam zona de la baza imaginii (tabla scanner, ultimele ~10% randuri)
    air_mask[int(h * 0.9):, :] = 0

    labeled = measure.label(air_mask, connectivity=2)
    regions  = measure.regionprops(labeled)
    if not regions:
        # slice fara continut (ex. la capetele volumului): returnam mască goală
        return np.zeros((h, w), dtype=bool)

    # eliminarea background-ului (aerul din afara pacientului)
    # background-ul atinge colțurile imaginii
    corner_labels = set([
        labeled[0, 0], labeled[0, -1],
        labeled[-1, 0], labeled[-1, -1],
    ])
    corner_labels.discard(0)

    # candidatii pentru plăman: regiuni de aer care nu ating colturile si sunt suficient de mari
    lung_mask = np.zeros((h, w), dtype=bool)
    for region in regions:
        if region.label not in corner_labels and region.area >= min_component_size:
            lung_mask[labeled == region.label] = True

    # daca nu am găsit candidati de plaman valid, returnam mască goala
    if not lung_mask.any():
        return np.zeros((h, w), dtype=bool)

    # fill holes per componenta (nodulii si vasele creeaza gauri)
    lung_mask = scipy.ndimage.binary_fill_holes(lung_mask)

    # dilatare pentru a include nodulii juxtapleurali
    # nodulii pot fi partial in afara mastii dacă sunt lipiti de peretele pleural
    if dilation_radius > 0:
        disk = morphology.disk(dilation_radius)
        lung_mask = morphology.dilation(lung_mask, disk)

    return lung_mask


def apply_lung_mask(
    hu_volume: np.ndarray,
    lung_mask: np.ndarray,
    fill_value: float = -1000.0,
) -> np.ndarray:
    """
    aplica masca de plaman pe volumul HU.
    zonele din afara plamanului sunt inlocuite cu fill_value (-1000 HU = aer).
    """
    masked = hu_volume.copy()
    masked[~lung_mask] = fill_value
    return masked
