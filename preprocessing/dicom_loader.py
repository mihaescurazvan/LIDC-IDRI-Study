import logging
from typing import Dict, List, Tuple

import numpy as np
import pylidc as pl

logger = logging.getLogger(__name__)


def load_dicom_series(patient_id: str) -> Tuple[np.ndarray, Dict]:
    """
    creeaza volumul CT-ului unui pacient + metadate
    """
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    if scan is None:
        raise ValueError(f"Pacientul {patient_id} nu exista in baza de date pylidc.")

    dicom_slices = scan.load_all_dicom_images(verbose=False)

    if not dicom_slices:
        raise RuntimeError(f"Nu s-au putut incarca imaginile DICOM pentru {patient_id}.")

    # sortare după poziția z a slice-urilor
    dicom_slices = _sort_slices_by_z(dicom_slices, patient_id)

    # validare dimensiuni
    _validate_slice_dimensions(dicom_slices, patient_id)

    # compunere volum
    pixel_volume = np.stack(
        [ds.pixel_array.astype(np.int16) for ds in dicom_slices],
        axis=0
    )

    # extragere metadata
    metadata = _extract_metadata(scan, dicom_slices)
    metadata["original_shape"] = pixel_volume.shape

    logger.debug(
        f"{patient_id}: loaded {pixel_volume.shape[0]} slices, "
        f"spacing z={metadata['slice_spacing_mm']:.3f}mm, "
        f"xy={metadata['pixel_spacing_mm']:.3f}mm"
    )

    return pixel_volume, metadata


def _sort_slices_by_z(
    dicom_slices: List, patient_id: str
) -> List:
    """
    sorteaza slice-urile DICOM după coordonata z din ImagePositionPatient
    """
    def get_z_position(ds):
        try:
            return float(ds.ImagePositionPatient[2])
        except (AttributeError, IndexError, TypeError):
            try:
                return float(ds.InstanceNumber)
            except (AttributeError, TypeError):
                return 0.0

    return sorted(dicom_slices, key=get_z_position)


def _validate_slice_dimensions(dicom_slices: List, patient_id: str) -> None:
    """
    verifica ca toate slice-urile au aceleasi dimensiuni
    """
    first_shape = dicom_slices[0].pixel_array.shape
    for i, ds in enumerate(dicom_slices[1:], start=1):
        if ds.pixel_array.shape != first_shape:
            logger.warning(
                f"{patient_id}: slice {i} are shape {ds.pixel_array.shape}, "
                f"diferit de primul slice {first_shape}."
            )


def _extract_metadata(scan, sorted_slices: List) -> Dict:
    first_ds = sorted_slices[0]

    # pixel spacing XY (mm/pixel)
    pixel_spacing_mm = float(scan.pixel_spacing)

    # slice spacing Z (mm între slice-uri consecutive)
    slice_spacing_mm = float(scan.slice_spacing)

    # fallback dacă slice_spacing este 0 sau invalid
    if slice_spacing_mm <= 0:
        z_positions = [float(ds.ImagePositionPatient[2]) for ds in sorted_slices
                       if hasattr(ds, "ImagePositionPatient")]
        if len(z_positions) > 1:
            diffs = np.diff(sorted(z_positions))
            slice_spacing_mm = float(np.median(np.abs(diffs)))
        else:
            slice_spacing_mm = 1.0
            logger.warning("Nu s-a putut determina slice spacing, folosit default 1.0mm")

    # rescale parametri pentru conversia HU
    rescale_slope = float(getattr(first_ds, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(first_ds, "RescaleIntercept", -1024.0))

    # pozitiile z ale tuturor slice-urilor
    z_positions = []
    for ds in sorted_slices:
        try:
            z_positions.append(float(ds.ImagePositionPatient[2]))
        except (AttributeError, IndexError, TypeError):
            z_positions.append(None)

    # originea volumului
    try:
        image_position = [float(v) for v in first_ds.ImagePositionPatient]
    except (AttributeError, TypeError):
        image_position = [0.0, 0.0, 0.0]

    return {
        "patient_id":          scan.patient_id,
        "series_instance_uid": scan.series_instance_uid,
        "study_instance_uid":  getattr(first_ds, "StudyInstanceUID", ""),
        "pixel_spacing_mm":    pixel_spacing_mm,
        "slice_spacing_mm":    slice_spacing_mm,
        "slice_thickness_mm":  float(getattr(first_ds, "SliceThickness", slice_spacing_mm)),
        "rescale_slope":       rescale_slope,
        "rescale_intercept":   rescale_intercept,
        "num_slices":          len(sorted_slices),
        "rows":                int(getattr(first_ds, "Rows", 512)),
        "cols":                int(getattr(first_ds, "Columns", 512)),
        "image_position":      image_position,
        "z_positions":         z_positions,
        "contrast_used":       bool(scan.contrast_used),
        # [z, y, x]
        "orig_spacing":        np.array([slice_spacing_mm, pixel_spacing_mm, pixel_spacing_mm]),
    }
