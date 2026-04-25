import logging
from statistics import median_high
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pylidc as pl

logger = logging.getLogger(__name__)


def extract_nodules_for_patient(
    patient_id: str,
    orig_spacing: np.ndarray,
    target_spacing: float = 1.0,
) -> List[Dict]:
    """
    extrage toti nodulii adnotati pentru un pacient.
    """
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    if scan is None:
        logger.warning(f"{patient_id}: pacientul nu exista in baza de date pylidc.")
        return []

    nodule_clusters = scan.cluster_annotations(verbose=False)

    if not nodule_clusters:
        logger.debug(f"{patient_id}: 0 noduli adnotati.")
        return []

    # spacing-ul in ordinea axelor pylidc: [row=y, col=x, slice=z]
    # orig_spacing din dicom_loader este [z, y, x]
    pixel_spacing = orig_spacing[1]   # y si x spacing (egal)
    slice_spacing = orig_spacing[0]   # z spacing

    nodule_records = []
    for nodule_idx, cluster in enumerate(nodule_clusters):
        record = _process_nodule_cluster(
            cluster=cluster,
            scan=scan,
            nodule_idx=nodule_idx,
            patient_id=patient_id,
            pixel_spacing=pixel_spacing,
            slice_spacing=slice_spacing,
            target_spacing=target_spacing,
        )
        if record is not None:
            nodule_records.append(record)

    logger.debug(
        f"{patient_id}: {len(nodule_records)} noduli extrasi din {len(nodule_clusters)} clustere."
    )

    return nodule_records


def _process_nodule_cluster(
    cluster: List,
    scan,
    nodule_idx: int,
    patient_id: str,
    pixel_spacing: float,
    slice_spacing: float,
    target_spacing: float,
) -> Optional[Dict]:
    """
    proceseaza un cluster de adnotari (1-4 radiologi per nodul)
    """
    if not cluster:
        return None

    num_annotations = len(cluster)

    # centroid consens: media centroizilor din toate adnotarile din cluster
    centroids = np.array([ann.centroid for ann in cluster])  # shape (n, 3): (row, col, slice)
    centroid_orig = centroids.mean(axis=0)  # (i, j, k) = (row, col, slice_idx)

    centroid_orig_row   = float(centroid_orig[0])
    centroid_orig_col   = float(centroid_orig[1])
    centroid_orig_slice = float(centroid_orig[2])

    # mapare la spatiul resampled (1mm) ---
    # row și col: multiplicate cu pixel_spacing
    # slice: multiplicat cu slice_spacing
    centroid_resampled_row = centroid_orig_row * pixel_spacing / target_spacing
    centroid_resampled_col = centroid_orig_col * pixel_spacing / target_spacing
    centroid_resampled_slice = centroid_orig_slice * slice_spacing / target_spacing

    x_world, y_world, z_world = _compute_world_coordinates(
        scan, centroid_orig_row, centroid_orig_col, centroid_orig_slice, pixel_spacing
    )

    # atribute malignitate
    malignancy_ratings = [ann.malignancy for ann in cluster]
    malignancy_by_radiologist = {
        f"malignancy_r{i + 1}": malignancy_ratings[i] if i < len(malignancy_ratings) else None
        for i in range(4)
    }
    malignancy_consensus = float(median_high(malignancy_ratings))

    # label binar: >=4 = cancer, <=2 = benign, =3 = ambiguu
    if malignancy_consensus >= 4:
        is_cancer = True
    elif malignancy_consensus <= 2:
        is_cancer = False
    else:
        is_cancer = None  # rating 3 = ambiguu

    # atribute morfologice (mediane)
    morphological_attrs = _compute_morphological_attributes(cluster)

    # dimensiuni nodul
    diameters = [ann.diameter for ann in cluster if hasattr(ann, "diameter")]
    volumes = [ann.volume for ann in cluster if hasattr(ann, "volume")]
    surface_areas = [ann.surface_area for ann in cluster if hasattr(ann, "surface_area")]

    diameter_mm = float(np.mean(diameters)) if diameters else None
    volume_mm3 = float(np.mean(volumes)) if volumes else None
    surface_area_mm2 = float(np.mean(surface_areas)) if surface_areas else None

    return {
        "patient_id": patient_id,
        "nodule_idx": nodule_idx,
        "num_annotations": num_annotations,
        # coordonate in volumul original (voxeli)
        "centroid_orig_row": centroid_orig_row,
        "centroid_orig_col": centroid_orig_col,
        "centroid_orig_slice": centroid_orig_slice,
        # coordonate in volumul resampled la 1mm (voxeli = mm față de origine)
        "centroid_resampled_row": centroid_resampled_row,
        "centroid_resampled_col": centroid_resampled_col,
        "centroid_resampled_slice": centroid_resampled_slice,
        # coordonate in spatiul DICOM (mm absolut)
        "x_world_mm": x_world,
        "y_world_mm": y_world,
        "z_world_mm": z_world,
        # dimensiuni
        "diameter_mm": diameter_mm,
        "volume_mm3": volume_mm3,
        "surface_area_mm2": surface_area_mm2,
        # malignitate
        **malignancy_by_radiologist,
        "malignancy_consensus":      malignancy_consensus,
        "is_cancer":                 is_cancer,
        # atribute morfologice
        **morphological_attrs,
    }


def _compute_world_coordinates(
    scan,
    centroid_row: float,
    centroid_col: float,
    centroid_slice: float,
    pixel_spacing: float,
) -> tuple:
    """
    converteste coordonatele voxel in coordonate world DICOM (mm)
    """
    try:
        origin = scan.get_path_to_dicom_files()
        x_world = float(centroid_col) * pixel_spacing
        y_world = float(centroid_row) * pixel_spacing

        slice_idx = int(round(centroid_slice))
        if hasattr(scan, "slice_zvals") and len(scan.slice_zvals) > 0:
            slice_idx = max(0, min(slice_idx, len(scan.slice_zvals) - 1))
            z_world = float(scan.slice_zvals[slice_idx])
        else:
            z_world = float(centroid_slice) * float(scan.slice_spacing)

    except Exception:
        # fallback la coordonate din spacing
        x_world = float(centroid_col) * pixel_spacing
        y_world = float(centroid_row) * pixel_spacing
        z_world = float(centroid_slice) * float(scan.slice_spacing)

    return x_world, y_world, z_world


def _compute_morphological_attributes(cluster: List) -> Dict:
    """
    calculeaza medianele atributelor morfologice din toate adnotarile clusterului
    """
    def safe_median(attr_name):
        values = []
        for ann in cluster:
            val = getattr(ann, attr_name, None)
            if val is not None:
                values.append(val)
        return float(np.median(values)) if values else None

    return {
        "subtlety_consensus": safe_median("subtlety"),
        "sphericity_consensus": safe_median("sphericity"),
        "margin_consensus": safe_median("margin"),
        "lobulation_consensus": safe_median("lobulation"),
        "spiculation_consensus": safe_median("spiculation"),
        "texture_consensus": safe_median("texture"),
        "calcification_consensus": safe_median("calcification"),
    }


def build_nodules_dataframe(all_records: List[Dict]) -> pd.DataFrame:
    """
    construieste DataFrame-ul final din toate inregistrarile de noduli
    """
    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    # sortare si reset index
    df = df.sort_values(["patient_id", "nodule_idx"]).reset_index(drop=True)

    return df
