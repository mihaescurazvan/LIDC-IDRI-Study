"""
Script to extract morphological features for all nodules using pylidc.
Run this LOCALLY (requires DICOM files + pylidc configured).
Output: nodule_morphological_features.csv  (upload to Kaggle dataset 'lidc-idri-metadata')

Usage:
    python extract_morphological_features.py
"""

import os
import sys
import numpy as np
import pandas as pd
from statistics import median_high

import pylidc as pl

META_CSV    = 'LIDC-IDRI-Preprocessing/data/Meta/meta_info.csv'
OUTPUT_CSV  = 'nodule_morphological_features.csv'


def safe_median(cluster, attr):
    vals = [getattr(ann, attr, None) for ann in cluster]
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def extract_for_patient(patient_id_int):
    pid_str = f'LIDC-IDRI-{patient_id_int:04d}'
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid_str).first()
    if scan is None:
        return []

    clusters = scan.cluster_annotations(verbose=False)
    if not clusters:
        return []

    pixel_spacing = scan.pixel_spacing
    slice_spacing = scan.slice_spacing

    records = []
    for nid, cluster in enumerate(clusters):
        if not cluster:
            continue

        centroids = np.array([ann.centroid for ann in cluster])
        centroid  = centroids.mean(axis=0)

        mal_ratings = [ann.malignancy for ann in cluster]
        mal_consensus = float(median_high(mal_ratings))
        if mal_consensus >= 4:
            is_cancer = True
        elif mal_consensus <= 2:
            is_cancer = False
        else:
            is_cancer = None

        diameters    = [ann.diameter     for ann in cluster if hasattr(ann, 'diameter')]
        volumes      = [ann.volume       for ann in cluster if hasattr(ann, 'volume')]
        surf_areas   = [ann.surface_area for ann in cluster if hasattr(ann, 'surface_area')]

        records.append({
            'patient_id':              patient_id_int,
            'nodule_idx':              nid,
            'num_annotations':         len(cluster),
            'centroid_orig_row':       float(centroid[0]),
            'centroid_orig_col':       float(centroid[1]),
            'centroid_orig_slice':     float(centroid[2]),
            'diameter_mm':             float(np.mean(diameters))   if diameters  else None,
            'volume_mm3':              float(np.mean(volumes))     if volumes    else None,
            'surface_area_mm2':        float(np.mean(surf_areas))  if surf_areas else None,
            'malignancy_consensus':    mal_consensus,
            'is_cancer':               is_cancer,
            'subtlety_consensus':      safe_median(cluster, 'subtlety'),
            'sphericity_consensus':    safe_median(cluster, 'sphericity'),
            'margin_consensus':        safe_median(cluster, 'margin'),
            'lobulation_consensus':    safe_median(cluster, 'lobulation'),
            'spiculation_consensus':   safe_median(cluster, 'spiculation'),
            'texture_consensus':       safe_median(cluster, 'texture'),
            'calcification_consensus': safe_median(cluster, 'calcification'),
        })
    return records


def main():
    meta = pd.read_csv(META_CSV)
    patient_ids = sorted(meta['patient_id'].unique().tolist())
    print(f'Extracting morphological features for {len(patient_ids)} patients...')

    all_records = []
    for i, pid in enumerate(patient_ids):
        records = extract_for_patient(int(pid))
        all_records.extend(records)
        if (i + 1) % 100 == 0:
            print(f'  {i+1}/{len(patient_ids)} patients processed, {len(all_records)} nodules so far')

    df = pd.DataFrame(all_records)
    df = df.sort_values(['patient_id', 'nodule_idx']).reset_index(drop=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'\nSaved {len(df)} nodule records to {OUTPUT_CSV}')
    print(df.head())


if __name__ == '__main__':
    main()
