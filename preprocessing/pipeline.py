import argparse
import logging
import sys
import time
import traceback
from typing import List, Optional

import numpy as np

# importam config-ul si modulele pipeline-ului
import preprocessing.config as cfg
from preprocessing import (
    annotation_extractor,
    dicom_loader,
    hu_converter,
    lung_segmentor,
    normalizer,
    resampler,
    utils,
)

logger = logging.getLogger(__name__)


def process_patient(patient_id: str) -> dict:
    """
    proceseaza un singur pacient prin toti pasii pipeline-ului.
    """
    t_start = time.time()
    peak_memory_start = utils.memory_usage_mb()

    out_volume_path = utils.volume_output_path(cfg.VOLUMES_DIR, patient_id)
    out_mask_path   = utils.mask_output_path(cfg.MASKS_DIR, patient_id)

    # skip daca deja procesat
    if cfg.SKIP_EXISTING and _is_already_processed(patient_id):
        logger.info(f"[SKIP] {patient_id}: volum existent, sarit.")
        return _make_log_record(patient_id, "skipped", duration_sec=0)

    try:
        # pasul 1: citire DICOM
        logger.info(f"[1/5] {patient_id}: citire DICOM...")
        pixel_volume, metadata = dicom_loader.load_dicom_series(patient_id)
        orig_shape = pixel_volume.shape
        orig_spacing = metadata["orig_spacing"]  # [z, y, x] in mm

        # pasul 2: conversie HU
        logger.info(f"[2/5] {patient_id}: conversie HU...")
        hu_volume = hu_converter.pixels_to_hu(
            pixel_volume,
            slope=metadata["rescale_slope"],
            intercept=metadata["rescale_intercept"],
        )
        del pixel_volume

        hu_converter.validate_hu_range(hu_volume, patient_id)

        # pasul 3: resampling la 1mm izotropic
        logger.info(
            f"[3/5] {patient_id}: resampling {orig_shape} -> 1mm izotropic (spacing z={orig_spacing[0]:.2f}mm, xy={orig_spacing[1]:.2f}mm)..."
        )
        resampled_hu, new_spacing = resampler.resample_volume(
            hu_volume,
            orig_spacing=orig_spacing,
            target_spacing=cfg.TARGET_SPACING_MM,
            order=cfg.ZOOM_ORDER,
            mode=cfg.ZOOM_MODE,
        )
        del hu_volume
        resampled_shape = resampled_hu.shape
        logger.info(f"-> shape resampled: {resampled_shape}")

        # pasul 4: segmentare plaman
        logger.info(f"[4/5] {patient_id}: segmentare plaman ({resampled_shape[0]} slice-uri)...")
        lung_mask = lung_segmentor.segment_lungs(
            resampled_hu,
            dilation_radius=cfg.SEG_DILATION_RADIUS,
            closing_radius=cfg.SEG_CLOSING_RADIUS,
            min_component_size=cfg.SEG_MIN_COMPONENT_SIZE,
        )
        masked_hu = lung_segmentor.apply_lung_mask(
            resampled_hu,
            lung_mask,
            fill_value=cfg.HU_FILL_VALUE,
        )
        del resampled_hu

        # pasul 5 & 6: Windowing + Normalizare
        logger.info(f"[5/5] {patient_id}: windowing + normalizare...")
        final_volume = normalizer.preprocess_volume(
            masked_hu,
            hu_min=cfg.HU_MIN,
            hu_max=cfg.HU_MAX,
            out_min=cfg.NORM_MIN,
            out_max=cfg.NORM_MAX,
            dtype=cfg.DTYPE_OUT,
        )
        del masked_hu

        # salvare volum și masca
        np.save(out_volume_path, final_volume)
        logger.info(f"-> salvat: {out_volume_path} ({final_volume.nbytes / 1e6:.1f} MB)")
        del final_volume

        if cfg.SAVE_LUNG_MASKS:
            np.save(out_mask_path, lung_mask)
        del lung_mask

        # pasul 7: extragere adnotarile nodulilor
        nodule_records = annotation_extractor.extract_nodules_for_patient(
            patient_id=patient_id,
            orig_spacing=orig_spacing,
            target_spacing=cfg.TARGET_SPACING_MM,
        )
        num_nodules = len(nodule_records)
        logger.info(f"{num_nodules} noduli extrasi")

        duration_sec = time.time() - t_start
        peak_memory  = utils.memory_usage_mb() - peak_memory_start

        logger.info(
            f"[OK] {patient_id}: procesat in {duration_sec:.1f}s, peak RAM delta: {peak_memory:.0f}MB"
        )

        return _make_log_record(
            patient_id,
            status="success",
            duration_sec=duration_sec,
            orig_shape=orig_shape,
            resampled_shape=resampled_shape,
            num_nodules=num_nodules,
            peak_memory_mb=max(0, peak_memory),
            nodule_records=nodule_records,
        )

    except Exception as e:
        duration_sec = time.time() - t_start
        error_msg = f"{type(e).__name__}: {e}"
        logger.error(f"[FAIL] {patient_id}: {error_msg}")
        logger.debug(traceback.format_exc())

        return _make_log_record(
            patient_id,
            status="failed",
            duration_sec=duration_sec,
            error_msg=error_msg,
        )


def _is_already_processed(patient_id: str) -> bool:
    """verifica daca fisierul de output exista deja pe disk."""
    import os
    return os.path.exists(utils.volume_output_path(cfg.VOLUMES_DIR, patient_id))


def _make_log_record(
    patient_id: str,
    status: str,
    duration_sec: float = 0,
    orig_shape=None,
    resampled_shape=None,
    num_nodules: int = 0,
    error_msg: str = "",
    peak_memory_mb: float = 0,
    nodule_records: Optional[List] = None,
) -> dict:
    return {
        "patient_id":     patient_id,
        "status":         status,
        "timestamp":      utils.now_iso(),
        "duration_sec":   round(duration_sec, 2),
        "orig_shape":     utils.shape_to_str(orig_shape) if orig_shape else "",
        "resampled_shape": utils.shape_to_str(resampled_shape) if resampled_shape else "",
        "num_nodules":    num_nodules,
        "error_msg":      error_msg,
        "peak_memory_mb": round(peak_memory_mb, 1),
        "_nodule_records": nodule_records or [],
    }


def run_pipeline(max_patients: Optional[int] = None) -> None:
    """
    ruleaza pipeline-ul pentru toti pacientii (sau un subset).
    """
    utils.setup_output_dirs(cfg)
    logger = utils.setup_logger("pipeline", log_file=cfg.LOG_FILE)

    logger.info("=" * 60)
    logger.info("Pipeline de preprocesare LIDC-IDRI")
    logger.info("=" * 60)

    # lista pacientilor disponibili
    all_patients = utils.get_patient_list(cfg.LIDC_DATA_ROOT)
    logger.info(f"Pacienti gasiti în {cfg.LIDC_DATA_ROOT}: {len(all_patients)}")

    limit = max_patients if max_patients is not None else cfg.MAX_PATIENTS
    if limit is not None:
        all_patients = all_patients[:limit]
        logger.info(f"Procesam {len(all_patients)} pacienți (MAX_PATIENTS={limit})")
    else:
        logger.info(f"Procesam toti {len(all_patients)} pacientii")

    # identificam pacientii deja procesati (pentru resume)
    log_df = utils.load_processing_log(cfg.PROC_LOG_CSV)
    already_done = set(log_df[log_df["status"] == "success"]["patient_id"].tolist())
    if already_done:
        logger.info(f"{len(already_done)} pacienți deja procesati (vor fi skipuiti)")

    # incarca nodulii deja extrasi din rularile anterioare pentru pacientii ce vor fi skipuiti
    # (evita pierderea lor la re-run cu SKIP_EXISTING=True)
    import os
    import pandas as pd
    all_nodule_records = []
    if cfg.SKIP_EXISTING and os.path.exists(cfg.NODULES_CSV):
        existing_nodules_df = pd.read_csv(cfg.NODULES_CSV)
        # pastram doar nodulii pacientilor care NU vor fi reprocesati in aceasta sesiune
        existing_nodules_df = existing_nodules_df[
            existing_nodules_df['patient_id'].isin(already_done)
        ]
        all_nodule_records = existing_nodules_df.to_dict('records')
        logger.info(f"Incarcati {len(all_nodule_records)} noduli existenti din {cfg.NODULES_CSV}")

    # loop principal
    n_success = n_failed = n_skipped = 0

    try:
        from tqdm import tqdm
        patient_iter = tqdm(all_patients, desc="Pacienti", unit="pacient")
    except ImportError:
        patient_iter = all_patients

    for patient_id in patient_iter:
        result = process_patient(patient_id)

        # agregate nodulii pentru CSV-ul final
        all_nodule_records.extend(result.pop("_nodule_records", []))

        # scrie in log imediat
        utils.append_to_log(cfg.PROC_LOG_CSV, result)

        # statistici
        if result["status"] == "success":
            n_success += 1
        elif result["status"] == "failed":
            n_failed += 1
        else:
            n_skipped += 1

    logger.info(f"Procesare completa: {n_success} OK | {n_failed} esuate | {n_skipped} skipped")

    if all_nodule_records:
        nodules_df = annotation_extractor.build_nodules_dataframe(all_nodule_records)
        nodules_df.to_csv(cfg.NODULES_CSV, index=False)
        logger.info(f"Salvat {cfg.NODULES_CSV} ({len(nodules_df)} noduli)")
    else:
        logger.warning("Niciun nodul extras")
    log_df_final = utils.load_processing_log(cfg.PROC_LOG_CSV)
    log_df_final.to_csv(cfg.PATIENTS_CSV, index=False)
    logger.info(f"Salvat {cfg.PATIENTS_CSV}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline de preprocesare DICOM pentru LIDC-IDRI"
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Numarul maxim de pacienti de procesat",
    )
    args = parser.parse_args()

    run_pipeline(max_patients=args.max_patients)


if __name__ == "__main__":
    main()
