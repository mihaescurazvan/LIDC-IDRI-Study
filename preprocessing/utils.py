"""
Utilități comune folosite de toate modulele din pipeline.
"""

import os
import logging
import csv
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd


def setup_output_dirs(config) -> None:
    """Creează toate directoarele de output dacă nu există."""
    dirs = [
        config.VOLUMES_DIR,
        config.MASKS_DIR,
        config.METADATA_DIR,
        config.FIGURES_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_patient_list(dicom_root: str) -> List[str]:
    """
    Returnează lista sortată de ID-uri pacienți (LIDC-IDRI-XXXX).
    Filtrează fișierele non-director (ex. LICENSE).
    """
    entries = [
        e for e in os.listdir(dicom_root)
        if os.path.isdir(os.path.join(dicom_root, e))
        and e.startswith("LIDC-IDRI-")
    ]
    entries.sort()
    return entries


def load_processing_log(log_path: str) -> pd.DataFrame:
    """Încarcă logul de procesare sau returnează DataFrame gol dacă nu există."""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=[
            "patient_id", "status", "timestamp", "duration_sec",
            "orig_shape", "resampled_shape", "num_nodules",
            "error_msg", "peak_memory_mb",
        ])
    return pd.read_csv(log_path)


def append_to_log(log_path: str, record: Dict) -> None:
    """
    Adaugă un singur rând în fișierul CSV de log.
    Creează header-ul dacă fișierul nu există.
    """
    fieldnames = [
        "patient_id", "status", "timestamp", "duration_sec",
        "orig_shape", "resampled_shape", "num_nodules",
        "error_msg", "peak_memory_mb",
    ]
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        # Asigură că toate câmpurile sunt prezente
        row = {k: record.get(k, "") for k in fieldnames}
        writer.writerow(row)


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Configurează un logger cu handler la consolă (INFO) și opțional la fișier (DEBUG).
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # Evită adăugarea de handlere duplicate

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler consolă
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Handler fișier
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def memory_usage_mb() -> float:
    """Returnează consumul curent de RAM al procesului în MB."""
    try:
        import psutil
        import os as _os
        process = psutil.Process(_os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback pentru Linux fără psutil
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except Exception:
            pass
    return 0.0


def now_iso() -> str:
    """Returnează timestamp-ul curent în format ISO."""
    return datetime.now().isoformat(timespec="seconds")


def shape_to_str(shape) -> str:
    """Convertește un tuple de shape în string de forma 'ZxHxW'."""
    return "x".join(str(s) for s in shape)


def volume_output_path(volumes_dir: str, patient_id: str) -> str:
    """Returnează calea completă pentru fișierul .npy al unui pacient."""
    return os.path.join(volumes_dir, f"{patient_id}.npy")


def mask_output_path(masks_dir: str, patient_id: str) -> str:
    """Returnează calea completă pentru masca de plămân a unui pacient."""
    return os.path.join(masks_dir, f"{patient_id}.npy")
