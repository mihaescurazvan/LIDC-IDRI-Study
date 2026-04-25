import logging

import numpy as np

logger = logging.getLogger(__name__)

# intervalul valid pentru HU clinic
HU_ARTIFACT_THRESHOLD = -1500.0
HU_MAX_VALID          =  3071.0


def pixels_to_hu(
    pixel_volume: np.ndarray,
    slope: float,
    intercept: float,
) -> np.ndarray:
    """
    converteste volumul de pixeli raw la Hounsfield Units
    """
    hu_volume = pixel_volume.astype(np.float32) * slope + intercept

    # eliminam valorile de padding unde nu exista data reale
    hu_volume = np.clip(hu_volume, HU_ARTIFACT_THRESHOLD, HU_MAX_VALID)

    return hu_volume


def validate_hu_range(hu_volume: np.ndarray, patient_id: str = "") -> bool:
    """
    verifica ca valorile HU sunt in intervalul asteptat pentru un CT pulmonar
    """
    prefix = f"[{patient_id}] " if patient_id else ""
    is_valid = True

    vol_min = float(hu_volume.min())
    vol_max = float(hu_volume.max())

    # Minimul așteptat după clipping este HU_ARTIFACT_THRESHOLD (-1500)
    # Verificăm că nu există valori sub pragul de clip (ar indica o eroare)
    if vol_min < HU_ARTIFACT_THRESHOLD - 1:
        logger.warning(
            f"{prefix}HU minim neasteptat: {vol_min:.0f} "
        )
        is_valid = False

    if vol_max > 3200:
        logger.warning(
            f"{prefix}HU maxim neașteptat: {vol_max:.0f} "
        )
        is_valid = False

    return is_valid
