import numpy as np

def apply_hu_window(
    hu_volume: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
) -> np.ndarray:
    """
    aplica fereastra HU (lung window) prin clipping.
    """
    return np.clip(hu_volume, hu_min, hu_max).astype(np.float32)


def normalize_to_range(
    windowed_volume: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    out_min: float = -1.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """
    normalizeaza linar din fereastra HU la intervalul [out_min, out_max].
    """
    hu_range = hu_max - hu_min
    out_range = out_max - out_min
    normalized = (windowed_volume - hu_min) / hu_range * out_range + out_min
    return normalized.astype(np.float32)


def cast_to_output_dtype(volume: np.ndarray, dtype: str = "float16") -> np.ndarray:
    """
    converteste volumul la tipul de date pentru salvare pe disk.
    """
    return volume.astype(dtype)


def preprocess_volume(
    hu_volume: np.ndarray,
    hu_min: float = -1000.0,
    hu_max: float = 400.0,
    out_min: float = -1.0,
    out_max: float = 1.0,
    dtype: str = "float16",
) -> np.ndarray:
    """
    functie compusă: aplica windowing, normalizare si cast in un singur pas.
    """
    windowed = apply_hu_window(hu_volume, hu_min, hu_max)
    normalized = normalize_to_range(windowed, hu_min, hu_max, out_min, out_max)
    return cast_to_output_dtype(normalized, dtype)
