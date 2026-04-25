"""
Configurare centralizată pentru pipeline-ul de preprocesare LIDC-IDRI.
Toți parametrii se modifică aici, nu în fișierele individuale.
"""

import os

# ---------------------------------------------------------------------------
# Paths - Date de intrare
# ---------------------------------------------------------------------------

LIDC_DATA_ROOT = "/home/razvan/licenta/LIDC_Data/manifest-1600709154662/LIDC-IDRI"

# ---------------------------------------------------------------------------
# Paths - Date de ieșire
# ---------------------------------------------------------------------------

OUTPUT_ROOT  = "/home/razvan/licenta/preprocessing_output"
VOLUMES_DIR  = os.path.join(OUTPUT_ROOT, "volumes")
MASKS_DIR    = os.path.join(OUTPUT_ROOT, "lung_masks")
METADATA_DIR = os.path.join(OUTPUT_ROOT, "metadata")
FIGURES_DIR  = os.path.join(OUTPUT_ROOT, "figures")

NODULES_CSV  = os.path.join(METADATA_DIR, "nodules.csv")
PATIENTS_CSV = os.path.join(METADATA_DIR, "patients.csv")
PROC_LOG_CSV = os.path.join(METADATA_DIR, "processing_log.csv")

# ---------------------------------------------------------------------------
# Parametri resampla
# ---------------------------------------------------------------------------

# Toate volumele vor fi resampleate la spacing izotropic de 1mm în toate axele
TARGET_SPACING_MM = 1.0

# Ordinul interpolației scipy.ndimage.zoom:
#   0 = nearest neighbor, 1 = trilinear (recomandat pentru CT), 3 = cubic
ZOOM_ORDER = 1
ZOOM_MODE  = "nearest"  # modul de padding la marginile volumului

# ---------------------------------------------------------------------------
# Parametri HU windowing
# ---------------------------------------------------------------------------

# Fereastra pulmonară standard: evidențiază nodulii și structurile pulmonare
HU_MIN = -1000.0   # aer (fundal plaman)
HU_MAX =  400.0    # tesut moale / perete pleural

# Valoarea cu care se umple zona din afara plămanului după mascare
HU_FILL_VALUE = -1000.0

# Clip pentru eliminarea artifact-urilor de padding scanner (-2048 la FOV boundary)
HU_CLIP_MIN = -1500.0
HU_CLIP_MAX =  3071.0

# ---------------------------------------------------------------------------
# Parametri normalizare
# ---------------------------------------------------------------------------

NORM_MIN = -1.0
NORM_MAX =  1.0

# Tipul de date pentru output (float16 economiseste ~50% fata de float32)
# float16 este suficient pentru range-ul [-1, 1] cu precizie ~0.001
DTYPE_OUT = "float16"

# ---------------------------------------------------------------------------
# Parametri segmentare plaman
# ---------------------------------------------------------------------------

# Numărul de clustere K-means (2 = aer vs. țesut)
SEG_N_CLUSTERS = 2

# Zona centrală a slice-ului folosită pentru K-means (evită tabla CT la margini)
SEG_KMEANS_ROW_SLICE = slice(100, 400)
SEG_KMEANS_COL_SLICE = slice(100, 400)

# Dilatare finală (pixeli) pentru a include nodulii juxtapleurali
# La 1mm/pixel, 10px = 10mm - suficient pentru noduli lipiti de peretele pleural
SEG_DILATION_RADIUS = 10

# Raza structuring element pentru binary_closing 3D (uneste slice-urile)
SEG_CLOSING_RADIUS = 5

# Numărul minim de voxeli pentru o componentă conexă validă (filtrare zgomot)
SEG_MIN_COMPONENT_SIZE = 1000

# ---------------------------------------------------------------------------
# Parametri pylidc / adnotări
# ---------------------------------------------------------------------------

# Toleranța de clustering pentru pylidc (implicit 0.1 mm)
PYLIDC_CLUSTER_TOLERANCE = 0.1

# Rating malignitate - praguri pentru eticheta binară
MALIGNANCY_CANCER_THRESHOLD    = 4  # >= 4 → cancer (True)
MALIGNANCY_BENIGN_THRESHOLD    = 2  # <= 2 → benign (False)
# == 3 → ambiguous (None) - exclus din clasificare binară

# ---------------------------------------------------------------------------
# Control procesare
# ---------------------------------------------------------------------------

# Numărul maxim de pacienți de procesat.
# None = procesează toți cei 1010 pacienți.
# Setează la un număr mic (ex. 5) pentru testarea rapidă a pipeline-ului.
MAX_PATIENTS = None

# Dacă True, skip pacienții care deja au fișierul .npy de output (resume)
SKIP_EXISTING = True

# Dacă True, salvează și măștile de plămân (bool arrays)
SAVE_LUNG_MASKS = True

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FILE = os.path.join(OUTPUT_ROOT, "pipeline.log")
