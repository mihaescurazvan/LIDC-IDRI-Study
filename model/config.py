"""
Configurare centralizata pentru modelul de clasificare noduli pulmonari.
"""

import os

# ---------------------------------------------------------------------------
# Paths - preluate din configurarea preprocessing-ului
# ---------------------------------------------------------------------------

PROJECT_ROOT = "/home/razvan/licenta"

# Date de intrare (generate de preprocessing pipeline)
VOLUMES_DIR  = os.path.join(PROJECT_ROOT, "preprocessing_output", "volumes")
NODULES_CSV  = os.path.join(PROJECT_ROOT, "preprocessing_output", "metadata", "nodules.csv")

# Output model
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "model_output")
BEST_MODEL_PATH  = os.path.join(MODEL_OUTPUT_DIR, "best_model.pth")
FIGURES_DIR      = os.path.join(MODEL_OUTPUT_DIR, "figures")

# ---------------------------------------------------------------------------
# Parametri extractie patch-uri
# ---------------------------------------------------------------------------

# Dimensiunea crop-ului 3D centrat pe centroidul nodulului (voxeli la 1mm/voxel)
PATCH_SIZE = 64  # 64 x 64 x 64 mm

# Valoarea de umplere pentru padding cand centroidul e prea aproape de margine
PATCH_FILL_VALUE = -1.0  # corespunde aerului dupa normalizare [-1, 1]

# Excludem nodulii cu malignancy_consensus == 3 (ambigui, nedefiniti clinic)
EXCLUDE_AMBIGUOUS = True

# ---------------------------------------------------------------------------
# Parametri model
# ---------------------------------------------------------------------------

DROPOUT = 0.3

# ---------------------------------------------------------------------------
# Parametri antrenare
# ---------------------------------------------------------------------------

BATCH_SIZE = 16      # potrivit pentru 100+ pacienti
LR         = 1e-3    # Adam learning rate
EPOCHS     = 50
WEIGHT_DECAY = 1e-4  # L2 regularizare

# Fractiunea de date pentru validare (restul merg la training)
# ATENTIE: cu ~8 noduli, aceasta e doar pentru demonstrarea fluxului
VAL_FRACTION = 0.25

# Seed pentru reproductibilitate
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Augmentare date
# ---------------------------------------------------------------------------

# Probabilitate flip aleator pe fiecare axa (aplicat doar la training)
FLIP_PROB = 0.5

# Rotatie aleatoare in grade (aplicata pe planul axial, axa Z)
ROTATE_MAX_DEG = 15
