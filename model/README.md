# model/

Modul de antrenare pentru clasificarea binară a nodulilor pulmonari pe volumele 3D generate de `preprocessing/`. Implementează o rețea convoluțională 3D (**SimpleCNN3D**) cu antrenare pe patch-uri centrate pe nodul.

## Structura modulelor

| Fișier | Rol |
|---|---|
| `config.py` | Toți hiperparametrii și căile (paths preluate din output-ul preprocessing) |
| `architecture.py` | Definiția `SimpleCNN3D` – 3 blocuri Conv3D + Global Avg Pool + MLP |
| `dataset.py` | `NodulePatchDataset` – extrage patch-uri 3D din volume, augmentare |
| `train.py` | Loop de antrenare, evaluare, salvare model, generare grafice |

## Arhitectura modelului (`SimpleCNN3D`)

```
Input: [B, 1, 64, 64, 64]
   │
   ├─ ConvBlock3D(1  → 16)   → [B, 16, 32, 32, 32]   Conv3D + BN + ReLU + MaxPool
   ├─ ConvBlock3D(16 → 32)   → [B, 32, 16, 16, 16]
   ├─ ConvBlock3D(32 → 64)   → [B, 64,  8,  8,  8]
   │
   ├─ AdaptiveAvgPool3d(1)   → [B, 64, 1, 1, 1]
   ├─ Flatten                → [B, 64]
   ├─ Linear(64 → 32) + ReLU
   ├─ Dropout(0.3)
   ├─ Linear(32 → 1)
   └─ Sigmoid                → probabilitate cancer ∈ [0, 1]
```

## Date de intrare

Modulul citește direct output-ul pipeline-ului de preprocesare:

```
preprocessing_output/
├── volumes/        # <patient_id>.npy  – float16, normalizat [-1, 1]
└── metadata/
    └── nodules.csv # centroizi noduli + label is_cancer
```

Fiecare sample din dataset este un **patch 3D de 64×64×64 voxeli** centrat pe centroidul nodulului. Padding-ul pentru nodulii apropiați de marginea volumului este umplut cu `-1.0` (echivalentul aerului după normalizare).

### Split train/val

Split-ul se face **la nivel de pacient** (nu de nodul), pentru a evita data leakage. Fracțiunea de validare este configurabilă (`VAL_FRACTION = 0.25`).

### Augmentări (doar la training)

- Flip aleator independent pe axele Z, Y, X (`FLIP_PROB = 0.5`)
- Rotație aleatoare pe planul axial ±15° (`ROTATE_MAX_DEG = 15`)

## Output antrenare

```
model_output/
├── best_model.pth       # checkpoint cu cel mai mic val_loss
└── figures/
    ├── training_curves.png   # loss train/val + AUC-ROC per epocă
    └── roc_curve.png         # curba ROC pe setul de validare
```

### Formatul checkpoint-ului

```python
{
    'epoch':       int,
    'model_state': OrderedDict,   # model.state_dict()
    'val_loss':    float,
    'val_auc':     float,
}
```

## Rulare antrenare

```bash
# Cu parametrii din config.py
python -m model.train

# Cu argumente custom
python -m model.train --epochs 100 --lr 0.0005 --batch-size 8
```

## Parametri cheie (`config.py`)

| Parametru | Valoare implicită | Descriere |
|---|---|---|
| `PATCH_SIZE` | `64` | Dimensiunea cubului extras (voxeli) |
| `EXCLUDE_AMBIGUOUS` | `True` | Exclude nodulii cu `is_cancer=None` (malignancy=3) |
| `BATCH_SIZE` | `16` | |
| `LR` | `1e-3` | Learning rate Adam |
| `EPOCHS` | `50` | |
| `WEIGHT_DECAY` | `1e-4` | Regularizare L2 |
| `VAL_FRACTION` | `0.25` | Fracțiunea pacienților pentru validare |
| `DROPOUT` | `0.3` | Dropout în stratul fully-connected |
| `FLIP_PROB` | `0.5` | Probabilitate flip per axă la augmentare |
| `ROTATE_MAX_DEG` | `15` | Unghi maxim rotație axială la augmentare |
| `RANDOM_SEED` | `42` | Seed pentru reproductibilitate |

## Dependențe

```bash
pip install torch>=2.0.0 numpy pandas scikit-learn matplotlib scipy
```

(Incluse și în `preprocessing/requirements.txt`)
