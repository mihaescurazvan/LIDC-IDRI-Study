# preprocessing/

Pipeline de preprocesare pentru setul de date **LIDC-IDRI** (1010 scanări CT pulmonare). Transformă seriile DICOM brute în volume 3D normalizate și extrage adnotările nodulilor în format CSV, gata pentru antrenarea modelului.

## Structura modulelor

| Fișier | Rol |
|---|---|
| `config.py` | Toți parametrii configurabili (paths, HU windowing, resampling, segmentare) |
| `pipeline.py` | Orchestrează cei 5 pași și scrie logurile CSV |
| `dicom_loader.py` | Citire serie DICOM cu `pydicom`, extrage pixel array și metadata |
| `hu_converter.py` | Conversie pixel → Hounsfield Units (slope/intercept din DICOM header) |
| `resampler.py` | Resampling izotropic la 1 mm/voxel cu `scipy.ndimage.zoom` |
| `lung_segmentor.py` | Segmentare plămân prin K-means + morfologie 3D; mascarea zonelor extrapulmonare |
| `normalizer.py` | HU windowing `[-1000, 400]` și normalizare liniară la `[-1, 1]` |
| `annotation_extractor.py` | Extrage centroizii și scorurile nodulilor din adnotările pylidc |
| `utils.py` | Helpers: setup directoare, logging, citire/scriere CSV |
| `setup_pylidc.py` | Configurare inițială `~/.pylidcrc` |

## Pașii pipeline-ului

```
DICOM brut
   │
   ├─ [1] dicom_loader    → pixel volume + metadata (spacing, slope, intercept)
   ├─ [2] hu_converter    → HU volume (float32)
   ├─ [3] resampler       → HU volume izotropic 1mm/voxel
   ├─ [4] lung_segmentor  → aplicare mască pulmonară (fill=-1000 HU în afară)
   ├─ [5] normalizer      → windowing [-1000, 400] HU → normalizare [-1, 1], float16
   └─ [6] annotation_extractor → centroizi noduli în spațiul resampled + label cancer
```

## Output

```
preprocessing_output/
├── volumes/            # <patient_id>.npy  – float16, shape (Z, Y, X)
├── lung_masks/         # <patient_id>.npy  – bool array (opțional)
└── metadata/
    ├── nodules.csv     # un rând per nodul: patient_id, centroid_*, is_cancer
    ├── patients.csv    # un rând per pacient: status procesare
    └── processing_log.csv
```

### Coloane `nodules.csv`

| Coloană | Descriere |
|---|---|
| `patient_id` | ID pacient LIDC-IDRI |
| `centroid_resampled_slice` | Coordonata Z în volumul resampled (mm → voxel) |
| `centroid_resampled_row` | Coordonata Y |
| `centroid_resampled_col` | Coordonata X |
| `is_cancer` | `True` (malignancy ≥ 4), `False` (≤ 2), `None` (= 3, ambiguu) |

## Instalare dependențe

```bash
pip install -r preprocessing/requirements.txt
```

### Configurare pylidc

`pylidc` are nevoie de un fișier `~/.pylidcrc` care indică calea spre datele LIDC-IDRI:

```bash
python preprocessing/setup_pylidc.py
```

## Rulare

```bash
# Toți cei 1010 pacienți
python -m preprocessing.pipeline

# Subset de test (primii N pacienți)
python -m preprocessing.pipeline --max-patients 5
```

Pipeline-ul suportă **resume**: dacă un volum `.npy` există deja pe disk și `SKIP_EXISTING = True` în `config.py`, pacientul este sărit.

## Parametri cheie (`config.py`)

| Parametru | Valoare implicită | Descriere |
|---|---|---|
| `TARGET_SPACING_MM` | `1.0` | Spacing izotropic țintă după resampling |
| `HU_MIN / HU_MAX` | `-1000 / 400` | Fereastră pulmonară pentru windowing |
| `NORM_MIN / NORM_MAX` | `-1.0 / 1.0` | Intervalul de normalizare al output-ului |
| `DTYPE_OUT` | `float16` | Economisește ~50% față de float32 |
| `SEG_DILATION_RADIUS` | `10` px | Include nodulii juxtapleurali la 1mm/px |
| `MALIGNANCY_CANCER_THRESHOLD` | `4` | Rating ≥ 4 → etichetat cancer |
| `MAX_PATIENTS` | `None` (toți) | Limită pentru rulări de test |
| `SKIP_EXISTING` | `True` | Activează resume-ul între rulări |
