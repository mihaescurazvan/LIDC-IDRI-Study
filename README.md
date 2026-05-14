# LIDC-IDRI Study — Lung Nodule Malignancy Classification

Bachelor's thesis project on the LIDC-IDRI dataset. The study combines a U-Net /
U-Net++ segmentation backbone with a CNN classifier and structured metadata, and
culminates in **DCMA-Net**, a dual cross-attention model fusing image features with
morphological attributes. The full pipeline goes from raw DICOM scans to a calibrated
malignancy probability, with an ablation across seven model variants.

## Models

The ablation covers seven configurations, each one adding a component on top of the previous:

| Tag    | Model                          | Components                                   |
|--------|--------------------------------|----------------------------------------------|
| A      | CNN baseline                   | image only, no segmentation, no metadata     |
| B      | Seg + Cls (U-Net)              | + segmentation mask as a second channel      |
| C      | Seg + Cls + Meta (U-Net)       | + metadata branch (MLP fusion)               |
| D      | Seg + Cls + Meta (U-Net++)     | U-Net swapped for nested U-Net               |
| D-NL   | D, no-leak metadata            | drops the diagnostic radiomic features       |
| E      | DCMA-Net (full meta)           | ECA fusion, dual cross-attention, deep sup.  |
| F      | DCMA-Net (no-leak meta)        | DCMA-Net trained without leaky features      |

Segmentation backbones live in [unet_model.py](unet_model.py), [unet_parts.py](unet_parts.py)
(U-Net) and [Nested_Unet.py](Nested_Unet.py) (U-Net++). All seven classification models
are defined and trained inside [lidc-idri-classification-final.ipynb](lidc-idri-classification-final.ipynb).

## Repository layout

```
LIDC-IDRI-Study/
├── LIDC-IDRI-Preprocessing/      # DICOM → .npy pipeline (pylidc-based)
│   ├── prepare_dataset.py        # main preprocessing script
│   ├── config_file_create.py     # generates lung.conf
│   ├── utils.py                  # lung segmentation utilities
│   ├── notebook/make_label.ipynb # train/val/test split + label cleanup
│   ├── requirements.txt
│   └── README.md
├── lidc-idri-classification-final.ipynb       # main training + evaluation notebook
├── lidc-idri-classification-400epochs.ipynb   # long-run training experiment
├── lidc-idri-classification-60epochs.ipynb    # short-run training experiment
├── lidc-idri-classification-new-models.ipynb  # DCMA variants exploration
├── unet_model.py, unet_parts.py               # U-Net segmentation backbone
├── Nested_Unet.py                             # U-Net++ segmentation backbone
├── extract_morphological_features.py          # pylidc → nodule_morphological_features.csv
├── meta.csv                                   # per-slice metadata + split
├── nodule_morphological_features.csv          # extracted nodule-level features
├── lidc-idri-nodule-counts-6-23-2015.xlsx     # official LIDC nodule counts
└── tcia-diagnosis-data-2012-04-20.xls         # TCIA diagnosis reference
```

> **Note.** Generated artifacts (trained `.pth` checkpoints, figures, evaluation CSVs)
> and the raw LIDC-IDRI DICOM dump are intentionally excluded from this repository —
> see [.gitignore](.gitignore). They are reproduced by running the notebooks below.

## Dataset

The project uses the [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)
public dataset (1010 CT scans, multi-radiologist nodule annotations). Malignancy labels
follow the **median-high consensus** rule across up to four radiologist ratings:
ratings ≥ 4 → cancer, ≤ 2 → benign, 3 → ambiguous (excluded).

## Pipeline

### 1. Preprocessing — DICOM to `.npy`

```bash
cd LIDC-IDRI-Preprocessing
pip install -r requirements.txt
python config_file_create.py        # generates lung.conf
python prepare_dataset.py           # extracts nodule patches + masks
```

Output:
- `data/Image/LIDC-IDRI-XXXX/*.npy` — 512×512 nodule slices
- `data/Mask/LIDC-IDRI-XXXX/*.npy`  — corresponding binary masks
- `data/Clean/`                     — nodule-free lung slices (held-out test)
- `data/Meta/meta_info.csv`         — per-slice metadata + train/val/test split

The train/val/test split and label cleanup are handled by
[LIDC-IDRI-Preprocessing/notebook/make_label.ipynb](LIDC-IDRI-Preprocessing/notebook/make_label.ipynb).
See [LIDC-IDRI-Preprocessing/README.md](LIDC-IDRI-Preprocessing/README.md) for details.

### 2. Morphological features

```bash
python extract_morphological_features.py
```

Produces [nodule_morphological_features.csv](nodule_morphological_features.csv) with
per-nodule consensus values for diameter, volume, surface area, subtlety, sphericity,
margin, lobulation, spiculation, texture, calcification.

### 3. Training & evaluation

The full training, ablation, statistical tests, calibration, Grad-CAM analysis and SHAP
explanations are in [lidc-idri-classification-final.ipynb](lidc-idri-classification-final.ipynb).
The notebook is self-contained: it defines all seven models, trains them, and produces every
artifact (checkpoints, figures, evaluation CSVs).

Earlier / exploratory notebooks:
- [lidc-idri-classification-60epochs.ipynb](lidc-idri-classification-60epochs.ipynb) — short-run sanity check
- [lidc-idri-classification-400epochs.ipynb](lidc-idri-classification-400epochs.ipynb) — long-run training
- [lidc-idri-classification-new-models.ipynb](lidc-idri-classification-new-models.ipynb) — DCMA-Net variants

## Results

Best validation metrics from the ablation:

| Model | AUC-ROC    | Accuracy | Sensitivity | Specificity | F1     | PR-AUC |
|-------|------------|----------|-------------|-------------|--------|--------|
| A     | 0.8792     | 0.8172   | 0.7642      | 0.8478      | 0.7535 | 0.8350 |
| B     | 0.9028     | 0.8517   | 0.7547      | 0.9076      | 0.7882 | 0.8819 |
| C     | 0.9482     | 0.8828   | 0.8113      | 0.9239      | 0.8350 | 0.9338 |
| **D** | **0.9501** | 0.9138   | 0.8113      | 0.9728      | 0.8731 | **0.9386** |
| D-NL  | 0.9239     | 0.8828   | 0.8396      | 0.9076      | 0.8396 | 0.9158 |
| E     | 0.9286     | 0.8310   | 0.8302      | 0.8315      | 0.7822 | 0.9199 |
| F     | 0.9172     | 0.9138   | 0.7830      | 0.9891      | 0.8691 | 0.9134 |

Statistical significance is reported via DeLong and McNemar tests; calibration via
reliability diagrams and Brier scores; interpretability via Grad-CAM (with IoU against
ground-truth masks) and SHAP on the metadata features. All of these are produced by
the main notebook.

## Reproducing the results

```bash
# 1. preprocess (requires the LIDC-IDRI DICOM dump)
cd LIDC-IDRI-Preprocessing && python prepare_dataset.py && cd ..

# 2. extract morphological features
python extract_morphological_features.py

# 3. open the main notebook and run all cells
jupyter notebook lidc-idri-classification-final.ipynb
```

The notebook fixes random seeds and writes every checkpoint, figure and CSV locally
(these outputs are git-ignored).