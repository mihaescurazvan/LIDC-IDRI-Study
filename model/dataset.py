import os
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import model.config as cfg


def _load_volume(volumes_dir: str, patient_id: str) -> np.ndarray:
    """incarca volumul .npy pentru un pacient si il returneaza ca float32."""
    path = os.path.join(volumes_dir, f"{patient_id}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Volum lipsa: {path}")
    return np.load(path).astype(np.float32)


def _extract_patch(
    volume: np.ndarray,
    z: int,
    y: int,
    x: int,
    patch_size: int,
    fill_value: float = -1.0,
) -> np.ndarray:
    """
    extrage un crop 3D de dimensiune (patch_size, patch_size, patch_size) centrat pe coordonatele (z, y, x).
    """
    half = patch_size // 2
    nz, ny, nx = volume.shape

    patch = np.full((patch_size, patch_size, patch_size), fill_value, dtype=np.float32)

    # calculam limitele in spatiul volumului
    z0, z1 = z - half, z + half
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half

    # clamp la limitele volumului
    vz0 = max(0, z0); vz1 = min(nz, z1)
    vy0 = max(0, y0); vy1 = min(ny, y1)
    vx0 = max(0, x0); vx1 = min(nx, x1)

    # offset corespunzator in patch
    pz0 = vz0 - z0; pz1 = pz0 + (vz1 - vz0)
    py0 = vy0 - y0; py1 = py0 + (vy1 - vy0)
    px0 = vx0 - x0; px1 = px0 + (vx1 - vx0)

    patch[pz0:pz1, py0:py1, px0:px1] = volume[vz0:vz1, vy0:vy1, vx0:vx1]
    return patch


def _augment_patch(patch: np.ndarray, flip_prob: float, rotate_max_deg: float) -> np.ndarray:
    """
    Augmentare simpla pe patch 3D:
    - Flip aleator pe axele Z, Y, X (nu schimba label-ul pentru noduli rotunzi)
    - Rotatie aleatoare pe planul axial (axa Z)
    """
    # flip pe fiecare axa independent
    for axis in range(3):
        if random.random() < flip_prob:
            patch = np.flip(patch, axis=axis).copy()

    # rotatie pe planul axial (slice 2D, axa 0 = Z)
    if rotate_max_deg > 0:
        angle = random.uniform(-rotate_max_deg, rotate_max_deg)
        if abs(angle) > 1e-3:
            from scipy.ndimage import rotate as scipy_rotate
            patch = scipy_rotate(patch, angle, axes=(1, 2), reshape=False,
                                 mode='constant', cval=-1.0)

    return patch


class NodulePatchDataset(Dataset):
    """
    Parametri
    ----------
    nodules_csv : str
        Calea catre nodules.csv generat de preprocessing pipeline
    volumes_dir : str
        Directorul cu volumele .npy (float16, normalizate [-1, 1])
    patch_size : int
        Dimensiunea cubului extras (default: 64)
    augment : bool
        Daca True, aplica augmentari (flip, rotatie) - folosit la training
    exclude_ambiguous : bool
        Daca True, exclude nodulii cu is_cancer=None (malignancy=3)
    patient_ids : list, optional
        Daca specificat, foloseste doar nodulii din acesti pacienti (pentru split)
    """

    def __init__(
        self,
        nodules_csv: str = cfg.NODULES_CSV,
        volumes_dir: str = cfg.VOLUMES_DIR,
        patch_size: int = cfg.PATCH_SIZE,
        augment: bool = False,
        exclude_ambiguous: bool = cfg.EXCLUDE_AMBIGUOUS,
        patient_ids: Optional[List[str]] = None,
    ):
        self.volumes_dir = volumes_dir
        self.patch_size = patch_size
        self.augment = augment

        df = pd.read_csv(nodules_csv)

        if exclude_ambiguous:
            df = df[df['is_cancer'].notna()].copy()

        if patient_ids is not None:
            df = df[df['patient_id'].isin(patient_ids)].copy()

        df = df.reset_index(drop=True)
        self.nodules = df

        # Pre-extrage toate patch-urile la initializare (evita incarcarea volumelor intregi in memorie la antrenare)
        self._patches = []
        self._labels  = []
        volume_cache: dict = {}
        for _, row in df.iterrows():
            patient_id = row['patient_id']
            if patient_id not in volume_cache:
                volume_cache[patient_id] = _load_volume(volumes_dir, patient_id)
            volume = volume_cache[patient_id]

            nz, ny, nx = volume.shape
            z = int(np.clip(round(row['centroid_resampled_slice']), 0, nz - 1))
            y = int(np.clip(round(row['centroid_resampled_row']),   0, ny - 1))
            x = int(np.clip(round(row['centroid_resampled_col']),   0, nx - 1))

            patch = _extract_patch(volume, z, y, x, patch_size, cfg.PATCH_FILL_VALUE)
            self._patches.append(patch)
            self._labels.append(float(row['is_cancer']))

        # elibereaza volumele din memorie
        del volume_cache

    def __len__(self) -> int:
        return len(self.nodules)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        patch = self._patches[idx].copy()

        if self.augment:
            patch = _augment_patch(patch, cfg.FLIP_PROB, cfg.ROTATE_MAX_DEG)

        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        label = torch.tensor(self._labels[idx], dtype=torch.float32)

        return patch_tensor, label

    def get_labels(self) -> List[int]:
        """Returneaza lista de label-uri (0/1) pentru toti nodulii."""
        return [int(v) for v in self._labels]

    def summary(self) -> str:
        n_cancer = int(self.nodules['is_cancer'].sum())
        n_benign = len(self.nodules) - n_cancer
        n_patients = self.nodules['patient_id'].nunique()
        return (f"NodulePatchDataset: {len(self.nodules)} noduli "
                f"({n_cancer} cancer, {n_benign} benign) "
                f"din {n_patients} pacienti | patch={self.patch_size}^3")


def make_train_val_split(
    nodules_csv: str = cfg.NODULES_CSV,
    val_fraction: float = cfg.VAL_FRACTION,
    random_seed: int = cfg.RANDOM_SEED,
    exclude_ambiguous: bool = cfg.EXCLUDE_AMBIGUOUS,
) -> Tuple[List[str], List[str]]:
    """
    Imparte pacientii in train/val la nivel de pacient (nu de nodul),
    pastrând proportia claselor cat mai echilibrata.

    Returneaza (train_patient_ids, val_patient_ids).
    """
    df = pd.read_csv(nodules_csv)
    if exclude_ambiguous:
        df = df[df['is_cancer'].notna()]

    # Grupeaza pe pacient si calculeaza fractiunea de cancer per pacient
    patient_stats = df.groupby('patient_id')['is_cancer'].agg(
        lambda x: x.astype(float).mean()
    ).reset_index()
    patient_stats.columns = ['patient_id', 'cancer_fraction']

    rng = np.random.RandomState(random_seed)
    patient_ids = patient_stats['patient_id'].tolist()
    rng.shuffle(patient_ids)

    n_val = max(1, round(len(patient_ids) * val_fraction))
    val_ids   = patient_ids[:n_val]
    train_ids = patient_ids[n_val:]

    return train_ids, val_ids
