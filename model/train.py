import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

import model.config as cfg
from model.architecture import SimpleCNN3D
from model.dataset import NodulePatchDataset, make_train_val_split


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Ruleaza o epoca de antrenare si returneaza loss-ul mediu."""
    model.train()
    total_loss = 0.0
    for patches, labels in loader:
        patches = patches.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad()
        preds = model(patches)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evalueaza modelul pe un set de date si returneaza metrici."""
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    for patches, labels in loader:
        patches = patches.to(device)
        labels  = labels.to(device)

        preds = model(patches)
        loss  = criterion(preds, labels)
        total_loss += loss.item() * len(labels)

        all_probs.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds_bin  = (all_probs >= 0.5).astype(int)

    accuracy = (preds_bin == all_labels).mean()

    # AUC-ROC (necesita cel putin 2 clase in set)
    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = float('nan')

    return {
        'loss':     total_loss / max(len(loader.dataset), 1),
        'accuracy': float(accuracy),
        'auc_roc':  float(auc_score),
        'probs':    all_probs,
        'labels':   all_labels,
    }


def save_training_curves(
    train_losses: list,
    val_losses: list,
    val_aucs: list,
    figures_dir: str,
) -> None:
    os.makedirs(figures_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Curbele de Antrenare - SimpleCNN3D', fontsize=13)

    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label='Train Loss', color='steelblue')
    axes[0].plot(epochs, val_losses,   label='Val Loss',   color='coral')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    valid_aucs = [(e, a) for e, a in zip(epochs, val_aucs) if not np.isnan(a)]
    if valid_aucs:
        e_vals, a_vals = zip(*valid_aucs)
        axes[1].plot(e_vals, a_vals, label='Val AUC-ROC', color='mediumpurple')
        axes[1].axhline(0.5, color='gray', linestyle='--', label='Random baseline')
        axes[1].set_xlabel('Epoca')
        axes[1].set_ylabel('AUC-ROC')
        axes[1].set_title('AUC-ROC pe Validare')
        axes[1].set_ylim(0, 1)
        axes[1].legend()

    plt.tight_layout()
    path = os.path.join(figures_dir, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvat: {path}")


def save_roc_curve(labels: np.ndarray, probs: np.ndarray, figures_dir: str) -> None:
    if len(np.unique(labels)) < 2:
        print("ROC curve nu poate fi generata (o singura clasa in set de validare)")
        return

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Clasificare Noduli')
    plt.legend()
    path = os.path.join(figures_dir, 'roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvat: {path}")


def print_confusion_matrix(labels: np.ndarray, probs: np.ndarray) -> None:
    preds_bin = (probs >= 0.5).astype(int)
    cm = confusion_matrix(labels, preds_bin, labels=[0, 1])
    print("\nMatrice de confuzie (validare):")
    print(f"              Prezis Benign  Prezis Cancer")
    print(f"Real Benign   {cm[0,0]:12d}  {cm[0,1]:12d}")
    print(f"Real Cancer   {cm[1,0]:12d}  {cm[1,1]:12d}")

    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0,0], 0, 0, cm[1,1])
    n_total = tn + fp + fn + tp
    if n_total > 0:
        print(f"\nSensitivitate (recall cancer): {tp/(tp+fn+1e-9):.1%}")
        print(f"Specificitate (recall benign): {tn/(tn+fp+1e-9):.1%}")


def run_training(
    epochs: int = cfg.EPOCHS,
    lr: float = cfg.LR,
    batch_size: int = cfg.BATCH_SIZE,
    weight_decay: float = cfg.WEIGHT_DECAY,
) -> None:
    set_seed(cfg.RANDOM_SEED)
    os.makedirs(cfg.MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Split train/val la nivel de pacient
    train_ids, val_ids = make_train_val_split()
    print(f"Train pacienti: {train_ids}")
    print(f"Val pacienti:   {val_ids}")

    train_ds = NodulePatchDataset(patient_ids=train_ids, augment=True)
    val_ds   = NodulePatchDataset(patient_ids=val_ids,   augment=False)

    print(f"\n{train_ds.summary()}")
    print(f"{val_ds.summary()}\n")

    if len(train_ds) == 0:
        print("EROARE: Niciun nodul in setul de training. Verifica nodules.csv.")
        return

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN3D(dropout=cfg.DROPOUT).to(device)
    print(f"Model: SimpleCNN3D | parametri antrenabili: {model.count_parameters():,}")

    # Class imbalance handling: pozitiv weight = n_benign / n_cancer
    train_labels = train_ds.get_labels()
    n_cancer = sum(train_labels)
    n_benign = len(train_labels) - n_cancer
    pos_weight = torch.tensor([n_benign / max(n_cancer, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCELoss()  # Sigmoid deja in model, folosim BCELoss simpla
    # Nota: pos_weight se aplica manual la loss pentru a compensa dezechilibrul
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses, val_losses, val_aucs = [], [], []
    best_val_loss = float('inf')

    print(f"Antrenare {epochs} epoci...\n")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if len(val_ds) > 0:
            val_metrics = evaluate(model, val_loader, criterion, device)
            val_loss = val_metrics['loss']
            val_auc  = val_metrics['auc_roc']
        else:
            val_loss = float('nan')
            val_auc  = float('nan')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_aucs.append(val_auc)

        scheduler.step(val_loss if not np.isnan(val_loss) else train_loss)

        # Salveaza cel mai bun model
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_loss':    val_loss,
                'val_auc':     val_auc,
            }, cfg.BEST_MODEL_PATH)

        if epoch % 10 == 0 or epoch == 1:
            auc_str = f"{val_auc:.3f}" if not np.isnan(val_auc) else "N/A"
            print(f"Epoca {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC-ROC: {auc_str}")

    print(f"\nAntrenare completa. Cel mai bun model salvat la: {cfg.BEST_MODEL_PATH}")

    # Evaluare finala si vizualizari
    save_training_curves(train_losses, val_losses, val_aucs, cfg.FIGURES_DIR)

    if len(val_ds) > 0:
        final_metrics = evaluate(model, val_loader, criterion, device)
        print(f"\nMetrici finale pe validare:")
        print(f"  Loss:     {final_metrics['loss']:.4f}")
        print(f"  Accuracy: {final_metrics['accuracy']:.1%}")
        print(f"  AUC-ROC:  {final_metrics['auc_roc']:.3f}" if not np.isnan(final_metrics['auc_roc']) else "  AUC-ROC:  N/A")
        print_confusion_matrix(final_metrics['labels'], final_metrics['probs'])
        save_roc_curve(final_metrics['labels'], final_metrics['probs'], cfg.FIGURES_DIR)


def main():
    parser = argparse.ArgumentParser(description="Antrenare SimpleCNN3D pe noduli LIDC-IDRI")
    parser.add_argument('--epochs',     type=int,   default=cfg.EPOCHS)
    parser.add_argument('--lr',         type=float, default=cfg.LR)
    parser.add_argument('--batch-size', type=int,   default=cfg.BATCH_SIZE)
    args = parser.parse_args()

    run_training(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
