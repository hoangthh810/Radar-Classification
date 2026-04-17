import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm.autonotebook import tqdm


from data.datasets import MapDataset
from models.sotacnn import SOTACNN
from utils.ema import EMA
from utils.general import increment_path, seed_everything, seed_worker
from utils.plots import plot_and_save_cm, plot_training_results

# =============================================================================
# CONSTANTS & GLOBALS
# =============================================================================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


# =============================================================================
# ARGUMENT PARSER
# =============================================================================
def parse_opt():
    """Parse parameters from command line."""
    parser = argparse.ArgumentParser(description="Train SOTA CNN for Radio Signal Classification")

    # Paths & Core Settings
    parser.add_argument("--data-dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--project", type=str, default=ROOT / "runs/train", help="Save results to project/name")
    parser.add_argument("--name", type=str, default="exp", help="Save results to project/name")
    parser.add_argument("--device", type=str, default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--workers", type=int, default=2, help="Max dataloader workers")

    # Hyperparameters
    parser.add_argument("--imgsz", type=int, default=224, help="Train/Val image size (pixels)")
    parser.add_argument("--epochs", type=int, default=40, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Total batch size")
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio (e.g., 0.8 for 80%)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for CosineAnnealing")
    parser.add_argument("--weight-decay", type=float, default=2e-4, help="Optimizer weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA decay rate")

    return parser.parse_args()


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def train(args):
    # -------------------------------------------------------------------------
    # 1. Setup Environments & Directories
    # -------------------------------------------------------------------------
    seed_everything(args.seed)

    # Device configuration
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    # Create save directories
    save_dir = increment_path(Path(args.project) / args.name)
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Save training arguments to YAML
    with open(save_dir / "opt.yaml", "w") as f:
        args_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        yaml.safe_dump(args_dict, f, sort_keys=False)

    print(f" Training Info:")
    print(f"   • Device: {device} (AMP: {use_amp})")
    print(f"   • Results saved to: {save_dir}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Augmentation
    # -------------------------------------------------------------------------
    train_transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.1, 3.3), value=0),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full_dataset = datasets.ImageFolder(root=args.data_dir)
    classes = full_dataset.classes
    num_classes = len(classes)

    # Dynamic Train/Val Split based on args
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    split_g = torch.Generator().manual_seed(args.seed)
    train_indices, val_indices = random_split(full_dataset, [train_size, val_size], generator=split_g)

    train_ds = MapDataset(train_indices, transform=train_transform)
    val_ds = MapDataset(val_indices, transform=val_transform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(args.seed),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
    )

    print(f"   • Classes ({num_classes}): {classes}")
    print(f"   • Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # -------------------------------------------------------------------------
    # 3. Model, Optimizer & Scheduler
    # -------------------------------------------------------------------------
    model = SOTACNN(num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   • Total Trainable Parameters: {total_params:,}")

    # Use arguments for hyperparameters
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ema = EMA(model, decay=args.ema_decay)

    history = {
        "train_loss": [], "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [],
        "val_loss": [], "val_acc": [], "val_precision": [], "val_recall": [], "val_f1": [],
    }
    best_val_acc = 0.0

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        # --- TRAINING ---
        model.train()
        train_preds, train_targets = [], []
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]", colour="cyan")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
            train_bar.set_postfix(loss=loss.item())

        epoch_train_loss = running_loss / len(train_loader)
        train_acc = (np.array(train_preds) == np.array(train_targets)).mean()
        train_precision = precision_score(train_targets, train_preds, average="macro", zero_division=0)
        train_recall = recall_score(train_targets, train_preds, average="macro", zero_division=0)
        train_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)

        scheduler.step()

        # -------------------------------------------------------------------------
        # VALIDATION (Using EMA Weights)
        # -------------------------------------------------------------------------
        ema.apply(model)
        try:
            model.eval()
            val_preds, val_targets = [], []
            val_running_loss = 0.0

            val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", colour="green")
            with torch.no_grad():
                for images, labels in val_bar:
                    images, labels = images.to(device), labels.to(device)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    val_running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
        finally:
            ema.restore(model)

        epoch_val_loss = val_running_loss / len(val_loader)
        val_acc = (np.array(val_preds) == np.array(val_targets)).mean()
        val_precision = precision_score(val_targets, val_preds, average="macro", zero_division=0)
        val_recall = recall_score(val_targets, val_preds, average="macro", zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        # --- LOGGING & SAVING ---
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(train_acc)
        history["train_precision"].append(train_precision)
        history["train_recall"].append(train_recall)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)

        pd.DataFrame(history).to_csv(save_dir / "results.csv", index_label="epoch")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} Summary ---")
        print(f"LR    | {current_lr:.7f}")
        print(
            f"TRAIN | Loss: {epoch_train_loss:.4f} | Acc: {train_acc:.4f} | P: {train_precision:.4f} | R: {train_recall:.4f} | F1: {train_f1:.4f}")
        print(
            f"VAL   | Loss: {epoch_val_loss:.4f} | Acc: {val_acc:.4f} | P: {val_precision:.4f} | R: {val_recall:.4f} | F1: {val_f1:.4f}")
        print("-" * 60)

        torch.save(model.state_dict(), weights_dir / "last.pt")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ema.apply(model)
            try:
                torch.save(model.state_dict(), weights_dir / "best.pt")
            finally:
                ema.restore(model)
            print(f" New best model saved! (Val Acc: {best_val_acc:.4f})")

    # -------------------------------------------------------------------------
    # 5. Final Evaluation & Plotting
    # -------------------------------------------------------------------------
    print("\nTraining completed. Generating plots and confusion matrices...")
    plot_training_results(history, save_dir)

    # Bake EMA weights permanently into model for final evaluation
    ema.apply_to_model(model)
    best_weight_path = weights_dir / "best.pt"
    if best_weight_path.exists():
        model.load_state_dict(torch.load(best_weight_path, weights_only=True))

    model.eval()

    # Train Confusion Matric
    train_preds_final, train_targets_final = [], []
    with torch.no_grad():
        for images, labels in tqdm(train_loader, desc="[Train CM]", colour="yellow"):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            train_preds_final.extend(predicted.cpu().numpy())
            train_targets_final.extend(labels.cpu().numpy())

    plot_and_save_cm(
        train_targets_final, train_preds_final, class_names=classes,
        save_path=save_dir / "train_confusion_matrix.png",
        title="Confusion Matrix - Train Set",
    )

    # Val Confusion Matric
    val_preds_final, val_targets_final = [], []
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="[Val CM]", colour="magenta"):
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_preds_final.extend(predicted.cpu().numpy())
            val_targets_final.extend(labels.cpu().numpy())

    plot_and_save_cm(
        val_targets_final, val_preds_final, class_names=classes,
        save_path=save_dir / "val_confusion_matrix.png",
        title="Confusion Matrix - Validation Set",
    )

    print(f" All done! Check '{save_dir}' for your results.")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    opt = parse_opt()
    train(opt)