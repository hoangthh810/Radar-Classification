import argparse
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.autonotebook import tqdm

from models.sotacnn import SOTACNN
from utils.general import increment_path
from utils.plots import plot_and_save_cm

# =============================================================================
# CONSTANTS & GLOBALS
# =============================================================================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


# =============================================================================
# ARGUMENT PARSER
# =============================================================================
def parse_opt():
    """
    Parse command-line arguments for model evaluation.

    Returns:
        argparse.Namespace: Parsed arguments including paths, device specs,
                            and hyperparameters.
    """
    parser = argparse.ArgumentParser(description="Evaluate SOTA CNN for Radio Signal Classification")

    # Paths & Core Settings
    parser.add_argument("--data-dir", type=str, default="test", help="Path to evaluation dataset directory")
    parser.add_argument("--project", type=str, default=ROOT / "runs/val", help="Save results to project/name")
    parser.add_argument("--name", type=str, default="exp", help="Save results to project/name")
    parser.add_argument("--device", type=str, default="", help="CUDA device (e.g., '0' or '0,1,2,3') or 'cpu'")
    parser.add_argument("--workers", type=int, default=2, help="Maximum dataloader workers")
    parser.add_argument("--weights", type=str, default="runs/train/exp/weights/best.pt",
                        help="Path to trained weights file (.pt)")

    # Hyperparameters
    parser.add_argument("--imgsz", type=int, default=224, help="Inference image size (pixels)")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size")

    return parser.parse_args()


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================
def val(args):
    """
    Main evaluation pipeline: sets up the environment, loads data and model,
    runs inference, and generates evaluation metrics and plots.
    """
    # -------------------------------------------------------------------------
    # 1. Environment & Directory Setup
    # -------------------------------------------------------------------------
    # Configure hardware device
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Enable Automatic Mixed Precision (AMP) only if using CUDA
    use_amp = device.type == "cuda"

    # Create an auto-incrementing save directory (e.g., runs/val/exp, runs/val/exp2)
    save_dir = increment_path(Path(args.project) / args.name)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[INFO] Initializing Evaluation Pipeline...")
    print(f"       Device   : {device} (AMP Enabled: {use_amp})")
    print(f"       Weights  : {args.weights}")
    print(f"       Save Dir : {save_dir}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # -------------------------------------------------------------------------
    # Standard transform for evaluation (No augmentation)
    transform = transforms.Compose([
        transforms.Resize((args.imgsz, args.imgsz)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    test_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)
    classes = test_dataset.classes
    num_classes = len(classes)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Data doesn't need to be shuffled for evaluation
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    print(f"       Dataset  : {len(test_dataset)} samples across {num_classes} classes.")
    print(f"       Classes  : {classes}\n")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Loading
    # -------------------------------------------------------------------------
    model = SOTACNN(num_classes).to(device)
    try:
        # Load state dict with strict=True to ensure architecture matches
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        print("[SUCCESS] Model weights loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load model weights. Details: {e}")
        return

    # -------------------------------------------------------------------------
    # 4. Inference Loop
    # -------------------------------------------------------------------------
    model.eval()  # Set model to evaluation mode (disables dropout, fixes batchnorm)
    test_preds_final, test_targets_final = [], []

    # Disable gradient calculation for faster inference and reduced memory usage
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Inferencing", colour="magenta", unit="batch"):
            images, labels = images.to(device), labels.to(device)

            # Forward pass with Automatic Mixed Precision
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)

            # Extract the class with the highest logit score
            _, predicted = torch.max(outputs, 1)

            test_preds_final.extend(predicted.cpu().numpy())
            test_targets_final.extend(labels.cpu().numpy())

    # -------------------------------------------------------------------------
    # 5. Metrics Calculation & Export
    # -------------------------------------------------------------------------
    # Compute global metrics using macro averaging
    acc = accuracy_score(test_targets_final, test_preds_final)
    prec = precision_score(test_targets_final, test_preds_final, average="macro", zero_division=0)
    rec = recall_score(test_targets_final, test_preds_final, average="macro", zero_division=0)
    f1 = f1_score(test_targets_final, test_preds_final, average="macro", zero_division=0)

    print("\n" + "=" * 60)
    print(" SUMMARY: METRICS EVALUATION ")
    print("=" * 60)
    print(f" Global Accuracy  : {acc:.4f}")
    print(f" Macro Precision  : {prec:.4f}")
    print(f" Macro Recall     : {rec:.4f}")
    print(f" Macro F1-Score   : {f1:.4f}")
    print("=" * 60)

    # Export Confusion Matrix visualization
    plot_and_save_cm(
        test_targets_final, test_preds_final, class_names=classes,
        save_path=save_dir / "confusion_matrix.png",
        title="Confusion Matrix - Test Set",
    )
    print(f"[EXPORT] Confusion Matrix saved to: {save_dir.name}/confusion_matrix.png")

    # Export detailed Classification Report
    report = classification_report(
        test_targets_final, test_preds_final,
        target_names=classes, digits=4, zero_division=0
    )
    with open(save_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[EXPORT] Classification Report saved to: {save_dir.name}/classification_report.txt")

    print(f"\n[DONE] Evaluation completed successfully. Check '{save_dir}' for full outputs.\n")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    opt = parse_opt()
    val(opt)