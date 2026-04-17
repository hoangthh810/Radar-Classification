import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================
def plot_and_save_cm(
    y_true, y_pred, class_names,
    save_path: str = "confusion_matrix.png",
    title: str = "Confusion Matrix",
) -> None:

    """Plot and save a heatmap confusion matrix.

    Args:
        y_true:      Ground-truth labels.
        y_pred:      Predicted labels.
        class_names: List of class name strings.
        save_path:   Output file path.
        title:       Plot title.
    """

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.savefig(save_path)
    print(f"Confusion Matrix đã được lưu tại: {save_path}")
    plt.close()


def plot_training_results(history: dict, save_dir) -> None:
    """Plot and save all training curves: loss, accuracy, precision, recall, F1.

    Args:
        history:  Dict with keys ``train_loss``, ``val_loss``, ``train_acc``,
                  ``val_acc``, ``train_precision``, ``train_recall``,
                  ``train_f1``, ``val_precision``, ``val_recall``, ``val_f1``.
        save_dir: Directory to write PNG files.
    """
    save_dir = Path(save_dir)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    plt.plot(epochs, history["val_loss"],   "r-o", label="Val Loss")
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epochs"); plt.ylabel("Loss")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png"); plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], "g-s", label="Train Accuracy")
    plt.plot(epochs, history["val_acc"],   "m-s", label="Val Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.xlabel("Epochs"); plt.ylabel("Accuracy")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / "accuracy_curve.png"); plt.close()

    # Precision curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_precision"], "c-o", label="Train Precision")
    plt.plot(epochs, history["val_precision"],   "c-x", label="Val Precision")
    plt.title("Train vs Validation Precision")
    plt.xlabel("Epochs"); plt.ylabel("Precision")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / "precision_curve.png"); plt.close()

    # Recall curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_recall"], "y-o", label="Train Recall")
    plt.plot(epochs, history["val_recall"],   "y-x", label="Val Recall")
    plt.title("Train vs Validation Recall")
    plt.xlabel("Epochs"); plt.ylabel("Recall")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / "recall_curve.png"); plt.close()

    # F1 curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_f1"], "k-o", label="Train F1")
    plt.plot(epochs, history["val_f1"],   "k-x", label="Val F1")
    plt.title("Train vs Validation F1 Score")
    plt.xlabel("Epochs"); plt.ylabel("F1 Score")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(save_dir / "f1_curve.png"); plt.close()