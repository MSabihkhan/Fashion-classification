# src/evaluate.py
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def main(data_path: Path, model_path: Path, cm_path: Path, metrics_path: Path) -> None:
    data = np.load(data_path)
    x_test, y_test = data["x_test"], data["y_test"]

    model = keras.models.load_model(model_path)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

    # Predicted class = index of the highest probability
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    cm = confusion_matrix(y_test, y_pred)

    # Confusion matrix image
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        ax=ax, xticks_rotation=45, colorbar=False, cmap="Blues")
    ax.set_title("Fashion-MNIST confusion matrix (test set)")
    fig.tight_layout()
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    # Per-class accuracy = correct predictions / total samples of that class
    per_class = cm.diagonal() / cm.sum(axis=1)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "num_test_samples": int(len(y_test)),
        "per_class_accuracy": {name: float(a) for name, a in zip(CLASS_NAMES, per_class)},
        "confusion_matrix": cm.tolist(),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test loss: {test_loss:.4f}  Test accuracy: {test_acc:.4f}")
    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained model on the test set.")
    parser.add_argument("--data-path", type=Path,
                        default=ROOT / "data" / "processed" / "fashion_mnist_processed.npz")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models" / "model.h5")
    parser.add_argument("--cm-path", type=Path, default=ROOT / "models" / "confusion_matrix.png")
    parser.add_argument("--metrics-path", type=Path, default=ROOT / "metrics.json")
    args = parser.parse_args()
    main(args.data_path, args.model_path, args.cm_path, args.metrics_path)