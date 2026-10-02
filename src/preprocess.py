# src/preprocess.py
import argparse
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]


def main(raw_path: Path, out_dir: Path, params_path: Path) -> None:
    with open(params_path) as f:
        params = yaml.safe_load(f)["preprocess"]

    data = np.load(raw_path)
    x_train, y_train = data["x_train"], data["y_train"]
    x_test, y_test = data["x_test"], data["y_test"]

    # Normalize to [0, 1]
    x_train = x_train.astype("float32") / 127.5 - 1.0
    x_test = x_test.astype("float32") / 127.5 - 1.0

    # Validation split from the training data
    x_tr, x_val, y_tr, y_val = train_test_split(
        x_train, y_train,
        test_size=params["test_size"],
        random_state=params["seed"],
        stratify=y_train,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "fashion_mnist_processed.npz",
        x_train=x_tr, y_train=y_tr,
        x_val=x_val, y_val=y_val,
        x_test=x_test, y_test=y_test,
    )

    print("train:", x_tr.shape, "val:", x_val.shape, "test:", x_test.shape)
    print("pixel range:", x_tr.min(), "to", x_tr.max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize and split Fashion-MNIST.")
    parser.add_argument("--raw-path", type=Path, default=ROOT / "data" / "raw" / "fashion_mnist.npz")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    main(args.raw_path, args.out_dir, args.params)