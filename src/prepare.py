# src/download_data.py
import argparse
from pathlib import Path

import numpy as np
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]


def main(out_dir: Path) -> None:
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fashion_mnist.npz"

    np.savez_compressed(
        out_path,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test,
    )

    print(f"Saved to {out_path}")
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Fashion-MNIST and save raw arrays.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()
    main(args.out_dir)