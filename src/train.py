# src/train.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]


def main(data_path: Path, model_dir: Path, params_path: Path) -> None:
    with open(params_path) as f:
        params = yaml.safe_load(f)["train"]

    data = np.load(data_path)
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]

    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(params["dense_units"], activation="relu"),
        keras.layers.Dropout(params["dropout_rate"]),
        keras.layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "model.h5")
    pd.DataFrame(history.history).to_csv(model_dir / "history.csv", index=False)

    print(f"Saved model to {model_dir / 'model.h5'}")
    print(f"Saved history to {model_dir / 'history.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Fashion-MNIST ANN.")
    parser.add_argument("--data-path", type=Path,
                        default=ROOT / "data" / "processed" / "fashion_mnist_processed.npz")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--params", type=Path, default=ROOT / "params.yaml")
    args = parser.parse_args()
    main(args.data_path, args.model_dir, args.params)