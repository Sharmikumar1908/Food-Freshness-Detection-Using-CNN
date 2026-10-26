"""Train and evaluate a MobileNetV2 food classifier.

Expected input layout:
dataset/
  train/<class_name>/*.jpg
  validation/<class_name>/*.jpg
  test/<class_name>/*.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


IMAGE_SIZE = (224, 224)
DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "dataset"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "model"
DEFAULT_EVALUATION_DIR = Path(__file__).resolve().parent / "evaluation"


def load_datasets(data_dir: Path, batch_size: int):
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    test_dir = data_dir / "test"
    for directory in (train_dir, validation_dir, test_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required dataset directory not found: {directory}")
    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if len(class_names) < 2:
        raise ValueError(f"At least two class folders are required under {train_dir}.")

    def make_dataset(directory: Path, shuffle: bool):
        return tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int",
            class_names=class_names,
            image_size=IMAGE_SIZE,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=42,
        )

    return (
        make_dataset(train_dir, True),
        make_dataset(validation_dir, False),
        make_dataset(test_dir, False),
        class_names,
    )


def build_model(number_of_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
        ],
        name="data_augmentation",
    )
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="food_image")
    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenet_normalization")(x)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax", name="food_class")(x)
    return tf.keras.Model(inputs, outputs, name="food_mobilenetv2"), backbone


def plot_history(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    metrics = (
        ("accuracy", "Accuracy", "accuracy.png"),
        ("loss", "Loss", "loss.png"),
    )
    for key, title, filename in metrics:
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(history.history[key], label=f"Training {title.lower()}")
        axis.plot(history.history[f"val_{key}"], label=f"Validation {title.lower()}")
        axis.set(title=f"Training and validation {title}", xlabel="Epoch", ylabel=title)
        axis.legend()
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)


def evaluate(
    model: tf.keras.Model,
    test_dataset: tf.data.Dataset,
    class_names: list[str],
    output_dir: Path,
) -> dict[str, float]:
    loss, accuracy = model.evaluate(test_dataset, verbose=1)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_dataset])
    probabilities = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix = confusion_matrix(true_labels, predicted_labels, labels=range(len(class_names)))
    figure_width = max(8, len(class_names) * 0.7)
    figure, axis = plt.subplots(figsize=(figure_width, figure_width * 0.8))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title="Test confusion matrix",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
    )
    axis.set_xticklabels(class_names, rotation=45, ha="right")
    axis.set_yticklabels(class_names)
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(figure)
    results = {"test_loss": float(loss), "test_accuracy": float(accuracy)}
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=DEFAULT_EVALUATION_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.fine_tune_epochs < 0 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive; fine-tune-epochs cannot be negative.")

    train_dataset, validation_dataset, test_dataset, class_names = load_datasets(
        args.data_dir, args.batch_size
    )
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    model, backbone = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, monitor="val_loss", factor=0.2),
    ]
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    histories = [history]

    if args.fine_tune_epochs:
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        fine_tune_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
        )
        histories.append(fine_tune_history)
    combined_history = {
        key: [value for training_history in histories for value in training_history.history[key]]
        for key in histories[0].history
    }
    combined = tf.keras.callbacks.History()
    combined.history = combined_history

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.evaluation_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "food_classifier.keras")
    (args.output_dir / "classes.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    plot_history(combined, args.evaluation_dir)
    results = evaluate(model, test_dataset, class_names, args.evaluation_dir)
    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Evaluation artifacts: {args.evaluation_dir}")


if __name__ == "__main__":
    main()
