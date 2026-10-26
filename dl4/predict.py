"""Load a trained classifier and predict a food category from an image."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "model" / "food_classifier.keras"
DEFAULT_CLASSES = PROJECT_DIR / "model" / "classes.json"
IMAGE_SIZE = (224, 224)


class ModelUnavailableError(RuntimeError):
    """Raised when prediction is requested before a compatible model is trained."""


class FoodClassifier:
    def __init__(self, model: Any, classes: list[str]) -> None:
        if not classes:
            raise ModelUnavailableError("The model class list is empty.")
        self.model = model
        self.classes = classes

    def predict(self, image_path: str | Path) -> tuple[str, float]:
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize(IMAGE_SIZE)
                image_array = np.asarray(image, dtype=np.float32)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Could not read image: {exc}") from exc
        probabilities = np.asarray(self.model.predict(image_array[None, ...], verbose=0))[0]
        if probabilities.ndim != 1 or len(probabilities) != len(self.classes):
            raise ModelUnavailableError(
                "The classifier output does not match model/classes.json. Retrain the model."
            )
        predicted_index = int(np.argmax(probabilities))
        return self.classes[predicted_index], float(probabilities[predicted_index])


@lru_cache(maxsize=2)
def load_classifier(
    model_path: str | Path = DEFAULT_MODEL,
    classes_path: str | Path = DEFAULT_CLASSES,
) -> FoodClassifier:
    model_path = Path(model_path)
    classes_path = Path(classes_path)
    if not model_path.is_file() or not classes_path.is_file():
        raise ModelUnavailableError(
            "No trained model is installed. Follow the README training steps to create "
            "model/food_classifier.keras and model/classes.json."
        )
    try:
        classes = json.loads(classes_path.read_text(encoding="utf-8"))
        if not isinstance(classes, list) or not all(isinstance(name, str) for name in classes):
            raise ValueError("Class metadata must be a JSON list of food names.")
        import tensorflow as tf

        model = tf.keras.models.load_model(model_path)
    except (ImportError, OSError, json.JSONDecodeError, ValueError) as exc:
        raise ModelUnavailableError(f"Could not load the trained classifier: {exc}") from exc
    return FoodClassifier(model, classes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Image file to classify")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    args = parser.parse_args()
    food_name, confidence = load_classifier(args.model, args.classes).predict(args.image)
    print(f"Food: {food_name}")
    print(f"Confidence: {confidence:.2%}")


if __name__ == "__main__":
    main()
