"""Flask web application for food classification and calorie estimates."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

from predict import ModelUnavailableError, load_classifier


BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.environ.get("FOOD_DATABASE", BASE_DIR / "database" / "food.db"))
UPLOAD_FOLDER = Path(os.environ.get("FOOD_UPLOAD_FOLDER", BASE_DIR / "static" / "uploads"))
NUTRITION_PATH = Path(os.environ.get("FOOD_NUTRITION_FILE", BASE_DIR / "nutrition.json"))
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_IMAGE_BYTES = 8 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Open a SQLite connection with rows addressable by column name."""
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    with get_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                food_name TEXT NOT NULL,
                confidence REAL NOT NULL,
                serving_size REAL NOT NULL CHECK (serving_size > 0),
                estimated_calories REAL NOT NULL,
                prediction_time TEXT NOT NULL
            )
            """
        )


def load_nutrition() -> dict[str, dict[str, Any]]:
    try:
        with NUTRITION_PATH.open(encoding="utf-8") as nutrition_file:
            data = json.load(nutrition_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load nutrition data from {NUTRITION_PATH}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Nutrition data must be a JSON object keyed by food name.")
    return data


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_image(file_storage: Any) -> str:
    filename = secure_filename(file_storage.filename or "")
    if not filename or not allowed_file(filename):
        raise ValueError("Choose a JPG, JPEG, PNG, or WebP image.")
    file_storage.stream.seek(0, os.SEEK_END)
    if file_storage.stream.tell() > MAX_IMAGE_BYTES:
        raise ValueError("Image must be 8 MB or smaller.")
    file_storage.stream.seek(0)
    try:
        with Image.open(file_storage.stream) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("The uploaded file is not a supported image.")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc
    file_storage.stream.seek(0)
    return filename


def serving_value(raw_value: Any) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Serving quantity must be a number greater than zero.") from exc
    if not 0 < value <= 100:
        raise ValueError("Serving quantity must be greater than 0 and no more than 100.")
    return value


def predict_and_save(file_storage: Any, servings_raw: Any) -> tuple[int, dict[str, Any]]:
    filename = validate_image(file_storage)
    servings = serving_value(servings_raw)
    image_name = f"{uuid.uuid4().hex}_{filename}"
    image_path = UPLOAD_FOLDER / image_name
    file_storage.save(image_path)
    try:
        food_name, confidence = load_classifier().predict(image_path)
        nutrition = load_nutrition()
        if food_name not in nutrition:
            raise RuntimeError(
                f"No nutrition entry is configured for '{food_name}'. Add it to nutrition.json."
            )
        food_nutrition = nutrition[food_name]
        calories = float(food_nutrition["calories_per_serving"]) * servings
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with get_db() as connection:
            cursor = connection.execute(
                """
                INSERT INTO predictions
                    (food_name, confidence, serving_size, estimated_calories, prediction_time)
                VALUES (?, ?, ?, ?, ?)
                """,
                (food_name, confidence, servings, calories, timestamp),
            )
            prediction_id = int(cursor.lastrowid)
        return prediction_id, {
            "food_name": food_name,
            "confidence": confidence,
            "serving_size": servings,
            "estimated_calories": calories,
            "prediction_time": timestamp,
            "image_url": url_for("static", filename=f"uploads/{image_name}"),
            "nutrition": food_nutrition,
        }
    except Exception:
        image_path.unlink(missing_ok=True)
        raise


def get_prediction(prediction_id: int) -> dict[str, Any] | None:
    with get_db() as connection:
        row = connection.execute(
            "SELECT * FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
    return dict(row) if row else None


@app.errorhandler(413)
def image_too_large(_: Exception):
    message = "Image must be 8 MB or smaller."
    if request.path.startswith("/api/"):
        return jsonify({"error": message}), 413
    return render_template("index.html", error=message), 413


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/predict")
def api_predict():
    uploaded_file = request.files.get("image")
    if not uploaded_file or not uploaded_file.filename:
        return jsonify({"error": "Select an image before submitting."}), 400
    try:
        prediction_id, result = predict_and_save(
            uploaded_file, request.form.get("serving_size", "1")
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ModelUnavailableError as exc:
        return jsonify({"error": str(exc)}), 503
    except (KeyError, RuntimeError, OSError) as exc:
        app.logger.exception("Prediction could not be completed")
        return jsonify({"error": str(exc)}), 500
    result["id"] = prediction_id
    result["result_url"] = url_for("result", prediction_id=prediction_id)
    return jsonify(result), 201


@app.get("/result/<int:prediction_id>")
def result(prediction_id: int):
    prediction = get_prediction(prediction_id)
    if prediction is None:
        return render_template("result.html", error="Prediction not found."), 404
    nutrition = load_nutrition()
    prediction["nutrition"] = nutrition.get(prediction["food_name"], {})
    prediction["image_url"] = request.args.get("image", "")
    return render_template("result.html", prediction=prediction)


@app.post("/api/predictions/<int:prediction_id>/serving")
def update_serving(prediction_id: int):
    payload = request.get_json(silent=True) or request.form
    try:
        servings = serving_value(payload.get("serving_size"))
        with get_db() as connection:
            row = connection.execute(
                "SELECT food_name FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
            if row is None:
                return jsonify({"error": "Prediction not found."}), 404
            nutrition = load_nutrition()
            calories_per_serving = float(nutrition[row["food_name"]]["calories_per_serving"])
            estimated_calories = calories_per_serving * servings
            connection.execute(
                """
                UPDATE predictions
                SET serving_size = ?, estimated_calories = ?
                WHERE id = ?
                """,
                (servings, estimated_calories, prediction_id),
            )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except (KeyError, RuntimeError, OSError) as exc:
        app.logger.exception("Serving estimate could not be updated")
        return jsonify({"error": str(exc)}), 500
    return jsonify(
        {
            "serving_size": servings,
            "estimated_calories": estimated_calories,
        }
    )


@app.get("/api/predictions")
def api_history():
    with get_db() as connection:
        rows = connection.execute(
            "SELECT * FROM predictions ORDER BY prediction_time DESC, id DESC"
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.get("/api/nutrition/<path:food_name>")
def api_nutrition(food_name: str):
    nutrition = load_nutrition()
    entry = nutrition.get(food_name)
    if entry is None:
        return jsonify({"error": "Nutrition information not found."}), 404
    return jsonify({"food_name": food_name, **entry})


@app.get("/history")
def history():
    with get_db() as connection:
        predictions = connection.execute(
            "SELECT * FROM predictions ORDER BY prediction_time DESC, id DESC"
        ).fetchall()
    return render_template("history.html", predictions=predictions)


@app.get("/about")
def about():
    return render_template("about.html")


init_db()

if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
