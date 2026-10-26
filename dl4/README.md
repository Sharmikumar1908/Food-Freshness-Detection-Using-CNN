# Food Image Classification and Calorie Estimation System

FoodLens is a college-project web application that demonstrates transfer learning for food-image classification. It predicts one food category from a photo, reports the model's confidence, and calculates an **approximate calorie estimate** from a reference value and the serving quantity entered by the user.

> **Model and evaluation status:** This repository does not include trained weights or fabricated evaluation results. Before training, image predictions are unavailable. Train on a real dataset to generate the model, class list, test metrics, plots, confusion matrix, and classification report.
>
> **Nutrition limitation:** Calories and macronutrients are reference estimates for a standard serving. They are not measured from an image and are not medical advice.

## Problem statement

Food logging can be tedious, and a food photograph alone cannot reveal portion mass or ingredients. This project demonstrates an accessible image-classification workflow while keeping the calorie calculation transparent: the user supplies serving quantity and the application multiplies it by an approximate per-serving reference value.

## Objectives

- Prepare food images consistently and classify them with a transfer-learned MobileNetV2 model.
- Show the predicted category and its model confidence.
- Estimate calories from a configurable nutrition table and user-entered serving quantity.
- Provide a responsive image-upload experience, prediction history, and nutrition overview.
- Evaluate a trained model on a held-out test split and save reproducible evaluation artifacts.

## System architecture

```text
Browser (upload, preview, serving quantity)
            |
            v
Flask web app / JSON API
   |        |                 |
   |        +--> Validate image --> Resize RGB to 224x224
   |                                  |
   |                                  v
   |                       MobileNetV2 classifier (.keras)
   |                                  |
   +--> nutrition.json <--- food label + confidence
   |                                  |
   +--> SQLite history <-- serving count x kcal per serving
            |
            v
      Result / history pages

Training workflow (separate from Flask):
Dataset train/validation/test folders --> train.py --> model + class list
                                        --> evaluation plots + test report
```

The Keras model contains the training-only augmentation layers and the MobileNetV2 input normalization layer. Serving images are converted to RGB, resized to 224×224, and passed through the trained model. Training uses ImageNet weights, a custom softmax classification head, validation monitoring, and optional fine-tuning.

## Project structure

```text
.
├── app.py                     # Flask pages, APIs, SQLite persistence
├── train.py                   # Training, fine-tuning, and held-out evaluation
├── prepare_dataset.py         # Prepare labeled local images for training
├── predict.py                 # Reusable classifier and CLI prediction
├── nutrition.json             # Approximate serving nutrition reference
├── requirements.txt
├── README.md
├── dataset/                   # Add train/validation/test class folders locally
├── model/                     # Generated model and classes.json
├── evaluation/                # Generated plots and test metrics
├── database/                  # SQLite DB created on first app start
├── templates/                 # Home, result, history, about
└── static/
    ├── css/style.css
    ├── js/script.js
    ├── images/hero-food.svg
    └── uploads/                # Uploaded images (ignored by Git)
```

## Technologies

- Python 3.11 or newer
- TensorFlow/Keras with MobileNetV2 ImageNet weights
- Flask, SQLite, Pillow, NumPy
- scikit-learn and Matplotlib for evaluation
- HTML, CSS and JavaScript for the responsive interface

OpenCV and Pandas are included as useful image/data tooling for project extensions; the starter's core preprocessing uses Pillow and Keras.

## Dataset

The primary dataset for this project is [FooDD: Food Detection Dataset for Calorie Measurement Using Food Images](https://ieee-dataport.org/open-access/foodd-food-detection-dataset-calorie-measurement-using-food-images) from IEEE DataPort. Its landing page describes food images captured with different cameras and lighting conditions for food recognition and calorie/nutrition estimation. It also notes that the human thumb visible in each image is approximately 5 cm × 1.2 cm. The reference object can help motivate scale-aware research; this starter classifier does **not** use it to infer portion mass or exact calories.

Download the dataset from its source and follow the source's access terms. The archive's annotation and folder structure is not assumed here. Curate a CSV with the actual image labels from the downloaded data; do not infer labels from filenames unless the dataset documentation explicitly defines that convention. `prepare_dataset.py` copies labeled images into deterministic, class-stratified folder splits and rejects missing files, duplicate image paths, unsafe labels, and classes with fewer than three images:

Dataset reference: P. Pouladzadeh, A. Yassine, and S. Shirmohammadi, “FooDD: Food Detection Dataset for Calorie Measurement Using Food Images,” ICIAP 2015 Workshops, LNCS 9281, pp. 441–448, [doi:10.1007/978-3-319-23222-5_54](https://doi.org/10.1007/978-3-319-23222-5_54).

```text
annotations.csv:
image,label
images/food_001.jpg,Pizza
images/food_002.jpg,Idli
...
```

Run the preparation script (use the real local paths and labels from the FooDD download):

```powershell
python prepare_dataset.py --annotations .\foodd_annotations.csv --image-root .\foodd_download --output-dir .\dataset --seed 42
```

The prepared structure is:

```text
dataset/
├── train/
│   ├── Apple/
│   ├── Banana/
│   ├── Biryani/
│   ├── Burger/
│   ├── Dosa/
│   ├── Idli/
│   ├── Pizza/
│   ├── Rice/
│   ├── Sandwich/
│   └── Samosa/
├── validation/
│   └── <same class folders>
└── test/
    └── <same class folders>
```

The class-directory names are used as model labels and are generated directly from the CSV labels. For categories used by the application, add a corresponding entry (matching the label exactly) to `nutrition.json`. Each entry requires `serving`, `calories_per_serving`, `protein_g`, `carbohydrates_g`, and `fat_g`. If using Food-101 or another permitted dataset instead, convert its genuine labels to the same CSV format or prepare the train/validation/test folders directly.

## Installation

From the project directory, create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow installation and performance depend on your operating system and hardware. On Windows, current TensorFlow releases generally use CPU in native Windows; consult TensorFlow's official install guide for the compatible Python version and GPU setup (WSL2 may be needed for GPU support).

## Train and evaluate the model

1. Download FooDD under its source terms and prepare the dataset folders as described above. Inspect labels carefully and do not put near-duplicate images of the same food/scene in both training and test.
2. Train the classifier:

   ```powershell
   python train.py --data-dir dataset --epochs 10 --fine-tune-epochs 5 --batch-size 32
   ```

   MobileNetV2 ImageNet weights are downloaded by Keras on the first run if they are not cached. The script saves `model/food_classifier.keras` and `model/classes.json`.

3. Review the generated `evaluation/` files:

   - `accuracy.png`: training and validation accuracy over epochs
   - `loss.png`: training and validation loss over epochs
   - `confusion_matrix.png`: per-class held-out test predictions
   - `classification_report.txt`: per-class precision, recall, F1, and support
   - `metrics.json`: actual held-out test loss and accuracy from that training run

No metric is asserted in advance. A test accuracy is reported only after the script evaluates the saved-in-memory model on the supplied test split. For an independent final evaluation, avoid selecting hyperparameters based on the test data.

Optional paths and epoch parameters:

```powershell
python train.py --data-dir .\dataset --output-dir .\model --evaluation-dir .\evaluation --epochs 15 --fine-tune-epochs 5 --batch-size 16
```

## Run the Flask application

After installing requirements and training a model:

```powershell
python app.py
```

Visit `http://127.0.0.1:5000/`. The app creates the SQLite database and predictions table on startup. Upload one JPG, JPEG, PNG, or WebP image (up to 8 MB), enter the approximate number of servings, and submit. The result page shows the class prediction, confidence, approximate calories, and reference macronutrients. Change the serving quantity on the result page to update the estimate and saved history.

If the model has not been trained or its model files are missing, prediction returns a clear setup message rather than a placeholder classification. This is expected for a fresh checkout.

Useful configuration environment variables:

- `FOOD_DATABASE`: SQLite database file path
- `FOOD_UPLOAD_FOLDER`: image upload directory
- `FOOD_NUTRITION_FILE`: nutrition JSON path
- `FLASK_DEBUG=1`: enable Flask debug mode for local development only

## CLI prediction example

```powershell
python predict.py .\sample_food.jpg
```

Example output shape (values depend on your actual trained model and image):

```text
Food: <predicted class>
Confidence: <model probability>
```

The score is the model's probability for its top class, not a guarantee that the food is correctly identified.

## Routes and API

- `GET /`: responsive home/upload page
- `POST /api/predict`: multipart form fields `image` and `serving_size`; classifies, estimates, and stores a prediction
- `GET /result/<id>`: prediction result page
- `POST /api/predictions/<id>/serving`: JSON `{"serving_size": 1.5}`; recalculates and persists calories
- `GET /api/predictions`: prediction history as JSON
- `GET /history`: prediction history page
- `GET /api/nutrition/<food_name>`: reference nutrition entry
- `GET /about`: project, dataset, model, and limitation information

The SQLite `predictions` table stores `id`, `food_name`, `confidence`, `serving_size`, `estimated_calories`, and UTC `prediction_time`. Uploaded images and the local database are ignored by Git.

## Example calorie calculation

For the reference entry **Dosa** (1 piece, approximately 168 kcal) and **2 servings**:

```text
Estimated Calories = 168 kcal/serving × 2 servings = 336 kcal (approximate)
```

Reference nutrition values are illustrative averages; recipes, brands, portion sizes, and cooking methods vary substantially.

## Evaluation metrics and limitations

Evaluation files and metric values are **pending until the model has been trained** on a real dataset with a held-out test split. Do not report an accuracy until `train.py` has completed successfully and the resulting test metrics have been reviewed.

Image classification cannot reliably determine serving weight, hidden ingredients, cooking oil, toppings, or recipe variations. The serving amount is user-provided. Confidence can be miscalibrated, the supported food classes constrain predictions, and a visually similar category may be selected. Calorie and macronutrient figures are approximate reference values and must not be used as a medical or dietary diagnosis.

## Future enhancements

- Build a better-curated multi-source dataset and document class balance and licensing.
- Add calibration, cross-validation, and per-class error analysis.
- Support multiple foods in one image and portion-size estimation from additional user input.
- Let users manage or export their local prediction history.
- Add optional authentication and deployment hardening before hosting.
