"""Split labeled local images into train/validation/test class folders.

CSV format: image,label
Image paths are relative to --image-root. Labels are the classifier class names.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SPLITS = ("train", "validation", "test")


def read_labeled_images(csv_path: Path, image_root: Path) -> dict[str, list[Path]]:
    samples: dict[str, list[Path]] = defaultdict(list)
    seen_paths: set[Path] = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or not {"image", "label"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain columns named 'image' and 'label'.")
        for row_number, row in enumerate(reader, start=2):
            image_value = (row.get("image") or "").strip()
            label = (row.get("label") or "").strip()
            if not image_value or not label:
                raise ValueError(f"CSV row {row_number} needs both image and label values.")
            if label in {".", ".."} or "/" in label or "\\" in label:
                raise ValueError(f"CSV row {row_number} has an invalid class label: {label!r}.")
            relative_path = Path(image_value)
            if relative_path.is_absolute():
                raise ValueError(f"CSV row {row_number} image path must be relative.")
            image_path = (image_root / relative_path).resolve()
            try:
                image_path.relative_to(image_root.resolve())
            except ValueError as exc:
                raise ValueError(f"CSV row {row_number} image path escapes --image-root.") from exc
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError(
                    f"CSV row {row_number} has an unsupported image extension: "
                    f"{image_path.suffix or '(none)'}."
                )
            if not image_path.is_file():
                raise FileNotFoundError(f"CSV row {row_number} image was not found: {image_path}")
            if image_path in seen_paths:
                raise ValueError(f"Image appears more than once in the CSV: {image_value}")
            seen_paths.add(image_path)
            samples[label].append(image_path)
    if len(samples) < 2:
        raise ValueError("At least two labeled food classes are required.")
    undersized = {label: len(paths) for label, paths in samples.items() if len(paths) < 3}
    if undersized:
        counts = ", ".join(f"{label}: {count}" for label, count in sorted(undersized.items()))
        raise ValueError(f"Each class needs at least 3 images for three splits; found {counts}.")
    return dict(samples)


def split_class(paths: list[Path], seed: int) -> dict[str, list[Path]]:
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    count = len(shuffled)
    train_end = max(1, int(count * 0.7))
    validation_count = max(1, int(count * 0.15))
    test_count = count - train_end - validation_count
    if test_count < 1:
        test_count = 1
        train_end -= 1
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end : train_end + validation_count],
        "test": shuffled[train_end + validation_count :],
    }


def prepare_dataset(
    csv_path: Path,
    image_root: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    samples = read_labeled_images(csv_path, image_root)
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory "
            "so existing dataset files are not overwritten."
        )
    split_samples = {
        label: split_class(paths, seed)
        for label, paths in sorted(samples.items())
    }
    summary: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        for label, paths_by_split in split_samples.items():
            source_paths = paths_by_split[split]
            if not source_paths:
                raise ValueError(f"Class {label!r} has no images in the {split} split.")
            destination = output_dir / split / label
            destination.mkdir(parents=True, exist_ok=True)
            for index, source in enumerate(source_paths):
                shutil.copy2(source, destination / f"{index:05d}_{source.name}")
            summary[split][label] = len(source_paths)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="CSV with image,label columns")
    parser.add_argument("--image-root", type=Path, required=True, help="Root directory containing images")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = prepare_dataset(args.annotations, args.image_root, args.output_dir, args.seed)
    print(f"Dataset prepared at {args.output_dir.resolve()}")
    for split, classes in summary.items():
        print(f"{split}: {sum(classes.values())} images across {len(classes)} classes")


if __name__ == "__main__":
    main()
