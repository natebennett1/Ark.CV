"""Utilities for preparing fish-only detection datasets and multi-task crop metadata."""
from __future__ import annotations

import csv
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import cv2

LOGGER = logging.getLogger(__name__)


@dataclass
class CropRecord:
    """Metadata describing a cropped fish image."""

    crop_path: Path
    species: str
    adipose: str
    split: str
    source_image: str
    bbox: Tuple[float, float, float, float]
    image_id: int
    annotation_id: int


@dataclass
class SplitSpec:
    """Description of a dataset split sourced from a Roboflow export."""

    name: str
    coco_annotations: Path
    images_dir: Path


@dataclass
class FishCascadePrepConfig:
    """Configuration for preparing fish-only detection and crop datasets."""

    output_dir: Path
    coco_annotations: Optional[Path] = None
    images_dir: Optional[Path] = None
    split_sources: Optional[Sequence[SplitSpec]] = None
    detection_class_name: str = "Fish"
    pad_ratio: float = 0.15
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    include_adipose_unknown: bool = True
    min_crop_pixels: int = 4
    split_seed: int = 17
    ensure_even_split: bool = True


@dataclass
class PreparedArtifacts:
    """Locations of the prepared detection annotations and crop metadata."""

    output_dir: Path
    detection_annotations: Mapping[str, Path]
    crop_metadata: Mapping[str, Path]
    combined_crop_csv: Path
    species_classes: List[str]
    adipose_classes: List[str]
    stats: Mapping[str, int]


def prepare_fish_only_detection_and_crops(config: FishCascadePrepConfig) -> PreparedArtifacts:
    """Create fish-only detection annotations and crop metadata from a COCO export."""

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "crops").mkdir(exist_ok=True)

    detection_paths: Dict[str, Path] = {}
    crop_csv_paths: Dict[str, Path] = {}
    combined_crop_records: List[CropRecord] = []
    species_names: set[str] = set()
    adipose_labels: set[str] = set()

    total_images = 0
    total_annotations = 0
    total_skipped = 0
    per_split_crop_counts: Dict[str, int] = {}

    if config.split_sources:
        for split_spec in config.split_sources:
            det_path, csv_path, split_records, split_stats, split_species, split_adipose = _process_split_dataset(
                split_spec,
                config,
                output_dir,
            )
            split_name = split_spec.name
            detection_paths[split_name] = det_path
            crop_csv_paths[split_name] = csv_path
            combined_crop_records.extend(split_records)
            species_names.update(split_species)
            adipose_labels.update(split_adipose)

            total_images += split_stats["images"]
            total_annotations += split_stats["annotations"]
            total_skipped += split_stats["skipped_crops"]
            per_split_crop_counts[split_name] = len(split_records)
    else:
        if config.coco_annotations is None or config.images_dir is None:
            raise ValueError("coco_annotations and images_dir must be provided when split_sources is not set")

        coco_ann_path = config.coco_annotations.expanduser().resolve()
        images_dir = config.images_dir.expanduser().resolve()

        if not coco_ann_path.exists():
            raise FileNotFoundError(f"COCO annotations not found: {coco_ann_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {images_dir}")

        LOGGER.info("Loading COCO annotations from %s", coco_ann_path)
        coco = json.loads(coco_ann_path.read_text())
        categories = coco.get("categories", [])
        cat_lookup = {c["id"]: c.get("name", str(c["id"])) for c in categories}

        images = coco.get("images", [])
        annotations = coco.get("annotations", [])

        if not images:
            raise ValueError("No images found in the COCO annotation file")
        if not annotations:
            LOGGER.warning("No annotations found; detection dataset will be empty")

        split_map = _assign_splits(images, config)
        detection_records: MutableMapping[str, List[dict]] = {"train": [], "val": [], "test": []}
        crop_records: MutableMapping[str, List[CropRecord]] = {"train": [], "val": [], "test": []}
        next_det_ann_id: MutableMapping[str, int] = {split: 1 for split in detection_records}

        skipped_crops = 0
        processed_annotations = 0

        img_by_id = {img["id"]: img for img in images}

        for ann in annotations:
            img_meta = img_by_id.get(ann.get("image_id"))
            if not img_meta:
                LOGGER.debug("Skipping annotation %s because the image is missing", ann.get("id"))
                continue

            split = split_map[img_meta["id"]]
            image_path = images_dir / img_meta["file_name"]

            if not image_path.exists():
                LOGGER.warning("Image %s referenced by annotation %s is missing", image_path, ann.get("id"))
                continue

            x, y, w, h = ann.get("bbox", [None, None, None, None])
            if None in (x, y, w, h) or w <= 0 or h <= 0:
                LOGGER.debug("Skipping annotation %s with invalid bbox %s", ann.get("id"), ann.get("bbox"))
                continue

            crop = _extract_crop(image_path, x, y, w, h, config.pad_ratio)
            if crop is None:
                skipped_crops += 1
                continue

            crop_h, crop_w = crop.shape[:2]
            if crop_h < config.min_crop_pixels or crop_w < config.min_crop_pixels:
                skipped_crops += 1
                continue

            crop_filename = _build_crop_filename(img_meta["file_name"], ann.get("id"))
            crop_rel_path = Path("crops") / split / crop_filename
            crop_abs_path = output_dir / crop_rel_path
            crop_abs_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(crop_abs_path), crop)

            species, adipose = _split_species_and_adipose(ann, cat_lookup, config.include_adipose_unknown)
            species_names.add(species)
            adipose_labels.add(adipose)

            annotation_identifier = ann.get("id", 0)
            try:
                annotation_id_int = int(annotation_identifier) if annotation_identifier is not None else 0
            except (TypeError, ValueError):
                LOGGER.debug("Coercing non-numeric annotation id %s to 0", annotation_identifier)
                annotation_id_int = 0

            crop_records[split].append(
                CropRecord(
                    crop_path=crop_rel_path,
                    species=species,
                    adipose=adipose,
                    split=split,
                    source_image=img_meta["file_name"],
                    bbox=(float(x), float(y), float(w), float(h)),
                    image_id=img_meta["id"],
                    annotation_id=annotation_id_int,
                )
            )

            iscrowd_value = ann.get("iscrowd")
            try:
                iscrowd_int = int(iscrowd_value) if iscrowd_value is not None else 0
            except (TypeError, ValueError):
                LOGGER.debug("Coercing non-numeric iscrowd value %s to 0", iscrowd_value)
                iscrowd_int = 0

            detection_records[split].append(
                {
                    "id": next_det_ann_id[split],
                    "image_id": img_meta["id"],
                    "category_id": 1,
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": iscrowd_int,
                }
            )
            next_det_ann_id[split] += 1
            processed_annotations += 1

        for split in ("train", "val", "test"):
            split_images = [img for img in images if split_map[img["id"]] == split]
            det_coco = {
                "info": coco.get("info", {}),
                "licenses": coco.get("licenses", []),
                "images": split_images,
                "annotations": detection_records[split],
                "categories": [{"id": 1, "name": config.detection_class_name}],
            }
            det_path = output_dir / f"anns_det_fish_only_{split}.json"
            det_path.write_text(json.dumps(det_coco))
            detection_paths[split] = det_path

            csv_path = output_dir / f"crops_meta_{split}.csv"
            _write_crop_csv(csv_path, crop_records[split])
            crop_csv_paths[split] = csv_path
            combined_crop_records.extend(crop_records[split])
            per_split_crop_counts[split] = len(crop_records[split])

        combined_csv = output_dir / "crops_meta.csv"
        _write_crop_csv(combined_csv, combined_crop_records)

        stats = {
            "images": len(images),
            "annotations": processed_annotations,
            "skipped_crops": skipped_crops,
            "train_crops": len(crop_records["train"]),
            "val_crops": len(crop_records["val"]),
            "test_crops": len(crop_records["test"]),
        }

        LOGGER.info(
            "Prepared detection annotations and %s crops (skipped %s)",
            processed_annotations - skipped_crops,
            skipped_crops,
        )

        return PreparedArtifacts(
            output_dir=output_dir,
            detection_annotations=detection_paths,
            crop_metadata=crop_csv_paths,
            combined_crop_csv=combined_csv,
            species_classes=sorted(species_names),
            adipose_classes=sorted(adipose_labels),
            stats=stats,
        )

    combined_csv = output_dir / "crops_meta.csv"
    _write_crop_csv(combined_csv, combined_crop_records)

    stats = {
        "images": total_images,
        "annotations": total_annotations,
        "skipped_crops": total_skipped,
    }
    for split, count in per_split_crop_counts.items():
        stats[f"{split}_crops"] = count

    LOGGER.info(
        "Prepared detection annotations for %s splits with %s crops (skipped %s)",
        len(detection_paths),
        sum(per_split_crop_counts.values()),
        total_skipped,
    )

    return PreparedArtifacts(
        output_dir=output_dir,
        detection_annotations=detection_paths,
        crop_metadata=crop_csv_paths,
        combined_crop_csv=combined_csv,
        species_classes=sorted(species_names),
        adipose_classes=sorted(adipose_labels),
        stats=stats,
    )


def _process_split_dataset(
    split_spec: SplitSpec,
    config: FishCascadePrepConfig,
    output_dir: Path,
) -> Tuple[Path, Path, List[CropRecord], Dict[str, int], set[str], set[str]]:
    split_name = split_spec.name
    safe_split = split_name.replace(" ", "_")

    coco_ann_path = split_spec.coco_annotations.expanduser().resolve()
    images_dir = split_spec.images_dir.expanduser().resolve()

    if not coco_ann_path.exists():
        raise FileNotFoundError(f"COCO annotations not found for split '{split_name}': {coco_ann_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found for split '{split_name}': {images_dir}")

    LOGGER.info("Loading %s split annotations from %s", split_name, coco_ann_path)
    coco = json.loads(coco_ann_path.read_text())

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    cat_lookup = {c["id"]: c.get("name", str(c["id"])) for c in categories}

    if not images:
        raise ValueError(f"No images found in the COCO annotation file for split '{split_name}'")
    if not annotations:
        LOGGER.warning("No annotations found for split '%s'; detection dataset will be empty", split_name)

    crop_records: List[CropRecord] = []
    detection_records: List[dict] = []
    species_names: set[str] = set()
    adipose_labels: set[str] = set()
    skipped_crops = 0
    processed_annotations = 0
    next_det_ann_id = 1

    img_by_id = {img["id"]: img for img in images}

    for ann in annotations:
        img_meta = img_by_id.get(ann.get("image_id"))
        if not img_meta:
            LOGGER.debug(
                "Skipping annotation %s in split '%s' because the image is missing",
                ann.get("id"),
                split_name,
            )
            continue

        image_path = images_dir / img_meta["file_name"]
        if not image_path.exists():
            LOGGER.warning(
                "Image %s referenced by annotation %s in split '%s' is missing",
                image_path,
                ann.get("id"),
                split_name,
            )
            continue

        x, y, w, h = ann.get("bbox", [None, None, None, None])
        if None in (x, y, w, h) or w <= 0 or h <= 0:
            LOGGER.debug(
                "Skipping annotation %s in split '%s' with invalid bbox %s",
                ann.get("id"),
                split_name,
                ann.get("bbox"),
            )
            continue

        crop = _extract_crop(image_path, x, y, w, h, config.pad_ratio)
        if crop is None:
            skipped_crops += 1
            continue

        crop_h, crop_w = crop.shape[:2]
        if crop_h < config.min_crop_pixels or crop_w < config.min_crop_pixels:
            skipped_crops += 1
            continue

        crop_filename = _build_crop_filename(img_meta["file_name"], ann.get("id"))
        crop_rel_path = Path("crops") / safe_split / crop_filename
        crop_abs_path = output_dir / crop_rel_path
        crop_abs_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_abs_path), crop)

        species, adipose = _split_species_and_adipose(ann, cat_lookup, config.include_adipose_unknown)
        species_names.add(species)
        adipose_labels.add(adipose)

        annotation_identifier = ann.get("id", 0)
        try:
            annotation_id_int = int(annotation_identifier) if annotation_identifier is not None else 0
        except (TypeError, ValueError):
            LOGGER.debug(
                "Coercing non-numeric annotation id %s to 0 in split '%s'",
                annotation_identifier,
                split_name,
            )
            annotation_id_int = 0

        crop_records.append(
            CropRecord(
                crop_path=crop_rel_path,
                species=species,
                adipose=adipose,
                split=split_name,
                source_image=img_meta["file_name"],
                bbox=(float(x), float(y), float(w), float(h)),
                image_id=img_meta["id"],
                annotation_id=annotation_id_int,
            )
        )

        iscrowd_value = ann.get("iscrowd")
        try:
            iscrowd_int = int(iscrowd_value) if iscrowd_value is not None else 0
        except (TypeError, ValueError):
            LOGGER.debug(
                "Coercing non-numeric iscrowd value %s to 0 in split '%s'",
                iscrowd_value,
                split_name,
            )
            iscrowd_int = 0

        detection_records.append(
            {
                "id": next_det_ann_id,
                "image_id": img_meta["id"],
                "category_id": 1,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": iscrowd_int,
            }
        )
        next_det_ann_id += 1
        processed_annotations += 1

    det_coco = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": detection_records,
        "categories": [{"id": 1, "name": config.detection_class_name}],
    }

    det_path = output_dir / f"anns_det_fish_only_{safe_split}.json"
    det_path.write_text(json.dumps(det_coco))

    csv_path = output_dir / f"crops_meta_{safe_split}.csv"
    _write_crop_csv(csv_path, crop_records)

    stats = {
        "images": len(images),
        "annotations": processed_annotations,
        "skipped_crops": skipped_crops,
    }

    LOGGER.info(
        "Split '%s': wrote %s detection boxes and %s crops (skipped %s)",
        split_name,
        processed_annotations,
        len(crop_records),
        skipped_crops,
    )

    return det_path, csv_path, crop_records, stats, species_names, adipose_labels
def load_crop_metadata(csv_paths: Sequence[Path]) -> List[CropRecord]:
    """Load crop metadata rows from one or more CSV files."""

    records: List[CropRecord] = []
    for csv_path in csv_paths:
        csv_path = csv_path.expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"Crop metadata CSV not found: {csv_path}")
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                records.append(
                    CropRecord(
                        crop_path=Path(row["crop_path"]),
                        species=row["species"],
                        adipose=row.get("adipose", "AU"),
                        split=row.get("split", ""),
                        source_image=row.get("source_image", ""),
                        bbox=(
                            float(row.get("bbox_x", 0.0)),
                            float(row.get("bbox_y", 0.0)),
                            float(row.get("bbox_w", 0.0)),
                            float(row.get("bbox_h", 0.0)),
                        ),
                        image_id=int(row.get("image_id", 0)),
                        annotation_id=int(row.get("annotation_id", 0)),
                    )
                )
    return records


def _assign_splits(images: Sequence[Mapping[str, object]], config: FishCascadePrepConfig) -> Dict[int, str]:
    image_ids = [int(img["id"]) for img in images]
    rng = random.Random(config.split_seed)
    rng.shuffle(image_ids)

    total = len(image_ids)
    val_count = max(1 if config.ensure_even_split and total > 1 else 0, int(round(total * config.val_ratio)))
    test_count = max(1 if config.ensure_even_split and total > 2 else 0, int(round(total * config.test_ratio)))
    if val_count + test_count >= total:
        val_count = max(0, min(val_count, total - 1))
        test_count = max(0, min(test_count, total - val_count - 1))

    split_map: Dict[int, str] = {}
    for idx, image_id in enumerate(image_ids):
        if idx < val_count:
            split_map[image_id] = "val"
        elif idx < val_count + test_count:
            split_map[image_id] = "test"
        else:
            split_map[image_id] = "train"

    return split_map


def _extract_crop(image_path: Path, x: float, y: float, w: float, h: float, pad_ratio: float):
    img = cv2.imread(str(image_path))
    if img is None:
        LOGGER.warning("Failed to read image %s", image_path)
        return None

    height, width = img.shape[:2]
    pad_w = w * pad_ratio
    pad_h = h * pad_ratio

    cx = x + w / 2.0
    cy = y + h / 2.0

    x1 = max(0.0, cx - (w / 2.0 + pad_w / 2.0))
    y1 = max(0.0, cy - (h / 2.0 + pad_h / 2.0))
    x2 = min(width, cx + (w / 2.0 + pad_w / 2.0))
    y2 = min(height, cy + (h / 2.0 + pad_h / 2.0))

    x1_i = int(math.floor(x1))
    y1_i = int(math.floor(y1))
    x2_i = int(math.ceil(x2))
    y2_i = int(math.ceil(y2))

    if x2_i <= x1_i or y2_i <= y1_i:
        return None

    return img[y1_i:y2_i, x1_i:x2_i]


def _build_crop_filename(image_name: str, annotation_id: Optional[int]) -> str:
    stem = Path(image_name).stem
    if annotation_id is None:
        return f"{stem}.jpg"
    return f"{stem}_{annotation_id}.jpg"


def _split_species_and_adipose(
    ann: Mapping[str, object],
    cat_lookup: Mapping[int, str],
    include_unknown: bool,
) -> Tuple[str, str]:
    cat_name = ann.get("category_name") or ann.get("cat_name") or ""
    if not cat_name:
        cat_name = cat_lookup.get(ann.get("category_id")) or "Unknown_AU"

    if isinstance(cat_name, str) and "_" in cat_name:
        species, adipose = cat_name.split("_", 1)
    else:
        species = str(cat_name)
        adipose = "AU"

    adipose = adipose or "AU"
    if not include_unknown and adipose not in {"AP", "AA"}:
        adipose = "AP"
    return species, adipose


def _write_crop_csv(path: Path, records: Iterable[CropRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = [
            "crop_path",
            "species",
            "adipose",
            "split",
            "source_image",
            "image_id",
            "annotation_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "crop_path": record.crop_path.as_posix(),
                    "species": record.species,
                    "adipose": record.adipose,
                    "split": record.split,
                    "source_image": record.source_image,
                    "image_id": record.image_id,
                    "annotation_id": record.annotation_id,
                    "bbox_x": record.bbox[0],
                    "bbox_y": record.bbox[1],
                    "bbox_w": record.bbox[2],
                    "bbox_h": record.bbox[3],
                }
            )
