"""Prepare fish-only COCO annotations and crop metadata from Roboflow exports."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.training import FishCascadePrepConfig, SplitSpec, prepare_fish_only_detection_and_crops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Roboflow COCO labels into fish-only datasets")
    parser.add_argument(
        "--dataset-root",
        help="Roboflow export root containing train/valid/test folders with COCO annotations",
    )
    parser.add_argument("--images-dir", help="Directory containing source images (single-split mode)")
    parser.add_argument("--coco-ann", help="COCO annotation JSON with species+adipose labels (single-split mode)")
    parser.add_argument("--output-dir", required=True, help="Output directory for prepared datasets")
    parser.add_argument("--pad-ratio", type=float, default=0.15, help="Padding ratio for crop extraction")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (by image)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio (by image)")
    parser.add_argument("--no-unknown-adipose", action="store_true", help="Map unlabelled adipose states to Present")
    parser.add_argument("--min-crop-size", type=int, default=4, help="Minimum crop dimension in pixels")
    parser.add_argument("--seed", type=int, default=17, help="Random seed for dataset splits")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    output_dir = Path(args.output_dir)

    if args.dataset_root:
        split_specs = _discover_split_specs(Path(args.dataset_root))
        if not split_specs:
            raise SystemExit(
                "No valid splits were found under the dataset root. Expecting directories such as train/, valid/, or test/."
            )

        config = FishCascadePrepConfig(
            output_dir=output_dir,
            split_sources=split_specs,
            pad_ratio=args.pad_ratio,
            include_adipose_unknown=not args.no_unknown_adipose,
            min_crop_pixels=args.min_crop_size,
        )
    else:
        if not args.images_dir or not args.coco_ann:
            raise SystemExit("--images-dir and --coco-ann are required when --dataset-root is not provided")

        config = FishCascadePrepConfig(
            output_dir=output_dir,
            coco_annotations=Path(args.coco_ann),
            images_dir=Path(args.images_dir),
            pad_ratio=args.pad_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            include_adipose_unknown=not args.no_unknown_adipose,
            min_crop_pixels=args.min_crop_size,
            split_seed=args.seed,
        )

    artifacts = prepare_fish_only_detection_and_crops(config)

    logging.info("Detection annotations: %s", {k: str(v) for k, v in artifacts.detection_annotations.items()})
    logging.info("Crop metadata: %s", {k: str(v) for k, v in artifacts.crop_metadata.items()})
    logging.info("Species classes: %s", ", ".join(artifacts.species_classes))
    logging.info("Adipose classes: %s", ", ".join(artifacts.adipose_classes))
    logging.info("Stats: %s", artifacts.stats)


def _discover_split_specs(dataset_root: Path) -> list[SplitSpec]:
    dataset_root = dataset_root.expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    logging.info("Scanning dataset root %s for Roboflow splits", dataset_root)

    split_candidates = [
        ("train", "train"),
        ("valid", "val"),
        ("val", "val"),
        ("validation", "val"),
        ("test", "test"),
    ]

    seen: set[str] = set()
    split_specs: list[SplitSpec] = []

    for dirname, canonical in split_candidates:
        split_dir = dataset_root / dirname
        if not split_dir.exists() or canonical in seen:
            continue

        ann_path = _locate_coco_file(split_dir)
        images_dir = split_dir / "images"
        if not images_dir.exists():
            images_dir = split_dir

        logging.info(
            "Discovered split '%s' (alias '%s') with annotations at %s and images at %s",
            dirname,
            canonical,
            ann_path,
            images_dir,
        )

        split_specs.append(
            SplitSpec(
                name=canonical,
                coco_annotations=ann_path,
                images_dir=images_dir,
            )
        )
        seen.add(canonical)

    return split_specs


def _locate_coco_file(split_dir: Path) -> Path:
    candidates = [
        split_dir / "_annotations.coco.json",
        split_dir / "annotations.coco.json",
        split_dir / f"{split_dir.name}_annotations.coco.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    json_files = list(split_dir.glob("*.coco.json")) or list(split_dir.glob("*.json"))
    if len(json_files) == 1:
        return json_files[0]

    raise FileNotFoundError(
        f"Could not locate a COCO annotation file in {split_dir}. Expected one of {candidates} or a single .json file."
    )


if __name__ == "__main__":
    main()
