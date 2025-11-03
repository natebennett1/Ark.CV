"""Command-line interface for multi-task species/adipose crop training."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.training import (
    TrainingConfig,
    load_crop_metadata,
    train_multitask_classifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-task species/adipose classifier")
    parser.add_argument("--train-csv", nargs="+", required=True, help="Training crop metadata CSV(s)")
    parser.add_argument("--val-csv", nargs="*", default=[], help="Validation crop metadata CSV(s)")
    parser.add_argument("--root-dir", required=True, help="Root directory used to resolve crop paths")
    parser.add_argument("--output-dir", required=True, help="Directory for checkpoints and logs")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--model", default="efficientnet_b2", help="Backbone from timm.create_model")
    parser.add_argument("--lr", type=float, default=2e-4, help="Base learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--adipose-weight", type=float, default=0.6, help="Loss weight for the adipose head")
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--freeze-backbone", type=int, default=0, help="Freeze backbone for N warmup epochs")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    train_records = load_crop_metadata([Path(p) for p in args.train_csv])
    val_records = load_crop_metadata([Path(p) for p in args.val_csv]) if args.val_csv else []

    config = TrainingConfig(
        train_records=train_records,
        val_records=val_records,
        root_dir=Path(args.root_dir),
        output_dir=Path(args.output_dir),
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_name=args.model,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adipose_loss_weight=args.adipose_weight,
        label_smoothing=args.label_smoothing,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        freeze_backbone_epochs=args.freeze_backbone,
        dropout=args.dropout,
        patience=args.patience,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Starting training on %s", device)

    results = train_multitask_classifier(config)
    logging.info("Best metrics: %s", results["best_metrics"])
    logging.info("Model saved to: %s", results["model_path"])
    logging.info("Log file: %s", results["metrics_file"])
    logging.info("Class mapping: %s", results["class_mapping_path"])


if __name__ == "__main__":
    main()
