"""Multi-task species/adipose classification utilities."""
from __future__ import annotations

import json
import logging
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from .data_prep import CropRecord

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainingConfig:
    train_records: Sequence[CropRecord]
    val_records: Sequence[CropRecord]
    root_dir: Path
    output_dir: Path
    image_size: int = 256
    batch_size: int = 32
    epochs: int = 40
    model_name: str = "efficientnet_b2"
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    adipose_loss_weight: float = 0.6
    label_smoothing: float = 0.05
    num_workers: int = 4
    amp: bool = True
    freeze_backbone_epochs: int = 0
    dropout: float = 0.1
    patience: int = 8


class RandomGamma:
    """Random gamma augmentation for PIL images."""

    def __init__(self, min_gamma: float = 0.9, max_gamma: float = 1.1, p: float = 0.5) -> None:
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.p = p

    def __call__(self, img):  # pragma: no cover - stochastic transform
        if random.random() > self.p:
            return img
        gamma = random.uniform(self.min_gamma, self.max_gamma)
        return TF.adjust_gamma(img, gamma)


class MultiTaskClassifier(nn.Module):
    """Shared-backbone classifier with species and adipose heads."""

    def __init__(self, model_name: str, num_species: int, num_adipose: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        feat_dim = getattr(self.backbone, "num_features", None) or getattr(self.backbone, "num_chs", None)
        if feat_dim is None:
            raise AttributeError(f"Unable to determine feature dimension for model {model_name}")
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.species_head = nn.Linear(feat_dim, num_species)
        self.adipose_head = nn.Linear(feat_dim, num_adipose)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(x)
        feats = self.dropout(feats)
        return self.species_head(feats), self.adipose_head(feats)


class CropDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """PyTorch dataset for crop metadata records."""

    def __init__(
        self,
        records: Sequence[CropRecord],
        root_dir: Path,
        species_to_idx: Dict[str, int],
        adipose_to_idx: Dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.records = list(records)
        self.root_dir = root_dir
        self.species_to_idx = species_to_idx
        self.adipose_to_idx = adipose_to_idx
        self.transform = transform

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        path = Path(record.crop_path)
        if not path.is_absolute():
            path = self.root_dir / path
        if not path.exists():
            raise FileNotFoundError(f"Crop image not found: {path}")
        with Image.open(path) as img:
            image = img.convert("RGB")
            tensor = self.transform(image)
        species_target = torch.tensor(self.species_to_idx[record.species], dtype=torch.long)
        adipose_target = torch.tensor(self.adipose_to_idx[record.adipose], dtype=torch.long)
        return tensor, species_target, adipose_target


def train_multitask_classifier(config: TrainingConfig) -> Dict[str, object]:
    """Train the multi-task crop classifier and return training artefacts."""

    if not config.train_records:
        raise ValueError("No training records provided")

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    root_dir = config.root_dir.expanduser().resolve()
    all_records = list(config.train_records) + list(config.val_records)

    species_names = sorted({rec.species for rec in all_records})
    adipose_names = sorted({rec.adipose for rec in all_records})

    species_to_idx = {name: idx for idx, name in enumerate(species_names)}
    adipose_to_idx = {name: idx for idx, name in enumerate(adipose_names)}

    LOGGER.info("Training multi-task classifier with %s species and %s adipose labels", len(species_names), len(adipose_names))

    train_transform, val_transform = _build_transforms(config.image_size)

    train_dataset = CropDataset(config.train_records, root_dir, species_to_idx, adipose_to_idx, train_transform)
    val_dataset = CropDataset(config.val_records, root_dir, species_to_idx, adipose_to_idx, val_transform) if config.val_records else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )
        if val_dataset is not None
        else None
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskClassifier(config.model_name, len(species_names), len(adipose_names), config.dropout).to(device)

    species_weights = _compute_class_weights(config.train_records, species_to_idx, attr="species")
    adipose_weights = _compute_class_weights(config.train_records, adipose_to_idx, attr="adipose")

    species_loss_fn = nn.CrossEntropyLoss(weight=species_weights.to(device), label_smoothing=config.label_smoothing)
    adipose_loss_fn = nn.CrossEntropyLoss(weight=adipose_weights.to(device), label_smoothing=config.label_smoothing)

    optimizer: Optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=config.amp and device.type == "cuda")

    if config.freeze_backbone_epochs > 0:
        _set_backbone_trainable(model, False)
        LOGGER.info("Freezing backbone for %s epochs", config.freeze_backbone_epochs)

    best_val_loss = float("inf")
    best_metrics: Dict[str, float] = {}
    epochs_since_improvement = 0

    metrics_path = output_dir / "training_log.jsonl"
    metrics_path.write_text("")

    for epoch in range(1, config.epochs + 1):
        if config.freeze_backbone_epochs > 0 and epoch == config.freeze_backbone_epochs + 1:
            _set_backbone_trainable(model, True)
            LOGGER.info("Unfroze backbone")

        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            species_loss_fn,
            adipose_loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            adipose_weight=config.adipose_loss_weight,
            amp=config.amp,
        )

        val_metrics = (
            _run_epoch(
                model,
                val_loader,
                device,
                species_loss_fn,
                adipose_loss_fn,
                optimizer=None,
                scaler=None,
                adipose_weight=config.adipose_loss_weight,
                amp=False,
            )
            if val_loader is not None
            else None
        )

        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_species_acc": train_metrics["species_acc"],
            "train_adipose_acc": train_metrics["adipose_acc"],
        }
        if val_metrics is not None:
            record.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_species_acc": val_metrics["species_acc"],
                    "val_adipose_acc": val_metrics["adipose_acc"],
                }
            )

        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")

        current_val = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        if current_val < best_val_loss:
            best_val_loss = current_val
            epochs_since_improvement = 0
            best_metrics = record.copy()
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "species_to_idx": species_to_idx,
                    "adipose_to_idx": adipose_to_idx,
                    "config": vars(config),
                },
                output_dir / "best_multitask.pt",
            )
        else:
            epochs_since_improvement += 1
            if val_loader is not None and config.patience and epochs_since_improvement >= config.patience:
                LOGGER.info("Early stopping at epoch %s", epoch)
                break

    mapping_path = output_dir / "class_mappings.json"
    mapping_path.write_text(json.dumps({"species": species_to_idx, "adipose": adipose_to_idx}, indent=2))

    return {
        "best_metrics": best_metrics,
        "metrics_file": metrics_path,
        "model_path": output_dir / "best_multitask.pt",
        "class_mapping_path": mapping_path,
    }


def _build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)
    train_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.RandomRotation(degrees=5),
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            jitter,
            RandomGamma(0.85, 1.15, p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, val_transform


def _run_epoch(
    model: MultiTaskClassifier,
    loader: Optional[DataLoader],
    device: torch.device,
    species_loss_fn: nn.Module,
    adipose_loss_fn: nn.Module,
    *,
    optimizer: Optional[Optimizer],
    scaler: Optional[torch.cuda.amp.GradScaler],
    adipose_weight: float,
    amp: bool,
) -> Dict[str, float]:
    if loader is None or len(loader) == 0:
        return {"loss": 0.0, "species_acc": 0.0, "adipose_acc": 0.0}

    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_species_correct = 0
    total_adipose_correct = 0
    total_samples = 0

    for images, species_targets, adipose_targets in loader:
        images = images.to(device, non_blocking=True)
        species_targets = species_targets.to(device, non_blocking=True)
        adipose_targets = adipose_targets.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                species_logits, adipose_logits = model(images)
                loss_species = species_loss_fn(species_logits, species_targets)
                loss_adipose = adipose_loss_fn(adipose_logits, adipose_targets)
                loss = loss_species + adipose_weight * loss_adipose
            if scaler is not None and amp and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                species_logits, adipose_logits = model(images)
                loss_species = species_loss_fn(species_logits, species_targets)
                loss_adipose = adipose_loss_fn(adipose_logits, adipose_targets)
                loss = loss_species + adipose_weight * loss_adipose

        batch_size = images.size(0)
        total_loss += loss.detach().item() * batch_size
        total_samples += batch_size
        total_species_correct += (species_logits.argmax(dim=1) == species_targets).sum().item()
        total_adipose_correct += (adipose_logits.argmax(dim=1) == adipose_targets).sum().item()

    return {
        "loss": total_loss / max(total_samples, 1),
        "species_acc": total_species_correct / max(total_samples, 1),
        "adipose_acc": total_adipose_correct / max(total_samples, 1),
    }


def _set_backbone_trainable(model: MultiTaskClassifier, requires_grad: bool) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = requires_grad


def _compute_class_weights(
    records: Sequence[CropRecord],
    label_to_idx: Dict[str, int],
    *,
    attr: str,
) -> torch.Tensor:
    counts = Counter(getattr(record, attr) for record in records)
    weights = [1.0] * len(label_to_idx)
    for label, idx in label_to_idx.items():
        count = counts.get(label, 0)
        if count == 0:
            LOGGER.warning("Label '%s' missing from training data; using unit weight", label)
            weight = 1.0
        else:
            weight = 1.0 / float(count)
        weights[idx] = weight
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    if weight_tensor.sum() > 0:
        weight_tensor = weight_tensor / weight_tensor.mean()
    return weight_tensor
