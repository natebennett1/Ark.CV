"""Dataset preparation and training utilities for the fish cascade workflow."""
from .data_prep import (
    CropRecord,
    FishCascadePrepConfig,
    PreparedArtifacts,
    SplitSpec,
    prepare_fish_only_detection_and_crops,
)
from .multitask_classifier import (
    ClassMapping,
    TrainingConfig,
    load_crop_metadata,
    train_multitask_classifier,
)

__all__ = [
    "CropRecord",
    "FishCascadePrepConfig",
    "PreparedArtifacts",
    "SplitSpec",
    "prepare_fish_only_detection_and_crops",
    "ClassMapping",
    "TrainingConfig",
    "load_crop_metadata",
    "train_multitask_classifier",
]
