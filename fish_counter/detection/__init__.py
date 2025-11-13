"""Detection module for YOLO-based fish detection."""

from .detector import FishDetector
from .adipose_detector import AdiposeDetector

__all__ = ['FishDetector', 'AdiposeDetector']