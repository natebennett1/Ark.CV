"""
Adipose fin detector for secondary classification.

This module provides adipose fin detection capabilities as a second-pass
refinement for salmonid species classification.
"""

import numpy as np
import torch
import traceback
from typing import Optional, Tuple
from ultralytics import YOLO

from ..config.settings import ModelConfig


class AdiposeDetector:
    """
    Secondary YOLO model for adipose fin detection.
    
    This detector is used to refine salmonid classifications by determining
    whether the adipose fin is present or absent.
    """
    
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model: Optional[YOLO] = None
        self.device = self._determine_device()

    def _determine_device(self) -> str:
        """Determine the best available device."""
        if self.model_config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.model_config.device

    def _patch_torch_load(self):
        """Apply compatibility patch for torch.load."""
        original_load = torch.load
        
        def patched_load(*args, **kwargs):
            if 'weights_only' not in kwargs:
                kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        return original_load
        
    def load_model(self) -> bool:
        """
        Load the adipose detection model.
        
        Returns:
            True if model loaded successfully, False if no model path provided
        """
        if not self.model_config.adipose_model_path:
            print("ℹ No adipose model provided; skipping second-pass refinement.")
            return False

        print(f"Attempting to load model from: {self.model_config.adipose_model_path}")

        # Apply torch.load patch for compatibility
        original_load = self._patch_torch_load()

        try:
            original_load = self._patch_torch_load()
            self.model = YOLO(self.model_config.adipose_model_path).to(self.device)
            self.model.model.eval()
            print(f"✔ Adipose model loaded successfully on device: {self.device}")
            return True
        except Exception as e:
            print(f"✖ Failed to load model: {e}")
            traceback.print_exc()
            raise
        finally:
            # Restore original torch.load
            torch.load = original_load
    
    def _expand_box(self, x1: int, y1: int, x2: int, y2: int, 
                   frame_width: int, frame_height: int, ratio: float = 0.20) -> Tuple[int, int, int, int]:
        """Expand bounding box by a given ratio."""
        box_width, box_height = (x2 - x1), (y2 - y1)
        dx, dy = int(box_width * ratio), int(box_height * ratio)
        
        expanded_x1 = max(0, x1 - dx)
        expanded_y1 = max(0, y1 - dy)
        expanded_x2 = min(frame_width - 1, x2 + dx)
        expanded_y2 = min(frame_height - 1, y2 + dy)
        
        return expanded_x1, expanded_y1, expanded_x2, expanded_y2
    
    def infer_adipose_status(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """
        Infer adipose fin status for a detected fish.
        
        Args:
            frame: Input frame
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Tuple of (status, confidence) where status is "Present", "Absent", or "Unknown"
        """
        if self.model is None:
            return "Unknown", 0.0
        
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame.shape[:2]
        
        # Expand the bounding box
        ex1, ey1, ex2, ey2 = self._expand_box(
            x1, y1, x2, y2, frame_width, frame_height, 
            self.model_config.adipose_expand_ratio
        )
        
        # Crop the region
        crop = frame[ey1:ey2, ex1:ex2]
        if crop.size == 0:
            return "Unknown", 0.0
        
        # Run inference
        results = self.model.predict(source=crop, verbose=False)[0]
        
        # Handle classifier head output (preferred)
        if hasattr(results, "probs") and results.probs is not None:
            probs = results.probs.data.float().cpu().numpy()
            best_idx = int(np.argmax(probs))
            confidence = float(probs[best_idx])
            
            # Get class names (default mapping if not available)
            names = getattr(self.model, "names", {0: "Absent", 1: "Present"})
            status = names.get(best_idx, "Unknown")
            
            return (status if confidence >= self.model_config.adipose_min_confidence else "Unknown", confidence)
        
        # Handle detection head output (fallback)
        if results.boxes is None or len(results.boxes) == 0:
            return "Unknown", 0.0
        
        confidences = results.boxes.conf.float().cpu().numpy()
        class_ids = results.boxes.cls.int().cpu().numpy()
        
        best_detection = int(np.argmax(confidences))
        confidence = float(confidences[best_detection])
        class_id = int(class_ids[best_detection])
        
        # Get class names
        names = getattr(self.model, "names", {0: "Absent", 1: "Present"})
        status = names.get(class_id, "Unknown")
        
        return (status if confidence >= self.model_config.adipose_min_confidence else "Unknown", confidence)
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.model is not None:
            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.model = None
        print("✔ Adipose detector resources cleaned up")
    
    @property
    def is_loaded(self) -> bool:
        """Check if adipose model is loaded."""
        return self.model is not None
    
    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()