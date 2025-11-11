"""
YOLO-based fish detector with tracking integration.

This module wraps the Ultralytics YOLO model and provides a clean interface
for fish detection with multi-object tracking.
"""

import os
import tempfile
import yaml
import torch
import traceback
from typing import Optional, Dict, Any, Tuple
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ..config.settings import ModelConfig, BotsortConfig


class FishDetector:
    """
    YOLO-based fish detector with integrated tracking.
    
    This class handles model loading, inference, and tracking in a stateful manner.
    """
    
    def __init__(self, model_config: ModelConfig, botsort_config: BotsortConfig):
        self.model_config = model_config
        self.botsort_config = botsort_config
        self.model: Optional[YOLO] = None
        self.device = self._determine_device()
        self.tracker_config_file = self._create_tracker_config_file()

    def _determine_device(self) -> str:
        """Determine the best available device."""
        if self.model_config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.model_config.device
    
    def _create_tracker_config_file(self) -> str:
        """Create a YAML configuration file for the tracker."""
        # Create temporary file
        fd, temp_file = tempfile.mkstemp(suffix='.yaml', prefix='botsort_config_')
        
        # Build config dictionary
        config = {
            "tracker_type": self.botsort_config.tracker_type,
            "track_high_thresh": self.botsort_config.track_high_thresh,
            "track_low_thresh": self.botsort_config.track_low_thresh,
            "new_track_thresh": self.botsort_config.new_track_thresh,
            "track_buffer": self.botsort_config.track_buffer,
            "match_thresh": self.botsort_config.match_thresh,
            "fuse_score": self.botsort_config.fuse_score,
            "gmc_method": self.botsort_config.gmc_method,
            "proximity_thresh": self.botsort_config.proximity_thresh,
            "appearance_thresh": self.botsort_config.appearance_thresh,
            "with_reid": self.botsort_config.with_reid,
            "max_time_lost": self.botsort_config.max_time_lost
        }
        
        try:
            with os.fdopen(fd, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception:
            os.close(fd)
            raise
            
        return temp_file
    
    def load_model(self) -> None:
        """Load the YOLO model with error handling."""
        if not self.model_config.model_path:
            raise ValueError("Model path not configured")
        
        print(f"Attempting to load model from: {self.model_config.model_path}")
        
        try:
            self.model = YOLO(self.model_config.model_path).to(self.device)
            self.model.model.eval()
            print(f"✔ Model loaded successfully on device: {self.device}")
        except Exception as e:
            print(f"✖ Failed to load model: {e}")
            traceback.print_exc()
            raise
        
        # Configure inference settings
        self.model.overrides["conf"] = self.model_config.confidence_threshold
        self.model.overrides["iou"] = self.model_config.iou_threshold
        self.model.overrides["max_det"] = self.model_config.max_detections
    
    def detect_and_track(self, frame: np.ndarray) -> Results:
        """
        Run detection and tracking on a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Ultralytics Results object with detections and tracking IDs
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            results = self.model.track(
                source=frame,
                persist=True,
                device=self.device,
                verbose=False,
                tracker=self.tracker_config_file
            )[0]
        except Exception:
            results = self.model.track(
                source=frame,
                persist=True,
                device=self.device,
                verbose=False,
                tracker="botsort.yaml"
            )[0]
        
        return results
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        if self.model is None:
            return f"class_{class_id}"
        
        if class_id < len(self.model.names):
            return self.model.names[class_id]
        return f"class_{class_id}"
    
    def extract_detections(self, results: Results) -> Tuple[
        Optional[np.ndarray],  # boxes
        Optional[np.ndarray],  # track_ids
        Optional[np.ndarray],  # confidences
        Optional[np.ndarray]   # class_ids
    ]:
        """
        Extract detection data from results in a consistent format.
        
        Returns:
            Tuple of (boxes, track_ids, confidences, class_ids) or None values if no detections
        """
        if results.boxes is None or results.boxes.id is None:
            return None, None, None, None
        
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        
        return boxes, track_ids, confidences, class_ids
    
    def cleanup(self) -> None:
        """Clean up GPU memory and resources."""
        if self.model is not None:
            # Clear GPU cache if using CUDA
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        self.model = None
        print("✔ Detector resources cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None