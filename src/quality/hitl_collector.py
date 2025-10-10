"""
Human-in-the-Loop (HITL) data collector for quality assurance.

This module handles collection of low-confidence detections and count events
for manual review by analysts.
"""

import os
import csv
import cv2
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, Set
import numpy as np

from ..config.settings import ManualReviewConfig


class HITLCollector:
    """
    Collects low-confidence tracks and count events for human review.
    
    This class manages the export of detection crops, frame captures, and metadata
    for tracks that need manual review due to low confidence or ambiguous classification.
    """
    
    def __init__(self, config: ManualReviewConfig, location: str, date_str: str):
        self.config = config
        self.location = location
        self.date_str = date_str
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata CSV
        self.meta_csv_path = os.path.join(self.config.output_dir, "metadata.csv")
        self._initialize_metadata_csv()
        
        # Track states for HITL collection
        self.tracks: Dict[int, Dict[str, Any]] = {}
        self._seen_hashes: Set[str] = set() if config.enable_deduplication else set()
    
    def _initialize_metadata_csv(self):
        """Initialize the metadata CSV file with headers."""
        if not os.path.exists(self.meta_csv_path):
            with open(self.meta_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "saved_at_utc", "location", "date_str", "video_name",
                    "frame_idx", "timestamp_sec", "track_id",
                    "pred_class", "pred_conf",
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "crop_path", "frame_path", "direction", "notes"
                ])
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        if image is None or image.size == 0:
            return 0.0
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    def _compute_image_hash(self, image: np.ndarray) -> str:
        """Compute SHA1 hash of resized image for deduplication."""
        small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
        return hashlib.sha1(small.tobytes()).hexdigest()
    
    def _candidate_score(self, crop: np.ndarray, area: int, confidence: float) -> tuple:
        """Calculate a score for ranking candidate crops."""
        sharpness = self._calculate_sharpness(crop)
        proximity_to_threshold = -(self.config.lowconf_threshold - confidence)
        return (sharpness, area, proximity_to_threshold)
    
    def _get_output_directory(self, pred_class_name: str, category: Optional[str] = None) -> str:
        """Generate output directory path for organizing crops."""
        date_dir = self.date_str.replace("-", "")
        species_dir = (pred_class_name or "unknown").replace(" ", "_").lower()
        location_dir = self.location.replace(" ", "_")
        
        if category:
            return os.path.join(self.config.output_dir, category, date_dir, location_dir, species_dir)
        return os.path.join(self.config.output_dir, date_dir, location_dir, species_dir)
    
    def _expand_bounding_box(self, x1: int, y1: int, x2: int, y2: int, 
                           frame_width: int, frame_height: int) -> tuple:
        """Expand bounding box by the configured ratio."""
        box_width, box_height = (x2 - x1), (y2 - y1)
        dx = int(box_width * self.config.expand_ratio)
        dy = int(box_height * self.config.expand_ratio)
        
        expanded_x1 = max(0, x1 - dx)
        expanded_y1 = max(0, y1 - dy)
        expanded_x2 = min(frame_width - 1, x2 + dx)
        expanded_y2 = min(frame_height - 1, y2 + dy)
        
        return expanded_x1, expanded_y1, expanded_x2, expanded_y2
    
    def _write_metadata_row(self, row: list):
        """Write a row to the metadata CSV."""
        with open(self.meta_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
    
    def observe_detection(self, 
                         frame: np.ndarray,
                         frame_idx: int,
                         timestamp_sec: float,
                         video_name: str,
                         bbox: tuple,
                         pred_class_name: str,
                         pred_confidence: float,
                         track_id: int,
                         direction: Optional[str] = None):
        """
        Observe a detection for potential HITL collection.
        
        Args:
            frame: Current video frame
            frame_idx: Frame index
            timestamp_sec: Timestamp in seconds
            video_name: Name of the video file
            bbox: Bounding box as (x1, y1, x2, y2)
            pred_class_name: Predicted class name
            pred_confidence: Prediction confidence
            track_id: Track ID
            direction: Movement direction if available
        """
        if track_id is None:
            return
        
        frame_height, frame_width = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Get or initialize track state
        track_state = self.tracks.get(track_id, {
            "last_seen": frame_idx,
            "high_conf_seen": False,
            "best_candidate": None
        })
        
        track_state["last_seen"] = frame_idx
        confidence = float(pred_confidence)
        
        # If we've seen high confidence for this track, skip HITL collection
        if confidence >= self.config.lowconf_threshold:
            track_state["high_conf_seen"] = True
            track_state["best_candidate"] = None
            self.tracks[track_id] = track_state
            return
        
        # If we've already seen high confidence, don't collect low-conf samples
        if track_state.get("high_conf_seen"):
            self.tracks[track_id] = track_state
            return
        
        # Expand bounding box and extract crop
        ex1, ey1, ex2, ey2 = self._expand_bounding_box(x1, y1, x2, y2, frame_width, frame_height)
        crop = frame[ey1:ey2, ex1:ex2]
        
        if crop is None or crop.size == 0:
            self.tracks[track_id] = track_state
            return
        
        # Check for duplicates if deduplication is enabled
        if self.config.enable_deduplication:
            image_hash = self._compute_image_hash(crop)
            if image_hash in self._seen_hashes:
                self.tracks[track_id] = track_state
                return
        
        # Calculate candidate score
        area = max(1, (ex2 - ex1) * (ey2 - ey1))
        score = self._candidate_score(crop, area, confidence)
        
        # Create candidate record
        candidate = {
            "crop": crop,
            "frame": frame.copy(),
            "bbox": (x1, y1, x2, y2),
            "frame_idx": int(frame_idx),
            "timestamp_sec": float(timestamp_sec),
            "video_name": video_name,
            "pred_class_name": pred_class_name,
            "pred_conf": confidence,
            "direction": direction,
            "score": score
        }
        
        # Keep the best candidate for this track
        best = track_state.get("best_candidate")
        if best is None or candidate["score"] > best["score"]:
            track_state["best_candidate"] = candidate
            if self.config.enable_deduplication:
                self._seen_hashes.add(image_hash)
        
        self.tracks[track_id] = track_state
    
    def _flush_track(self, track_id: int):
        """Flush the best candidate for a track to disk."""
        track_state = self.tracks.get(track_id)
        if not track_state or track_state.get("high_conf_seen"):
            return
        
        candidate = track_state.get("best_candidate")
        if not candidate:
            return
        
        # Create output directory
        output_dir = self._get_output_directory(candidate["pred_class_name"])
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate unique filenames
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base_name = (f"{timestamp}_f{candidate['frame_idx']:06d}_t{track_id}_"
                    f"{candidate['pred_class_name']}_{candidate['pred_conf']:.2f}")
        
        crop_path = os.path.join(output_dir, base_name + "_crop.jpg")
        frame_path = os.path.join(output_dir, base_name + "_frame.jpg")
        
        # Save images
        cv2.imwrite(crop_path, candidate["crop"], [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(frame_path, candidate["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        # Write metadata
        x1, y1, x2, y2 = candidate["bbox"]
        self._write_metadata_row([
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            self.location,
            self.date_str,
            candidate["video_name"],
            candidate["frame_idx"],
            f"{candidate['timestamp_sec']:.3f}",
            track_id,
            candidate["pred_class_name"],
            f"{candidate['pred_conf']:.4f}",
            int(x1), int(y1), int(x2), int(y2),
            crop_path,
            frame_path,
            candidate["direction"] if candidate["direction"] is not None else "",
            "low_conf_track_best"
        ])
    
    def flag_count_event(self,
                        frame: np.ndarray,
                        frame_idx: int,
                        timestamp_sec: float,
                        video_name: str,
                        bbox: tuple,
                        pred_class_name: str,
                        pred_confidence: float,
                        track_id: int,
                        direction: Optional[str] = None):
        """
        Flag a count event for review due to low confidence.
        
        This method immediately saves the detection without batching since
        it represents an actual count that happened.
        """
        if frame is None:
            return
        
        x1, y1, x2, y2 = bbox
        frame_height, frame_width = frame.shape[:2]
        
        # Ensure valid bounding box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2 = min(frame_width, max(x1 + 1, int(x2)))
        y2 = min(frame_height, max(y1 + 1, int(y2)))
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return
        
        # Annotate frame with detection
        annotated_frame = frame.copy()
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Create output directory for count reviews
        output_dir = self._get_output_directory(pred_class_name, category="count_review")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate filenames
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        base_name = (f"{timestamp}_f{frame_idx:06d}_t{track_id}_"
                    f"{pred_class_name}_{pred_confidence:.2f}")
        
        crop_path = os.path.join(output_dir, base_name + "_crop.jpg")
        frame_path = os.path.join(output_dir, base_name + "_frame.jpg")
        
        # Save images
        cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(frame_path, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        
        # Write metadata
        self._write_metadata_row([
            datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            self.location,
            self.date_str,
            video_name,
            frame_idx,
            f"{timestamp_sec:.3f}",
            track_id,
            pred_class_name,
            f"{pred_confidence:.4f}",
            int(x1), int(y1), int(x2), int(y2),
            crop_path,
            frame_path,
            direction if direction is not None else "",
            "count_below_threshold"
        ])
    
    def garbage_collect_inactive(self, current_frame_idx: int):
        """Remove inactive tracks that haven't been seen recently."""
        tracks_to_remove = []
        
        for track_id, track_state in self.tracks.items():
            if current_frame_idx - track_state["last_seen"] > self.config.track_gap_frames:
                self._flush_track(track_id)
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            self.tracks.pop(track_id, None)
    
    def flush_all_tracks(self):
        """Flush all remaining tracks at the end of processing."""
        for track_id in list(self.tracks.keys()):
            self._flush_track(track_id)
            self.tracks.pop(track_id, None)
    
    def get_stats(self) -> dict:
        """Get statistics about HITL collection."""
        return {
            "active_tracks": len(self.tracks),
            "seen_hashes": len(self._seen_hashes),
            "output_directory": self.config.output_dir,
            "metadata_file": self.meta_csv_path
        }