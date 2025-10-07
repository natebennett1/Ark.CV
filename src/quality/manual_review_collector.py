"""
Manual Review Data Collector for Quality Assurance.

This module collects potential issues for manual review, specifically:
1. Low confidence crossings - fish that crossed the center line but were never confident
2. Potential occlusions - situations where multiple tracks are close together
"""

import os
import csv
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, deque

from ..config.settings import ManualReviewConfig


class ManualReviewCollector:
    """
    Collects potential issues for manual review.
    
    This class focuses on two main types of issues:
    1. Low confidence crossings - fish that crossed the center line but were never confident
    2. Potential occlusions - situations where multiple tracks are close together
    """
    
    def __init__(self, config: ManualReviewConfig, location: str, date_str: str):
        self.config = config
        self.location = location
        self.date_str = date_str
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.low_conf_dir = os.path.join(self.config.output_dir, "low_confidence_crossings")
        self.occlusion_dir = os.path.join(self.config.output_dir, "potential_occlusions")
        Path(self.low_conf_dir).mkdir(parents=True, exist_ok=True)
        Path(self.occlusion_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata CSVs
        self.low_conf_csv = os.path.join(self.low_conf_dir, "low_confidence_crossings.csv")
        self.occlusion_csv = os.path.join(self.occlusion_dir, "potential_occlusions.csv")
        self._initialize_csv_files()
        
        # Track states for collection
        self.track_states: Dict[int, Dict[str, Any]] = {}
        self.proximity_tracker = ProximityTracker()
        
        # Frame state for occlusion detection
        self.current_frame_detections: List[Tuple[int, Tuple[int, int, int, int]]] = []
        self.center_line_position: Optional[int] = None
    
    def _initialize_csv_files(self):
        """Initialize the CSV files with headers."""
        # Low confidence crossings CSV
        if not os.path.exists(self.low_conf_csv):
            with open(self.low_conf_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "location", "date", "video_name", "frame_idx", 
                    "timestamp_sec", "track_id", "species", "confidence",
                    "direction", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                    "image_path", "notes"
                ])
        
        # Potential occlusions CSV
        if not os.path.exists(self.occlusion_csv):
            with open(self.occlusion_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "location", "date", "video_name", "frame_idx",
                    "timestamp_sec", "involved_tracks", "proximity_score",
                    "near_center_line", "image_path", "notes"
                ])
    
    def set_center_line_position(self, center_line: int):
        """Set the center line position for occlusion detection."""
        self.center_line_position = center_line
    
    def observe_detection(self, 
                         frame: np.ndarray,
                         frame_idx: int,
                         timestamp_sec: float,
                         video_name: str,
                         bbox: Tuple[int, int, int, int],
                         pred_class_name: str,
                         pred_confidence: float,
                         track_id: int,
                         direction: Optional[str] = None):
        """
        Observe a detection for potential manual review collection.
        """
        if track_id is None:
            return
        
        # Update track state
        track_state = self.track_states.get(track_id, {
            "ever_high_confidence": False,
            "best_confidence": 0.0,
            "frames_seen": 0,
            "species_votes": deque(maxlen=5),
            "first_seen_frame": frame_idx,
            "last_seen_frame": frame_idx,
            "bbox_history": deque(maxlen=10)
        })
        
        track_state["frames_seen"] += 1
        track_state["last_seen_frame"] = frame_idx
        track_state["species_votes"].append(pred_class_name)
        track_state["bbox_history"].append(bbox)
        track_state["best_confidence"] = max(track_state["best_confidence"], pred_confidence)
        
        # Check if this detection is high confidence
        if pred_confidence >= self.config.lowconf_threshold:
            track_state["ever_high_confidence"] = True
        
        self.track_states[track_id] = track_state
        
        # Store current frame detections for proximity analysis
        self.current_frame_detections.append((track_id, bbox))
    
    def end_frame_processing(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, video_name: str):
        """
        Process end of frame - detect potential occlusions at peak moments only.
        """
        if len(self.current_frame_detections) >= 2:
            # Check for potential occlusions - only returns peak moments
            occlusions = self.proximity_tracker.detect_potential_occlusions(
                self.current_frame_detections, 
                self.center_line_position,
                frame_idx
            )
            
            for occlusion_info in occlusions:
                self._save_occlusion_event(
                    frame, frame_idx, timestamp_sec, video_name, occlusion_info
                )
        
        # Clear frame detections
        self.current_frame_detections.clear()
    
    def flag_crossing_event(self,
                           frame: np.ndarray,
                           frame_idx: int,
                           timestamp_sec: float,
                           video_name: str,
                           bbox: Tuple[int, int, int, int],
                           pred_class_name: str,
                           pred_confidence: float,
                           track_id: int,
                           direction: str):
        """
        Flag a crossing event for review if the fish was never confident during its track.
        
        Only flag crossings where the fish was NEVER confident at any point.
        """
        track_state = self.track_states.get(track_id)
        
        # Only flag if this track was never confident
        if track_state and not track_state.get("ever_high_confidence", False):
            # Determine the most common species from votes
            species_votes = track_state.get("species_votes", [])
            if species_votes:
                species_counts = defaultdict(int)
                for vote in species_votes:
                    species_counts[vote] += 1
                most_common_species = max(species_counts, key=species_counts.get)
            else:
                most_common_species = pred_class_name
            
            self._save_low_confidence_crossing(
                frame, frame_idx, timestamp_sec, video_name, bbox,
                most_common_species, track_state["best_confidence"], 
                track_id, direction, track_state
            )
    
    def _save_low_confidence_crossing(self, frame: np.ndarray, frame_idx: int, 
                                    timestamp_sec: float, video_name: str,
                                    bbox: Tuple[int, int, int, int], species: str,
                                    confidence: float, track_id: int, direction: str,
                                    track_state: Dict[str, Any]):
        """Save a low confidence crossing event."""
        try:
            x1, y1, x2, y2 = bbox
            
            # Create annotated frame
            annotated_frame = frame.copy()
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(annotated_frame, f"LOW CONF CROSSING: {species} {confidence:.2f}", 
                       (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(annotated_frame, f"Track {track_id} | {direction}", 
                       (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Generate filename
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"low_conf_{timestamp_str}_f{frame_idx:06d}_t{track_id}_{species}_{direction}.jpg"
            image_path = os.path.join(self.low_conf_dir, filename)
            
            # Save image
            cv2.imwrite(image_path, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            # Write to CSV
            with open(self.low_conf_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    self.location, self.date_str, video_name, frame_idx,
                    f"{timestamp_sec:.3f}", track_id, species, f"{confidence:.4f}",
                    direction, x1, y1, x2, y2, image_path,
                    f"Never confident during {track_state['frames_seen']} frames"
                ])
                
            print(f"💡 MANUAL REVIEW: Low confidence crossing saved - Track {track_id} ({species}) {direction}")
            
        except Exception as e:
            print(f"Error saving low confidence crossing: {e}")
    
    def _save_occlusion_event(self, frame: np.ndarray, frame_idx: int,
                            timestamp_sec: float, video_name: str,
                            occlusion_info: Dict[str, Any]):
        """Save a potential occlusion event."""
        try:
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Draw all involved tracks
            involved_tracks = occlusion_info["tracks"]
            # yellow, purple, cyan, light green
            colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (128, 255, 128)]
            
            for i, (track_id, bbox) in enumerate(involved_tracks):
                x1, y1, x2, y2 = bbox
                color = colors[i % len(colors)]
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(annotated_frame, f"T{track_id}", 
                           (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Add occlusion warning
            cv2.putText(annotated_frame, "POTENTIAL OCCLUSION", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Generate filename
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            track_ids_str = "_".join(str(t[0]) for t in involved_tracks)
            filename = f"occlusion_{timestamp_str}_f{frame_idx:06d}_tracks_{track_ids_str}.jpg"
            image_path = os.path.join(self.occlusion_dir, filename)
            
            # Save image
            cv2.imwrite(image_path, annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            # Write to CSV
            with open(self.occlusion_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    self.location, self.date_str, video_name, frame_idx,
                    f"{timestamp_sec:.3f}", track_ids_str, 
                    f"{occlusion_info['proximity_score']:.4f}",
                    occlusion_info.get("near_center_line", False), image_path,
                    f"Potential occlusion between {len(involved_tracks)} tracks"
                ])
                
            print(f"🔍 MANUAL REVIEW: Peak occlusion moment saved - Tracks {track_ids_str} (score: {occlusion_info['proximity_score']:.2f})")
            
        except Exception as e:
            print(f"Error saving occlusion event: {e}")
    
    def garbage_collect_inactive(self, current_frame_idx: int):
        """Clean up inactive track states."""
        inactive_tracks = []
        for track_id, state in self.track_states.items():
            # Remove tracks that haven't been seen for a while
            if current_frame_idx - state.get("last_seen_frame", current_frame_idx) > self.config.track_gap_frames:
                inactive_tracks.append(track_id)
        
        for track_id in inactive_tracks:
            self.track_states.pop(track_id, None)
    
    def finalize_processing(self):
        """Finalize processing and save any remaining peak occlusions."""
        # Save any remaining peak occlusions that might not have been captured
        # Use very high threshold to avoid saving weak occlusions at video end
        for occlusion_key, occlusion_data in self.proximity_tracker.active_occlusions.items():
            if occlusion_data["proximity_score"] > self.proximity_tracker.proximity_threshold * 2.0:  # Very high threshold
                print(f"🔍 MANUAL REVIEW: Final peak occlusion saved - {occlusion_key} (score: {occlusion_data['proximity_score']:.2f})")
        
        # Clear active tracking
        self.proximity_tracker.active_occlusions.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about manual review collection."""
        return {
            "active_tracks": len(self.track_states),
            "output_directory": self.config.output_dir,
            "low_conf_csv": self.low_conf_csv,
            "occlusion_csv": self.occlusion_csv,
            "tracks_never_confident": sum(1 for state in self.track_states.values() 
                                        if not state.get("ever_high_confidence", False))
        }


class ProximityTracker:
    """
    Tracks proximity between detections to identify the peak moment of occlusion.
    """
    
    def __init__(self, proximity_threshold: float = 0.6, center_line_buffer: int = 50):
        self.proximity_threshold = proximity_threshold
        self.center_line_buffer = center_line_buffer
        
        # Track ongoing occlusions to find peak moments
        self.active_occlusions: Dict[str, Dict[str, Any]] = {}
        self.occlusion_cooldown: Dict[str, int] = {}  # Prevent multiple captures of same occlusion
        self.cooldown_frames = 120  # 4 seconds at 30fps - longer cooldown to prevent duplicates
    
    def _get_occlusion_key(self, track_ids: List[int]) -> str:
        """Generate a consistent key for a pair of tracks."""
        return "_".join(map(str, sorted(track_ids)))
    
    def _is_in_cooldown(self, occlusion_key: str, current_frame: int) -> bool:
        """Check if this occlusion pair is in cooldown period."""
        last_capture = self.occlusion_cooldown.get(occlusion_key, -1000)
        return current_frame - last_capture < self.cooldown_frames
    
    def _set_cooldown(self, occlusion_key: str, current_frame: int):
        """Set cooldown for this occlusion pair."""
        self.occlusion_cooldown[occlusion_key] = current_frame
    
    def calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_distance(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate normalized distance between centers of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        
        # Normalize by average box size
        avg_size = ((x2_1 - x1_1) + (y2_1 - y1_1) + (x2_2 - x1_2) + (y2_2 - y1_2)) / 4
        return distance / max(avg_size, 1)
    
    def detect_potential_occlusions(self, detections: List[Tuple[int, Tuple[int, int, int, int]]], 
                                  center_line: Optional[int] = None, frame_idx: int = 0) -> List[Dict[str, Any]]:
        """
        Detect potential occlusions between tracks, but only capture single peak moment
        and only BEFORE the center line (where fish might merge before counting).
        
        Args:
            detections: List of (track_id, bbox) tuples
            center_line: Position of center line for filtering occlusions
            frame_idx: Current frame index for tracking occlusion progression
            
        Returns:
            List of occlusion events to save (only peak moments before center line)
        """
        occlusions_to_save = []
        current_occlusions = {}
        
        # Check all pairs of detections
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                track1_id, bbox1 = detections[i]
                track2_id, bbox2 = detections[j]
                
                # Calculate centers
                center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
                center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
                
                # ONLY check occlusions BEFORE the center line
                if center_line is not None:
                    # Both fish must be before (left of) the center line
                    if center1[0] >= center_line or center2[0] >= center_line:
                        continue  # Skip this pair - not before center line
                
                # Calculate proximity metrics
                iou = self.calculate_iou(bbox1, bbox2)
                distance = self.calculate_distance(bbox1, bbox2)
                
                # Combine metrics for proximity score (more aggressive threshold)
                proximity_score = iou + (1.0 / (1.0 + distance))
                
                # Check if this qualifies as potential occlusion (higher threshold)
                if proximity_score > self.proximity_threshold:
                    occlusion_key = self._get_occlusion_key([track1_id, track2_id])
                    
                    # Skip if in cooldown period
                    if self._is_in_cooldown(occlusion_key, frame_idx):
                        continue
                    
                    # Track this occlusion
                    current_occlusions[occlusion_key] = {
                        "tracks": [(track1_id, bbox1), (track2_id, bbox2)],
                        "proximity_score": proximity_score,
                        "iou": iou,
                        "distance": distance,
                        "near_center_line": True,  # All are near center line since we filtered
                        "frame_idx": frame_idx
                    }
        
        # Update active occlusions and detect peaks - improved peak detection
        for occlusion_key, current_data in current_occlusions.items():
            if occlusion_key in self.active_occlusions:
                # Update existing occlusion
                prev_data = self.active_occlusions[occlusion_key]
                
                # More sophisticated peak detection:
                # 1. Score must be significantly high
                # 2. Score must be decreasing by a meaningful amount
                # 3. Previous score must have been a local maximum
                score_decrease = prev_data["proximity_score"] - current_data["proximity_score"]
                is_significant_peak = prev_data["proximity_score"] > self.proximity_threshold * 1.5
                is_meaningful_decrease = score_decrease > 0.1
                
                if is_significant_peak and is_meaningful_decrease:
                    # This was the peak moment - save it and set long cooldown
                    occlusions_to_save.append(prev_data)
                    self._set_cooldown(occlusion_key, frame_idx)
                    # Remove from active tracking to prevent multiple captures
                    del self.active_occlusions[occlusion_key]
                    print(f"📍 Peak occlusion detected for {occlusion_key}: score {prev_data['proximity_score']:.3f} → {current_data['proximity_score']:.3f}")
                    continue
                
                # Update with current data if not a peak
                self.active_occlusions[occlusion_key] = current_data
            else:
                # New occlusion detected
                self.active_occlusions[occlusion_key] = current_data
                print(f"🔍 New occlusion started: {occlusion_key} (score: {current_data['proximity_score']:.3f})")
        
        # Clean up occlusions that are no longer active (fish separated)
        active_keys = set(current_occlusions.keys())
        to_remove = []
        
        for occlusion_key in self.active_occlusions:
            if occlusion_key not in active_keys:
                # Occlusion ended without being captured - only save if it was significant
                last_data = self.active_occlusions[occlusion_key]
                if last_data["proximity_score"] > self.proximity_threshold * 1.8:  # Very high threshold for end-captures
                    occlusions_to_save.append(last_data)
                    self._set_cooldown(occlusion_key, frame_idx)
                    print(f"📍 End-of-occlusion capture for {occlusion_key}: score {last_data['proximity_score']:.3f}")
                to_remove.append(occlusion_key)
        
        for key in to_remove:
            del self.active_occlusions[key]
        
        return occlusions_to_save