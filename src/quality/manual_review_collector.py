"""
Occlusion Video Clip Collector for Quality Assurance.

This module captures short video clips when potential fish occlusions occur,
allowing for later manual verification to ensure counting accuracy.

An occlusion is detected when:
- Multiple fish bounding boxes get close to each other (based on IoU and distance)
- Fish are approaching or near the center line (where counts happen)
"""

import os
import csv
import cv2
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Deque
from collections import deque

from ..config.settings import ManualReviewConfig


class ManualReviewCollector:
    """
    Captures video clips of potential fish occlusions for manual review.
    
    The collector maintains a circular buffer of recent frames, and when an occlusion
    is detected, saves a video clip containing frames before, during, and after the event.
    """
    
    def __init__(self, config: ManualReviewConfig, location: str, date_str: str, 
                 video_fps: float, frame_width: int, frame_height: int):
        self.config = config
        self.location = location
        self.date_str = date_str
        self.video_fps = video_fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self.clips_dir = os.path.join(self.config.output_dir, "occlusion_clips")
        Path(self.clips_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata CSV
        self.metadata_csv = os.path.join(self.clips_dir, "occlusion_clips.csv")
        self._initialize_csv()
        
        # Circular frame buffer for pre-event frames
        # Buffer enough frames for the pre-event duration
        buffer_size = int(self.config.clip_pre_event_sec * self.video_fps) + 10
        self.frame_buffer: Deque[Tuple[np.ndarray, int, float, str]] = deque(maxlen=buffer_size)
        
        # Occlusion detection and tracking
        self.proximity_detector = OcclusionDetector(
            proximity_threshold=self.config.occlusion_proximity_threshold,
            iou_weight=self.config.occlusion_iou_weight,
            distance_weight=self.config.occlusion_distance_weight
        )
        
        # Active clip recording state
        self.active_clips: Dict[str, ActiveClipRecorder] = {}
        
        # Statistics
        self.clips_saved = 0
        self.occlusions_detected = 0
        self.occlusions_skipped = 0  # Skipped due to overlap prevention
        
        # Center line position
        self.center_line_position: Optional[int] = None
    
    def _initialize_csv(self):
        """Initialize the metadata CSV file with headers."""
        if not os.path.exists(self.metadata_csv):
            with open(self.metadata_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "location", "date", "video_name", 
                    "start_frame", "peak_frame", "end_frame",
                    "start_time_sec", "peak_time_sec", "end_time_sec",
                    "involved_tracks", "peak_proximity_score", 
                    "clip_path", "notes"
                ])
    
    def set_center_line_position(self, center_line: int):
        """Set the center line position for occlusion detection."""
        self.center_line_position = center_line
    
    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, 
                     video_name: str, detections: List[Tuple[int, Tuple[int, int, int, int]]]):
        """
        Process a single frame for occlusion detection and clip recording.
        
        Args:
            frame: The current video frame
            frame_idx: Frame number
            timestamp_sec: Timestamp in seconds
            video_name: Name of the source video
            detections: List of (track_id, bbox) tuples for this frame
        """
        # Add frame to circular buffer
        # Only buffer frames when fish are detected or clips are actively recording
        # This significantly reduces memory copies during empty sections of video
        if len(detections) >= 1 or len(self.active_clips) > 0:
            self.frame_buffer.append((frame.copy(), frame_idx, timestamp_sec, video_name))
        
        # Detect occlusions in current frame
        if len(detections) >= 2:
            occlusions = self.proximity_detector.detect_occlusions(
                detections, 
                self.center_line_position,
                frame_idx
            )
            
            for occlusion_event in occlusions:
                self._handle_occlusion_event(occlusion_event, frame_idx, timestamp_sec)
        
        # Update active clip recorders
        self._update_active_clips(frame, frame_idx, timestamp_sec, video_name)
    
    def _handle_occlusion_event(self, occlusion_event: Dict[str, Any], 
                               frame_idx: int, timestamp_sec: float):
        """Handle a detected occlusion event."""
        occlusion_key = occlusion_event["occlusion_key"]
        
        # Check if we're already recording this occlusion
        if occlusion_key in self.active_clips:
            # Update peak if this is stronger
            clip_recorder = self.active_clips[occlusion_key]
            if occlusion_event["proximity_score"] > clip_recorder.peak_proximity_score:
                clip_recorder.update_peak(frame_idx, timestamp_sec, occlusion_event)
        else:
            # Check if this new occlusion would overlap with any active clips
            if self._would_overlap_with_active_clips(frame_idx):
                # Skip this occlusion to avoid overlapping clips
                self.occlusions_skipped += 1
                print(f"⏭️ Skipping occlusion {occlusion_key} - would overlap with active clip")
                return
            
            # Start new clip recording
            self.occlusions_detected += 1
            print(f"🔍 Occlusion detected: {occlusion_key} (score: {occlusion_event['proximity_score']:.3f})")
            
            # Create new clip recorder with buffered frames
            clip_recorder = ActiveClipRecorder(
                occlusion_key=occlusion_key,
                occlusion_event=occlusion_event,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                video_fps=self.video_fps,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                config=self.config,
                buffered_frames=list(self.frame_buffer)  # Copy current buffer
            )
            
            self.active_clips[occlusion_key] = clip_recorder
    
    def _would_overlap_with_active_clips(self, new_frame_idx: int) -> bool:
        """
        Check if a new occlusion starting at new_frame_idx would overlap with any active clips.
        
        A clip starting at new_frame_idx will eventually span from:
        (new_frame_idx - pre_event_frames) to (new_frame_idx + post_event_frames)
        
        We check if this range would overlap with any currently recording clip's range.
        """
        if not self.active_clips:
            return False
        
        # Calculate the frame range this new clip would cover
        pre_event_frames = int(self.config.clip_pre_event_sec * self.video_fps)
        post_event_frames = int(self.config.clip_post_event_sec * self.video_fps)
        new_start = new_frame_idx - pre_event_frames
        new_end = new_frame_idx + post_event_frames
        
        # Check against all active clips
        for clip_recorder in self.active_clips.values():
            # Get the frame range of the active clip
            active_start = clip_recorder.start_frame_idx
            active_end = clip_recorder.peak_frame_idx + post_event_frames  # Estimated end
            
            # Check for overlap: two ranges overlap if one starts before the other ends
            if not (new_end < active_start or new_start > active_end):
                return True  # Overlap detected
        
        return False
    
    def _update_active_clips(self, frame: np.ndarray, frame_idx: int, 
                            timestamp_sec: float, video_name: str):
        """Update all active clip recorders and finalize completed ones."""
        completed_keys = []
        
        for occlusion_key, clip_recorder in self.active_clips.items():
            # Add frame to recorder
            clip_recorder.add_frame(frame, frame_idx, timestamp_sec, video_name)
            
            # Check if recording is complete
            if clip_recorder.is_complete():
                # Save the clip
                self._save_clip(clip_recorder)
                completed_keys.append(occlusion_key)
        
        # Remove completed clips
        for key in completed_keys:
            del self.active_clips[key]
    
    def _save_clip(self, clip_recorder: 'ActiveClipRecorder'):
        """Save a completed occlusion clip to disk."""
        try:
            # Generate filename
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            track_ids_str = "_".join(str(t) for t in clip_recorder.track_ids)
            filename = f"occlusion_{timestamp_str}_f{clip_recorder.peak_frame_idx:06d}_tracks_{track_ids_str}.mp4"
            clip_path = os.path.join(self.clips_dir, filename)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config.clip_codec)
            video_writer = cv2.VideoWriter(clip_path, fourcc, self.video_fps, 
                                          (self.frame_width, self.frame_height))
            
            if not video_writer.isOpened():
                print(f"✖ Failed to create video writer for {clip_path}")
                return
            
            # Write all frames with annotations
            for frame, frame_idx, _, _ in clip_recorder.frames:
                annotated_frame = self._annotate_occlusion_frame(
                    frame.copy(), 
                    clip_recorder.involved_detections,
                    frame_idx == clip_recorder.peak_frame_idx
                )
                video_writer.write(annotated_frame)
            
            video_writer.release()
            
            # Write metadata to CSV
            self._write_clip_metadata(clip_recorder, clip_path)
            
            self.clips_saved += 1
            duration = clip_recorder.end_timestamp_sec - clip_recorder.start_timestamp_sec
            print(f"💾 Saved occlusion clip: {filename} ({duration:.1f}s, score: {clip_recorder.peak_proximity_score:.3f})")
            
        except Exception as e:
            print(f"✖ Error saving occlusion clip: {e}")
    
    def _annotate_occlusion_frame(self, frame: np.ndarray, 
                                 detections: List[Tuple[int, Tuple[int, int, int, int]]],
                                 is_peak: bool) -> np.ndarray:
        """Annotate a frame with occlusion detection information."""
        # Colors for different tracks (yellow, cyan, magenta, light green)
        colors = [(0, 255, 255), (255, 255, 0), (255, 0, 255), (128, 255, 128)]
        
        # Draw bounding boxes for involved tracks
        for i, (track_id, bbox) in enumerate(detections):
            x1, y1, x2, y2 = bbox
            color = colors[i % len(colors)]
            thickness = 4 if is_peak else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"T{track_id}", (x1, max(15, y1 - 10)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add warning label
        label = "PEAK OCCLUSION" if is_peak else "OCCLUSION EVENT"
        label_color = (0, 0, 255) if is_peak else (0, 165, 255)
        cv2.putText(frame, label, (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)
        
        return frame
    
    def _write_clip_metadata(self, clip_recorder: 'ActiveClipRecorder', clip_path: str):
        """Write clip metadata to CSV."""
        try:
            with open(self.metadata_csv, "a", newline="", encoding="utf-8") as f:
                track_ids_str = "_".join(str(t) for t in clip_recorder.track_ids)
                csv.writer(f).writerow([
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    self.location, self.date_str, clip_recorder.video_name,
                    clip_recorder.start_frame_idx, clip_recorder.peak_frame_idx, 
                    clip_recorder.end_frame_idx,
                    f"{clip_recorder.start_timestamp_sec:.3f}",
                    f"{clip_recorder.peak_timestamp_sec:.3f}",
                    f"{clip_recorder.end_timestamp_sec:.3f}",
                    track_ids_str, f"{clip_recorder.peak_proximity_score:.4f}",
                    clip_path, f"Occlusion between {len(clip_recorder.track_ids)} tracks"
                ])
        except Exception as e:
            print(f"✖ Error writing clip metadata: {e}")
    
    def finalize_processing(self):
        """Finalize processing and save any remaining active clips."""
        print(f"Finalizing manual review collector...")
        
        # Save all remaining active clips
        for occlusion_key, clip_recorder in self.active_clips.items():
            if len(clip_recorder.frames) > 0:
                print(f"Saving remaining clip: {occlusion_key}")
                self._save_clip(clip_recorder)
        
        self.active_clips.clear()
        
        print(f"✔ Manual review finalized: {self.clips_saved} clips saved, "
              f"{self.occlusions_detected} occlusions detected, "
              f"{self.occlusions_skipped} skipped (overlap prevention)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about manual review collection."""
        return {
            "clips_saved": self.clips_saved,
            "occlusions_detected": self.occlusions_detected,
            "occlusions_skipped": self.occlusions_skipped,
            "active_clips": len(self.active_clips),
            "output_directory": self.clips_dir,
            "metadata_csv": self.metadata_csv
        }


class ActiveClipRecorder:
    """
    Records frames for an active occlusion event.
    
    Manages the collection of frames from pre-event buffer through post-event duration,
    tracking the peak occlusion moment for annotation purposes.
    """
    
    def __init__(self, occlusion_key: str, occlusion_event: Dict[str, Any],
                 frame_idx: int, timestamp_sec: float, video_fps: float,
                 frame_width: int, frame_height: int, config: ManualReviewConfig,
                 buffered_frames: List[Tuple[np.ndarray, int, float, str]]):
        self.occlusion_key = occlusion_key
        self.video_fps = video_fps
        self.config = config
        
        # Peak information
        self.peak_frame_idx = frame_idx
        self.peak_timestamp_sec = timestamp_sec
        self.peak_proximity_score = occlusion_event["proximity_score"]
        self.track_ids = [t[0] for t in occlusion_event["tracks"]]
        self.involved_detections = occlusion_event["tracks"]
        
        # Video info
        self.video_name = buffered_frames[-1][3] if buffered_frames else ""
        
        # Frame collection
        self.frames: List[Tuple[np.ndarray, int, float, str]] = []
        
        # Add pre-event frames from buffer
        pre_event_frames_needed = int(config.clip_pre_event_sec * video_fps)
        start_idx = max(0, len(buffered_frames) - pre_event_frames_needed)
        self.frames.extend(buffered_frames[start_idx:])
        
        # Recording state
        self.start_frame_idx = self.frames[0][1] if self.frames else frame_idx
        self.start_timestamp_sec = self.frames[0][2] if self.frames else timestamp_sec
        self.post_event_frames_needed = int(config.clip_post_event_sec * video_fps)
        self.post_event_frames_recorded = 0
        self.recording_post_event = False
    
    def update_peak(self, frame_idx: int, timestamp_sec: float, occlusion_event: Dict[str, Any]):
        """Update the peak moment if a stronger occlusion is detected."""
        self.peak_frame_idx = frame_idx
        self.peak_timestamp_sec = timestamp_sec
        self.peak_proximity_score = occlusion_event["proximity_score"]
        self.involved_detections = occlusion_event["tracks"]
    
    def add_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, video_name: str):
        """Add a frame to the recording."""
        # Check if we've reached the peak and should start post-event recording
        if frame_idx > self.peak_frame_idx and not self.recording_post_event:
            self.recording_post_event = True
        
        if self.recording_post_event:
            self.post_event_frames_recorded += 1
        
        self.frames.append((frame.copy(), frame_idx, timestamp_sec, video_name))
        self.end_frame_idx = frame_idx
        self.end_timestamp_sec = timestamp_sec
    
    def is_complete(self) -> bool:
        """Check if we've recorded enough post-event frames."""
        return self.recording_post_event and self.post_event_frames_recorded >= self.post_event_frames_needed


class OcclusionDetector:
    """
    Detects potential occlusions between fish based on bounding box proximity.
    
    An occlusion is detected when multiple fish bounding boxes are close to each other,
    measured by a combination of Intersection over Union (IoU) and centroid distance.
    """
    
    def __init__(self, proximity_threshold: float = 0.3, 
                 iou_weight: float = 0.6, distance_weight: float = 0.4):
        self.proximity_threshold = proximity_threshold
        self.iou_weight = iou_weight
        self.distance_weight = distance_weight
        
        # Track ongoing occlusions to avoid duplicate clips
        self.active_occlusions: Dict[str, int] = {}  # occlusion_key -> last_frame_idx
        self.recorded_pairs: set = set()
    
    def detect_occlusions(self, detections: List[Tuple[int, Tuple[int, int, int, int]]], 
                         center_line: Optional[int], frame_idx: int) -> List[Dict[str, Any]]:
        """
        Detect occlusions in the current frame.
        
        Args:
            detections: List of (track_id, bbox) tuples
            center_line: Position of center line (occlusions near center line are prioritized)
            frame_idx: Current frame index
            
        Returns:
            List of occlusion events detected in this frame
        """
        occlusions = []
        
        # Check all pairs of detections
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                track1_id, bbox1 = detections[i]
                track2_id, bbox2 = detections[j]
                
                # Calculate proximity score
                iou = self._calculate_iou(bbox1, bbox2)
                distance_score = self._calculate_distance_score(bbox1, bbox2)
                proximity_score = (self.iou_weight * iou + 
                                 self.distance_weight * distance_score)
                
                # Check if this qualifies as an occlusion
                if proximity_score > self.proximity_threshold:
                    occlusion_key = self._get_occlusion_key([track1_id, track2_id])
                    
                    # Check if we've already recorded this pair
                    if occlusion_key in self.recorded_pairs:
                        continue
                    
                    # Calculate center positions for prioritization
                    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
                    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
                    
                    # Check if near center line
                    near_center_line = False
                    if center_line is not None:
                        # Consider "near" as within 100 pixels of center line
                        buffer = 100
                        near_center_line = (abs(center1[0] - center_line) < buffer or 
                                          abs(center2[0] - center_line) < buffer)
                    
                    # Record this occlusion
                    self.active_occlusions[occlusion_key] = frame_idx
                    self.recorded_pairs.add(occlusion_key)
                    
                    occlusions.append({
                        "occlusion_key": occlusion_key,
                        "tracks": [(track1_id, bbox1), (track2_id, bbox2)],
                        "proximity_score": proximity_score,
                        "iou": iou,
                        "distance_score": distance_score,
                        "near_center_line": near_center_line,
                        "frame_idx": frame_idx
                    })
        
        return occlusions
    
    def _get_occlusion_key(self, track_ids: List[int]) -> str:
        """Generate a consistent key for a pair of tracks."""
        return "_".join(map(str, sorted(track_ids)))
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                       box2: Tuple[int, int, int, int]) -> float:
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
    
    def _calculate_distance_score(self, box1: Tuple[int, int, int, int], 
                                  box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate normalized distance score between bounding box centers.
        Returns a score from 0 to 1, where 1 means very close, 0 means far apart.
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate centers
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        # Calculate Euclidean distance
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        
        # Normalize by average box diagonal (rough measure of fish size)
        diag1 = ((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)**0.5
        diag2 = ((x2_2 - x1_2)**2 + (y2_2 - y1_2)**2)**0.5
        avg_diag = (diag1 + diag2) / 2
        
        # Convert to a score (closer = higher score)
        # Use exponential decay: score = exp(-distance/avg_diag)
        if avg_diag > 0:
            normalized_distance = distance / avg_diag
            return np.exp(-normalized_distance)
        return 0.0
