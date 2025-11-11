"""
Quality Assurance Video Clip Collector for Manual Review.

This module captures short video clips when quality assurance events occur,
allowing for later manual verification to ensure counting accuracy.

Supported event types:
- Occlusions: Multiple fish bounding boxes get close to each other
- Low confidence crossings: Fish crosses with low species classification confidence
- Unknown species: Fish crosses with no species identification
- Bull trout: Special case - always captured regardless of overlap
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
from .quality_event import QualityEvent, EventType, EventPriority
from .clip_recorder import ClipRecorder
from .occlusion_detector import OcclusionDetector


class ManualReviewCollector:
    """
    Captures video clips of quality assurance events for manual review.
    
    The collector maintains a circular buffer of recent frames, and when a QA event
    is detected, saves a video clip containing frames before, during, and after the event.
    
    Handles multiple event types and prevents clip overlap (except for critical events
    like bull trout).
    """
    
    def __init__(self, config: ManualReviewConfig, location: str, date_str: str, 
                 video_fps: float, frame_width: int, frame_height: int,
                 upstream_direction: str = "right_to_left"):
        self.config = config
        self.location = location
        self.date_str = date_str
        self.video_fps = video_fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.upstream_direction = upstream_direction
        
        # Ensure output directory exists
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        self.clips_dir = os.path.join(self.config.output_dir, "qa_clips")
        Path(self.clips_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata CSV
        self.metadata_csv = os.path.join(self.clips_dir, "qa_clips.csv")
        self._initialize_csv()
        
        # Circular frame buffer for pre-event frames
        buffer_size = int(self.config.clip_pre_event_sec * self.video_fps) + 10

        # Store frame references (not copies) until actually needed for a clip
        self.frame_buffer: Deque[Tuple[np.ndarray, int, float, str]] = deque(maxlen=buffer_size)
        
        # Event detectors
        self.occlusion_detector = OcclusionDetector(
            proximity_threshold=self.config.occlusion_proximity_threshold,
            iou_weight=self.config.occlusion_iou_weight,
            distance_weight=self.config.occlusion_distance_weight,
            upstream_direction=self.upstream_direction
        )
        
        # Active clip recording state
        self.active_clips: Dict[str, ClipRecorder] = {}
        
        # Statistics
        self.stats = {
            "clips_saved": 0,
            "occlusions_detected": 0,
            "low_conf_detected": 0,
            "unknown_species_detected": 0,
            "bull_trout_detected": 0,
            "events_merged": 0,
            "events_skipped": 0  # Skipped due to overlap prevention
        }
        
        # Center line position
        self.center_line_position: Optional[int] = None
    
    def _initialize_csv(self):
        """Initialize the metadata CSV file with headers."""
        if not os.path.exists(self.metadata_csv):
            with open(self.metadata_csv, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "timestamp", "location", "date", "video_name", 
                    "event_types", "start_frame", "peak_frame", "end_frame",
                    "start_time_sec", "peak_time_sec", "end_time_sec",
                    "involved_tracks", "peak_score", "species", "confidence",
                    "direction", "clip_path", "notes"
                ])
    
    def set_center_line_position(self, center_line: int):
        """Set the center line position for event detection."""
        self.center_line_position = center_line
    
    def process_frame(self, frame: np.ndarray, frame_idx: int, timestamp_sec: float, 
                     video_name: str, detections: List[Tuple[int, Tuple[int, int, int, int]]]):
        """
        Process a single frame for QA event detection and clip recording.
        
        Args:
            frame: The current video frame
            frame_idx: Frame number
            timestamp_sec: Timestamp in seconds
            video_name: Name of the source video
            detections: List of (track_id, bbox) tuples for this frame
        """
        # Add frame reference to circular buffer
        if len(detections) >= 1 or len(self.active_clips) > 0:
            self.frame_buffer.append((frame, frame_idx, timestamp_sec, video_name))
        
        # Detect occlusions in current frame
        if len(detections) >= 2:
            occlusion_events = self.occlusion_detector.detect_occlusions(
                detections, 
                self.center_line_position,
                frame_idx,
                timestamp_sec
            )
            
            for event in occlusion_events:
                self.handle_qa_event(event)
                self.stats["occlusions_detected"] += 1
        
        # Update active clip recorders
        self._update_active_clips(frame, frame_idx, timestamp_sec, video_name)
    
    def report_crossing_event(self, track_id: int, bbox: Tuple[int, int, int, int],
                             frame_idx: int, timestamp_sec: float, video_name: str,
                             species: Optional[str], confidence: Optional[float],
                             direction: str):
        """
        Report a fish crossing event for potential QA clip capture.
        
        Args:
            track_id: ID of the crossing fish
            bbox: Bounding box of the fish
            frame_idx: Frame number
            timestamp_sec: Timestamp in seconds
            video_name: Name of the source video
            species: Detected species (or None if unknown)
            confidence: Classification confidence (or None if unknown)
            direction: Crossing direction ("Upstream" or "Downstream")
        """
        # Determine event type
        if species == "BullTrout":
            event_type = EventType.BULL_TROUT
            proximity_score = 1.0  # Maximum priority
            self.stats["bull_trout_detected"] += 1
        elif species is None or species == "Unknown":
            event_type = EventType.UNKNOWN_SPECIES
            proximity_score = 0.9  # High priority
            self.stats["unknown_species_detected"] += 1
        elif confidence is not None and confidence < 0.8:  # Configurable threshold
            event_type = EventType.LOW_CONFIDENCE
            proximity_score = 1.0 - confidence  # Lower confidence = higher score
            self.stats["low_conf_detected"] += 1
        else:
            # Don't create an event for confident crossings
            return
        
        # Create quality event
        event = QualityEvent(
            event_type=event_type,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            proximity_score=proximity_score,
            track_ids=[track_id],
            detections=[(track_id, bbox)],
            species=species,
            confidence=confidence,
            direction=direction,
            notes=f"{direction} crossing"
        )
        
        self.handle_qa_event(event)
    
    def handle_qa_event(self, event: QualityEvent):
        """
        Handle a detected QA event.
        
        Determines whether to start a new clip, merge with existing, or skip.
        """
        event_key = event.get_event_key()
        priority = event.get_priority()
        
        # Check if this exact track combination is already being recorded
        if event_key in self.active_clips:
            # Update existing clip (merge event)
            self.active_clips[event_key].merge_event(event)
            self.stats["events_merged"] += 1
            print(f"🔄 Merged {event.event_type.value} into existing clip {event_key}")
            return
        
        # Check for overlaps with other active clips
        if priority != EventPriority.CRITICAL:
            overlapping_clip = self._find_overlapping_clip(event)
            if overlapping_clip:
                # Merge into the overlapping clip
                overlapping_clip.merge_event(event)
                self.stats["events_merged"] += 1
                print(f"🔄 Merged {event.event_type.value} into overlapping clip")
                return
        
        # Bull trout and other critical events always get their own clip
        print(f"🔍 QA Event detected: {event.event_type.value} - Track {event_key} "
              f"(score: {event.proximity_score:.3f})")
        
        # Deep copy buffered frames now (only when actually creating a clip)
        # This avoids copying frames on every single frame processed
        buffered_frames_copy = [(frame.copy(), fidx, ts, vname) 
                                for frame, fidx, ts, vname in self.frame_buffer]
        
        # Create new clip recorder
        clip_recorder = ClipRecorder(
            event=event,
            video_fps=self.video_fps,
            pre_event_sec=self.config.clip_pre_event_sec,
            post_event_sec=self.config.clip_post_event_sec,
            buffered_frames=buffered_frames_copy
        )
        
        self.active_clips[event_key] = clip_recorder
    
    def _find_overlapping_clip(self, event: QualityEvent) -> Optional[ClipRecorder]:
        """
        Find an active clip that would overlap with this event.
        
        Returns the clip to merge into, or None if no overlap.
        """
        if not self.active_clips:
            return None
        
        # Calculate the frame range this new event would cover
        pre_frames = int(self.config.clip_pre_event_sec * self.video_fps)
        post_frames = int(self.config.clip_post_event_sec * self.video_fps)
        new_start = event.frame_idx - pre_frames
        new_end = event.frame_idx + post_frames
        
        # Check against all active clips
        for clip_recorder in self.active_clips.values():
            clip_start, clip_end = clip_recorder.get_frame_range()
            
            # Check for overlap
            if not (new_end < clip_start or new_start > clip_end):
                return clip_recorder  # Found overlapping clip
        
        return None
    
    def _update_active_clips(self, frame: np.ndarray, frame_idx: int, 
                            timestamp_sec: float, video_name: str):
        """Update all active clip recorders and finalize completed ones."""
        completed_keys = []
        
        for event_key, clip_recorder in self.active_clips.items():
            # Add frame to recorder (copy it since we're storing it)
            clip_recorder.add_frame(frame.copy(), frame_idx, timestamp_sec, video_name)
            
            # Check if recording is complete
            if clip_recorder.is_complete():
                # Save the clip
                self._save_clip(clip_recorder)
                completed_keys.append(event_key)
                
        # Remove completed clips
        for key in completed_keys:
            del self.active_clips[key]
    
    def _save_clip(self, clip_recorder: ClipRecorder):
        """Save a completed QA clip to disk."""
        try:
            # Generate filename
            timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            event_types = "_".join(sorted(set(e.value for e in clip_recorder.get_event_types())))
            track_ids_str = "_".join(str(t) for t in clip_recorder.get_all_track_ids())
            filename = (f"qa_{event_types}_f{clip_recorder.peak_frame_idx:06d}_"
                       f"tracks_{track_ids_str}_{timestamp_str}.mp4")
            clip_path = os.path.join(self.clips_dir, filename)
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*self.config.clip_codec)
            video_writer = cv2.VideoWriter(clip_path, fourcc, self.video_fps, 
                                          (self.frame_width, self.frame_height))
            
            if not video_writer.isOpened():
                print(f"✖ Failed to create video writer for {clip_path}")
                return
            
            # Write all frames with annotations
            # Don't draw bounding boxes - pipeline already annotated them
            # Just add QA event labels in top-right corner
            for frame, frame_idx, _, _ in clip_recorder.frames:
                annotated_frame = clip_recorder.annotate_frame(frame, frame_idx, draw_boxes=False)
                video_writer.write(annotated_frame)
            
            video_writer.release()
            
            # Write metadata to CSV
            self._write_clip_metadata(clip_recorder, clip_path)
            
            self.stats["clips_saved"] += 1
            duration = clip_recorder.end_timestamp_sec - clip_recorder.start_timestamp_sec
            print(f"💾 Saved QA clip: {filename} ({duration:.1f}s, "
                  f"types: {event_types}, score: {clip_recorder.peak_proximity_score:.3f})")
            
        except Exception as e:
            print(f"✖ Error saving QA clip: {e}")
    
    def _write_clip_metadata(self, clip_recorder: ClipRecorder, clip_path: str):
        """Write clip metadata to CSV."""
        try:
            with open(self.metadata_csv, "a", newline="", encoding="utf-8") as f:
                event = clip_recorder.primary_event
                event_types_str = ",".join(sorted(set(e.value for e in clip_recorder.get_event_types())))
                track_ids_str = "_".join(str(t) for t in clip_recorder.get_all_track_ids())
                
                csv.writer(f).writerow([
                    datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    self.location, self.date_str, clip_recorder.video_name,
                    event_types_str,
                    clip_recorder.start_frame_idx, clip_recorder.peak_frame_idx, 
                    clip_recorder.end_frame_idx,
                    f"{clip_recorder.start_timestamp_sec:.3f}",
                    f"{clip_recorder.peak_timestamp_sec:.3f}",
                    f"{clip_recorder.end_timestamp_sec:.3f}",
                    track_ids_str, 
                    f"{clip_recorder.peak_proximity_score:.4f}",
                    event.species or "N/A",
                    f"{event.confidence:.4f}" if event.confidence is not None else "N/A",
                    event.direction or "N/A",
                    clip_path,
                    event.notes
                ])
        except Exception as e:
            print(f"✖ Error writing clip metadata: {e}")
    
    def finalize_processing(self):
        """Finalize processing and save any remaining active clips."""
        print(f"Finalizing manual review collector...")
        
        # Save all remaining active clips
        for event_key, clip_recorder in self.active_clips.items():
            if len(clip_recorder.frames) > 0:
                print(f"Saving remaining clip: {event_key}")
                self._save_clip(clip_recorder)
                
        self.active_clips.clear()
        
        print(f"✔ Manual review finalized: {self.stats['clips_saved']} clips saved")
        print(f"  Events: {self.stats['occlusions_detected']} occlusions, "
              f"{self.stats['low_conf_detected']} low-conf, "
              f"{self.stats['unknown_species_detected']} unknown, "
              f"{self.stats['bull_trout_detected']} bull trout")
        print(f"  Merged: {self.stats['events_merged']}, Skipped: {self.stats['events_skipped']}")