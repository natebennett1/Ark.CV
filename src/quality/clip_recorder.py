"""
Video clip recorder for quality assurance events.

This module handles the recording of video clips around QA events,
including pre-event buffering and post-event capture.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Set

from .quality_event import QualityEvent, EventType


class ClipRecorder:
    """
    Records frames for a quality assurance event.
    
    Manages the collection of frames from pre-event buffer through post-event duration,
    tracking the peak moment and handling event merging.
    """
    
    def __init__(self, event: QualityEvent, video_fps: float,
                 pre_event_sec: float, post_event_sec: float,
                 buffered_frames: List[Tuple[np.ndarray, int, float, str]]):
        """
        Initialize a clip recorder.
        
        Args:
            event: The initial quality event triggering this recording
            video_fps: Video frame rate
            pre_event_sec: Seconds before event to include
            post_event_sec: Seconds after event to include
            buffered_frames: Pre-buffered frames to include
        """
        self.video_fps = video_fps
        self.pre_event_sec = pre_event_sec
        self.post_event_sec = post_event_sec
        
        # Event information (can be updated via merging)
        self.events: List[QualityEvent] = [event]
        self.primary_event = event
        
        # Track all detections by frame for accurate annotation
        # This ensures we show ALL fish involved, not just merged ones
        self.track_detections: Dict[int, Tuple[int, int, int, int]] = {}
        for track_id, bbox in event.detections:
            self.track_detections[track_id] = bbox
        
        # Peak information
        self.peak_frame_idx = event.frame_idx
        self.peak_timestamp_sec = event.timestamp_sec
        self.peak_proximity_score = event.proximity_score
        
        # Video info
        self.video_name = buffered_frames[-1][3] if buffered_frames else ""
        
        # Frame collection
        self.frames: List[Tuple[np.ndarray, int, float, str]] = []
        
        # Add pre-event frames from buffer
        pre_event_frames_needed = int(pre_event_sec * video_fps)
        start_idx = max(0, len(buffered_frames) - pre_event_frames_needed)
        self.frames.extend(buffered_frames[start_idx:])
        
        # Recording state
        self.start_frame_idx = self.frames[0][1] if self.frames else event.frame_idx
        self.start_timestamp_sec = self.frames[0][2] if self.frames else event.timestamp_sec
        self.post_event_frames_needed = int(post_event_sec * video_fps)
        self.post_event_frames_recorded = 0
        self.recording_post_event = False
        self.end_frame_idx = self.start_frame_idx
        self.end_timestamp_sec = self.start_timestamp_sec
    
    def merge_event(self, new_event: QualityEvent):
        """
        Merge a new event into this recording.
        
        This happens when overlapping events are detected. The clip will
        be upgraded to capture all relevant information.
        """
        self.events.append(new_event)
        
        # Update track detections with new event's detections
        for track_id, bbox in new_event.detections:
            # Always update with latest bbox for this track
            self.track_detections[track_id] = bbox
        
        # Merge the primary event
        self.primary_event = self.primary_event.merge_with(new_event)
        
        # Update peak if this event is more severe
        if new_event.proximity_score > self.peak_proximity_score:
            self.peak_frame_idx = new_event.frame_idx
            self.peak_timestamp_sec = new_event.timestamp_sec
            self.peak_proximity_score = new_event.proximity_score
    
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
    
    def get_frame_range(self) -> Tuple[int, int]:
        """Get the frame range this clip covers (or will cover)."""
        estimated_end = self.peak_frame_idx + self.post_event_frames_needed
        return (self.start_frame_idx, estimated_end)
    
    def get_all_track_ids(self) -> List[int]:
        """Get all track IDs involved in this clip."""
        return sorted(self.track_detections.keys())
    
    def get_event_types(self) -> List[EventType]:
        """Get all event types captured in this clip."""
        return list(set(event.event_type for event in self.events))
    
    def get_event_summary(self) -> Dict[int, List[str]]:
        """
        Get a summary of which tracks are involved in which event types.
        
        Returns:
            Dict mapping track_id to list of event type descriptions
        """
        track_events: Dict[int, Set[str]] = {}
        
        for event in self.events:
            event_label = event.get_display_label()
            for track_id in event.track_ids:
                if track_id not in track_events:
                    track_events[track_id] = set()
                track_events[track_id].add(event_label)
        
        # Convert sets to sorted lists
        return {tid: sorted(list(labels)) for tid, labels in track_events.items()}
    
    def annotate_frame(self, frame: np.ndarray, frame_idx: int, draw_boxes: bool = False) -> np.ndarray:
        """
        Annotate a frame with event information.
        
        By default, only adds text labels in the top-right corner and doesn't draw
        bounding boxes (since the pipeline already annotates those). Set draw_boxes=True
        to add colored bounding boxes around involved fish.
        
        Args:
            frame: The frame to annotate (will be copied)
            frame_idx: The frame index
            draw_boxes: Whether to draw bounding boxes (default: False to avoid clutter)
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        is_peak = (frame_idx == self.peak_frame_idx)
        frame_height, frame_width = annotated.shape[:2]
        
        # Colors for different tracks
        colors = [
            (0, 255, 255),    # Yellow
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (128, 255, 128),  # Light green
            (255, 128, 0),    # Orange
            (0, 255, 128),    # Spring green
            (255, 0, 128),    # Pink
            (128, 128, 255)   # Light blue
        ]
        
        # Only draw bounding boxes if explicitly requested
        if draw_boxes:
            # Get event summary for labeling
            track_event_summary = self.get_event_summary()
            
            # Draw bounding boxes for ALL involved tracks
            for i, (track_id, bbox) in enumerate(self.track_detections.items()):
                x1, y1, x2, y2 = bbox
                color = colors[i % len(colors)]
                thickness = 4 if is_peak else 2
                
                # Draw bbox
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                
                # Draw track ID
                cv2.putText(annotated, f"T{track_id}", (x1, max(15, y1 - 10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw event types for this track (if multiple events)
                if track_id in track_event_summary and len(self.events) > 1:
                    event_labels = track_event_summary[track_id]
                    for j, event_label in enumerate(event_labels):
                        label_y = y2 + 20 + (j * 20)
                        cv2.putText(annotated, event_label, (x1, label_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add main event label at TOP RIGHT
        event_types = self.get_event_types()
        
        # Calculate starting X position for right-aligned text
        # We'll use right edge minus some margin
        right_margin = 10
        
        if len(event_types) > 1 or EventType.MULTIPLE_ISSUES in event_types:
            label = "MULTIPLE ISSUES"
            y_offset = 40
            
            # Main label (right-aligned)
            label_text = f"{'PEAK ' if is_peak else ''}{label}"
            label_color = (0, 0, 255) if is_peak else (0, 165, 255)
            
            # Measure text to right-align it
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = frame_width - text_size[0] - right_margin
            
            cv2.putText(annotated, label_text, (text_x, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)
            
            # Add sub-labels for each unique event (right-aligned)
            y_offset += 35
            unique_event_labels = set()
            for event in self.events:
                if event.event_type != EventType.MULTIPLE_ISSUES:
                    unique_event_labels.add(event.get_display_label())
            
            for event_label in sorted(unique_event_labels):
                sub_text = f"• {event_label}"
                sub_size = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                sub_x = frame_width - sub_size[0] - right_margin
                
                cv2.putText(annotated, sub_text, (sub_x, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                y_offset += 30
        else:
            # Single event type (right-aligned)
            label = self.primary_event.get_display_label()
            label_text = f"{'PEAK ' if is_peak else ''}{label}"
            label_color = (0, 0, 255) if is_peak else (0, 165, 255)
            
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = frame_width - text_size[0] - right_margin
            
            cv2.putText(annotated, label_text, (text_x, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 3)
        
        # Add detailed info at bottom for each track involved
        bottom_y = annotated.shape[0] - 20
        for event in self.events:
            if event.species or event.confidence is not None or event.direction:
                track_info_parts = []
                if event.track_ids:
                    track_info_parts.append(f"T{','.join(map(str, event.track_ids))}")
                if event.species:
                    track_info_parts.append(event.species)
                if event.confidence is not None:
                    track_info_parts.append(f"conf:{event.confidence:.2f}")
                if event.direction:
                    track_info_parts.append(event.direction)
                
                info_text = " | ".join(track_info_parts)
                cv2.putText(annotated, info_text, (10, bottom_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                bottom_y -= 25  # Stack multiple lines if needed
        
        return annotated