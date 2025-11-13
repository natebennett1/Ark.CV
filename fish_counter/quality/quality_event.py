"""
Quality assurance event types and data structures.

This module defines the different types of QA events that can trigger
video clip collection for manual review.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional


class EventType(Enum):
    """Types of quality assurance events."""
    OCCLUSION = "occlusion"
    BULL_TROUT = "bull_trout"  # Special case: always capture
    UNKNOWN_SPECIES = "unknown_species"
    UNKNOWN_ADIPOSE = "unknown_adipose"
    LOW_CONFIDENCE = "low_confidence"
    MULTIPLE_ISSUES = "multiple_issues"  # Merged event types


class EventPriority(Enum):
    """Priority levels for event handling."""
    CRITICAL = 3  # Bull trout - always capture, no overlap prevention
    HIGH = 2      # Occlusions, unknown species
    MEDIUM = 1    # Low confidence crossings


@dataclass
class QualityEvent:
    """
    Represents a quality assurance event that should be reviewed.
    
    This abstraction allows different event types (occlusions, low-confidence
    crossings, etc.) to be handled uniformly by the clip recording system.
    """
    
    event_type: EventType
    frame_idx: int
    timestamp_sec: float

    proximity_score: float  # Severity/confidence score (0-1)
    track_ids: List[int]
    detections: List[Tuple[int, Tuple[int, int, int, int]]]  # (track_id, bbox)
    
    # Event-specific metadata
    species: Optional[str] = None
    confidence: Optional[float] = None
    direction: Optional[str] = None  # "Upstream" or "Downstream"
    notes: str = ""
    
    def get_priority(self) -> EventPriority:
        """Get the priority level for this event."""
        if self.event_type == EventType.BULL_TROUT:
            return EventPriority.CRITICAL
        elif self.event_type in (EventType.OCCLUSION, EventType.UNKNOWN_SPECIES):
            return EventPriority.HIGH
        else:
            return EventPriority.MEDIUM
    
    def get_event_key(self) -> str:
        """Generate a unique key for this event based on involved tracks."""
        return "_".join(map(str, sorted(self.track_ids)))
    
    def get_display_label(self) -> str:
        """Get a human-readable label for frame annotation."""
        labels = {
            EventType.OCCLUSION: "OCCLUSION",
            EventType.LOW_CONFIDENCE: f"LOW CONF ({self.confidence:.2f})" if self.confidence is not None else "LOW CONF",
            EventType.UNKNOWN_SPECIES: "UNKNOWN SPECIES",
            EventType.UNKNOWN_ADIPOSE: "UNKNOWN ADIPOSE",
            EventType.BULL_TROUT: "BULL TROUT",
            EventType.MULTIPLE_ISSUES: "MULTIPLE ISSUES"
        }
        return labels.get(self.event_type, "QA EVENT")
    
    def merge_with(self, other: 'QualityEvent') -> 'QualityEvent':
        """
        Merge this event with another overlapping event.
        
        Combines information from both events to create a unified clip.
        """
        # Take the higher priority event type, or upgrade to MULTIPLE_ISSUES
        if self.event_type != other.event_type:
            new_type = EventType.MULTIPLE_ISSUES
        else:
            new_type = self.event_type
        
        # Combine track IDs
        combined_tracks = list(set(self.track_ids + other.track_ids))
        
        # Combine detections (deduplicate by track_id)
        detection_map = {tid: bbox for tid, bbox in self.detections}
        for tid, bbox in other.detections:
            detection_map[tid] = bbox  # Later detection overwrites
        combined_detections = list(detection_map.items())
        
        # Use the higher proximity score
        max_score = max(self.proximity_score, other.proximity_score)
        
        # Combine notes
        notes_parts = []
        if self.notes:
            notes_parts.append(self.notes)
        if other.notes:
            notes_parts.append(other.notes)
        combined_notes = "; ".join(notes_parts)
        
        return QualityEvent(
            event_type=new_type,
            frame_idx=min(self.frame_idx, other.frame_idx),  # Use earlier frame
            timestamp_sec=min(self.timestamp_sec, other.timestamp_sec),
            proximity_score=max_score,
            track_ids=combined_tracks,
            detections=combined_detections,
            species=self.species or other.species,
            confidence=min(self.confidence or 1.0, other.confidence or 1.0),
            direction=self.direction or other.direction,
            notes=combined_notes
        )