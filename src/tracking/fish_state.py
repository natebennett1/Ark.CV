"""
Fish state management for tracking individual fish across frames.

This module handles the temporal state of tracked fish, including species voting,
direction detection, and crossing counting.
"""

import logging
from collections import deque, defaultdict
from typing import Dict, Optional, Tuple, List

from ..config.settings import CountingConfig

logger = logging.getLogger(__name__)


class FishState:
    """
    State information for a single tracked fish.
    
    This class maintains temporal information about a fish track including
    species classification votes, position history, and crossing counts.
    """

    def __init__(self, track_id: int, last_x: int = 0, last_y: int = 0, 
                 stability_window: int = 3, adipose_window: int = 3):
        self.track_id = track_id
        self.last_x = last_x
        self.last_y = last_y
        self.length_inches = 0.0
        self.last_confidence = 0.0
        self.last_count_frame = -1000000
        self.crossing_count = 0
        
        # Crossing detection state
        self.last_side: Optional[str] = None
        
        # Anti-oscillation tracking
        self.last_crossing_direction: Optional[str] = None
        self.consecutive_opposite_crossings = 0

        # Species detection history: stores species confidence history with no length limit for confidence lookup
        self.species_detection_history: Dict[str, List[float]] = defaultdict(list)

        # Temporal voting window: last N detections for stability
        self.recent_species_detections = deque(maxlen=stability_window)

        # Adipose detection history: stores all adipose detections with confidence (no limit)
        # Format: {"Present": [conf1, conf2, ...], "Absent": [conf1, conf2, ...], "Unknown": [...]}
        self.adipose_detection_history: Dict[str, List[float]] = defaultdict(list)
    
    def update_position(self, x: int, y: int):
        """Update the fish's position."""
        self.last_x = x
        self.last_y = y
    
    def add_species_detection(self, species: str, confidence: float):
        """Add a species detection with its confidence."""
        detection = (species, confidence)
        self.species_detection_history[species].append(confidence)
        self.recent_species_detections.append(detection)
    
    def add_adipose_detection(self, adipose_status: str, confidence: float):
        """Add an adipose fin detection with its confidence."""
        self.adipose_detection_history[adipose_status].append(confidence)
    
    def get_stable_species_with_confidence(self) -> Tuple[str, float]:
        """
        Get the most voted species from recent detections along with its max confidence.
        
        Returns:
            Tuple of (species_name, max_confidence_for_that_species)
        """
        # Group detections by species
        species_detections = defaultdict(list)
        for species, conf in self.recent_species_detections:
            species_detections[species].append(conf)
        
        # Find species with most votes
        vote_counts = {species: len(confs) for species, confs in species_detections.items()}
        
        # Create recency map for tie-breaking
        recency_map = {}
        for i, (species, _) in enumerate(reversed(self.recent_species_detections)):
            if species not in recency_map:
                recency_map[species] = i
        
        # Get winning species (by vote count, ties broken by recency)
        best_species = max(vote_counts.items(), 
                        key=lambda kv: (kv[1], -recency_map[kv[0]]))[0]
        
        # Get max confidence for that species across ALL history (not just recent window)
        all_confidences_for_species = self.species_detection_history[best_species]
        max_conf = max(all_confidences_for_species)
        
        return best_species, max_conf
    
    def get_stable_adipose(self) -> str:
        """
        Get the most confident adipose fin status from entire detection history.
        
        Strategy: Since adipose fins are small and easily obstructed, we prioritize
        any clear detection over temporal voting. If we ever saw "Present" or "Absent"
        with high confidence, use that. Only fall back to "Unknown" if we never got
        a confident detection.
        
        Returns:
            Adipose status: "Present", "Absent", or "Unknown"
        """
        # Collect max confidence for each status (ignoring "Unknown")
        status_confidences = {}
        
        for status in ["Present", "Absent"]:
            if self.adipose_detection_history[status]:
                status_confidences[status] = max(self.adipose_detection_history[status])
        
        # If we have any confident detections, use the one with highest confidence
        if status_confidences:
            best_status = max(status_confidences.items(), key=lambda x: x[1])[0]
            return best_status
        
        # If we only have "Unknown" detections or no detections at all
        return "Unknown"
    
    def can_count(self, current_frame: int, cooldown_frames: int = 0) -> bool:
        """Check if this fish can be counted (respects cooldown)."""
        return current_frame - self.last_count_frame >= cooldown_frames
    
    def should_count_crossing(self, direction: str, current_frame: int, cooldown_frames: int = 0) -> bool:
        """
        Check if a crossing should be counted, considering anti-oscillation logic.
        
        Args:
            direction: Direction of crossing ("Upstream" or "Downstream")
            current_frame: Current frame number
            cooldown_frames: Minimum frames between counts
            
        Returns:
            True if crossing should be counted, False if it's likely an oscillation
        """
        # Basic cooldown check
        if not self.can_count(current_frame, cooldown_frames):
            return False
        
        # Anti-oscillation logic
        if self.last_crossing_direction is not None:
            # Check if this is the opposite direction from the last crossing
            is_opposite = (
                (self.last_crossing_direction == "Upstream" and direction == "Downstream") or
                (self.last_crossing_direction == "Downstream" and direction == "Upstream")
            )
            
            if is_opposite:
                # Calculate time since last crossing
                frames_since_last = current_frame - self.last_count_frame
                
                # If it's too soon after the last crossing, likely an oscillation
                # Use a more aggressive cooldown for opposite directions
                oscillation_cooldown = max(cooldown_frames, 15)  # At least 0.5 seconds at 30fps
                
                if frames_since_last < oscillation_cooldown:
                    logger.info("OSCILLATION DETECTED: Track %d tried to cross %s only %d frames after %s",
                                 self.track_id, direction, frames_since_last, self.last_crossing_direction)
                    return False
                
                # Track consecutive opposite crossings
                self.consecutive_opposite_crossings += 1
                
                # If we have too many rapid back-and-forth crossings, be more restrictive
                if self.consecutive_opposite_crossings >= 2:
                    extended_cooldown = 30  # 1 second at 30fps
                    if frames_since_last < extended_cooldown:
                        logger.info("EXCESSIVE OSCILLATION: Track %d blocked after %d rapid reversals",
                                    self.track_id, self.consecutive_opposite_crossings)
                        return False
            else:
                # Same direction as last crossing, reset oscillation counter
                self.consecutive_opposite_crossings = 0
        
        return True
    
    def record_count(self, frame_number: int, direction: str):
        """Record that this fish was counted."""
        self.crossing_count += 1
        self.last_count_frame = frame_number
        self.last_crossing_direction = direction


class FishStateManager:
    """
    Manages state for all tracked fish.
    
    This class provides a centralized way to manage fish states across
    the video processing pipeline.
    """
    
    def __init__(self, counting_config: CountingConfig):
        self.config = counting_config
        self.fish_states: Dict[int, FishState] = {}
        self.track_trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=counting_config.trail_max_length)
        )
    
    def get_or_create_state(self, track_id: int, initial_x: int = 0, initial_y: int = 0) -> FishState:
        """Get existing fish state or create a new one."""
        if track_id not in self.fish_states:
            state = FishState(
                track_id=track_id,
                last_x=initial_x,
                last_y=initial_y
            )

            # Set proper maxlen for voting queues
            state.recent_species_detections = deque(maxlen=self.config.stability_window)
            
            self.fish_states[track_id] = state
        
        return self.fish_states[track_id]
    
    def update_trail(self, track_id: int, x: int, y: int):
        """Update the trail for a fish."""
        self.track_trails[track_id].append((x, y))
    
    def get_trail(self, track_id: int) -> list:
        """Get the trail points for a fish."""
        return list(self.track_trails[track_id])
    
    def cleanup_inactive_tracks(self, active_track_ids: set):
        """Remove states for tracks that are no longer active."""
        inactive_ids = set(self.fish_states.keys()) - active_track_ids
        
        for track_id in inactive_ids:
            self.fish_states.pop(track_id, None)
            self.track_trails.pop(track_id, None)
    
    def get_state(self, track_id: int) -> Optional[FishState]:
        """Get state for a specific track ID."""
        return self.fish_states.get(track_id)