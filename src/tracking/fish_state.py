"""
Fish state management for tracking individual fish across frames.

This module handles the temporal state of tracked fish, including species voting,
direction detection, and crossing counting.
"""

from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Dict, Optional

from ..config.settings import TrackingConfig


@dataclass
class FishState:
    """
    State information for a single tracked fish.
    
    This class maintains temporal information about a fish track including
    species classification votes, position history, and crossing counts.
    """
    
    track_id: int
    last_x: int = 0
    last_y: int = 0
    length_inches: float = 0.0
    last_confidence: float = 0.0
    last_count_frame: int = -1000000  # Frame of last count event
    crossing_count: int = 0
    
    # Temporal voting queues
    species_votes: deque = field(default_factory=deque)
    adipose_votes: deque = field(default_factory=deque)
    
    def __post_init__(self):
        """Initialize deques with proper maxlen if not set."""
        if not hasattr(self.species_votes, 'maxlen') or self.species_votes.maxlen is None:
            # Convert to proper deque with maxlen
            items = list(self.species_votes) if self.species_votes else []
            self.species_votes = deque(items, maxlen=3)  # Default window
            
        if not hasattr(self.adipose_votes, 'maxlen') or self.adipose_votes.maxlen is None:
            items = list(self.adipose_votes) if self.adipose_votes else []
            self.adipose_votes = deque(items, maxlen=3)  # Default window
    
    def update_position(self, x: int, y: int):
        """Update the fish's position."""
        self.last_x = x
        self.last_y = y
    
    def add_species_vote(self, species: str):
        """Add a species classification vote."""
        self.species_votes.append(species)
    
    def add_adipose_vote(self, adipose_status: str):
        """Add an adipose fin status vote."""
        if adipose_status and adipose_status != "Unknown":
            self.adipose_votes.append(adipose_status)
    
    def get_stable_species(self) -> Optional[str]:
        """Get the most voted species classification."""
        if not self.species_votes:
            return None
        
        # Count votes
        vote_counts = defaultdict(int)
        for vote in self.species_votes:
            vote_counts[vote] += 1
        
        if not vote_counts:
            return None
        
        # Break ties by most recent vote
        best_species = max(vote_counts.items(), 
                          key=lambda kv: (kv[1], list(self.species_votes)[::-1].index(kv[0])))
        return best_species[0]
    
    def get_stable_adipose(self) -> Optional[str]:
        """Get the most voted adipose fin status."""
        if not self.adipose_votes:
            return None
        
        vote_counts = defaultdict(int)
        for vote in self.adipose_votes:
            vote_counts[vote] += 1
        
        if not vote_counts:
            return None
        
        best_adipose = max(vote_counts.items(),
                          key=lambda kv: (kv[1], list(self.adipose_votes)[::-1].index(kv[0])))
        return best_adipose[0]
    
    def can_count(self, current_frame: int, cooldown_frames: int = 0) -> bool:
        """Check if this fish can be counted (respects cooldown)."""
        return current_frame - self.last_count_frame >= cooldown_frames
    
    def record_count(self, frame_number: int):
        """Record that this fish was counted."""
        self.crossing_count += 1
        self.last_count_frame = frame_number


class FishStateManager:
    """
    Manages state for all tracked fish.
    
    This class provides a centralized way to manage fish states across
    the video processing pipeline.
    """
    
    def __init__(self, tracking_config: TrackingConfig):
        self.config = tracking_config
        self.fish_states: Dict[int, FishState] = {}
        self.track_trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=tracking_config.trail_max_length)
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
            state.species_votes = deque(maxlen=self.config.stability_window)
            state.adipose_votes = deque(maxlen=self.config.adipose_window)
            
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
    
    def get_all_states(self) -> Dict[int, FishState]:
        """Get all current fish states."""
        return self.fish_states.copy()
    
    def get_state(self, track_id: int) -> Optional[FishState]:
        """Get state for a specific track ID."""
        return self.fish_states.get(track_id)
    
    def calculate_fish_length(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate fish length from bounding box."""
        diagonal_pixels = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return diagonal_pixels / self.config.pixels_per_inch if hasattr(self.config, 'pixels_per_inch') else diagonal_pixels / 25.253


def majority_vote(vote_queue: deque) -> Optional[str]:
    """
    Perform majority vote on a queue of votes.
    
    Args:
        vote_queue: Deque containing votes
        
    Returns:
        The winning vote, or None if queue is empty
    """
    if not vote_queue:
        return None
    
    # Count votes
    counts = defaultdict(int)
    for vote in vote_queue:
        counts[vote] += 1
    
    # Break ties by most recent vote
    best = max(counts.items(), 
              key=lambda kv: (kv[1], list(vote_queue)[::-1].index(kv[0])))
    return best[0]