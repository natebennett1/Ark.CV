"""
Fish state management for tracking individual fish across frames.

This module handles the temporal state of tracked fish, including species voting,
direction detection, and crossing counting.
"""

from dataclasses import dataclass, field
from collections import deque, defaultdict
from typing import Dict, Optional

from ..config.settings import CountingConfig


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
    
    # Crossing detection state
    last_side: Optional[str] = None  # "left", "right", or None (in center zone)
    
    # Anti-oscillation tracking
    last_crossing_direction: Optional[str] = None  # Track last crossing direction
    consecutive_opposite_crossings: int = 0  # Count of back-and-forth crossings
    
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
                oscillation_cooldown = max(cooldown_frames, 30)  # At least 1 second at 30fps
                
                if frames_since_last < oscillation_cooldown:
                    print(f"🚫 OSCILLATION DETECTED: Track {self.track_id} tried to cross {direction} "
                          f"only {frames_since_last} frames after {self.last_crossing_direction}")
                    return False
                
                # Track consecutive opposite crossings
                self.consecutive_opposite_crossings += 1
                
                # If we have too many rapid back-and-forth crossings, be more restrictive
                if self.consecutive_opposite_crossings >= 2:
                    extended_cooldown = 60  # 2 seconds at 30fps
                    if frames_since_last < extended_cooldown:
                        print(f"🚫 EXCESSIVE OSCILLATION: Track {self.track_id} blocked after "
                              f"{self.consecutive_opposite_crossings} rapid reversals")
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