"""
Tracking manager for direction detection and crossing analysis.

This module handles the high-level tracking logic including direction detection,
center line crossing, and count debouncing.
"""

from typing import Optional, Tuple, Dict, Set, TYPE_CHECKING

from ..config.settings import TrackingConfig
from .fish_state import FishStateManager, FishState


class TrackingManager:
    """
    High-level tracking manager that coordinates fish tracking across frames.
    
    This class handles:
    - Direction detection based on center line crossing
    - Count debouncing and cooldown logic
    - Track state management
    """
    
    def __init__(self, tracking_config: TrackingConfig):
        self.config = tracking_config
        self.state_manager = FishStateManager(tracking_config)
        self.center_line: Optional[int] = None
        self.frame_width: Optional[int] = None
    
    def initialize_frame_info(self, frame_width: int, frame_height: int):
        """Initialize frame dimensions and center line position."""
        self.frame_width = frame_width
        self.center_line = int(frame_width * self.config.center_line_position)
    
    def determine_direction(self, fish_state: FishState, curr_x: int) -> Optional[str]:
        """
        Determine fish movement direction based on center line crossing using stateful detection.
        
        Args:
            fish_state: Fish state object to track crossing state
            curr_x: Current x-coordinate
            
        Returns:
            "Downstream", "Upstream", or None if no crossing detected
        """
        if self.center_line is None or self.frame_width is None:
            return None
        
        # Calculate delta threshold (percentage of frame width)
        delta = int(self.frame_width * self.config.cross_delta_percent)
        
        # Define the zones
        left_zone = self.center_line - delta
        right_zone = self.center_line + delta
        
        # Determine current side
        if curr_x < left_zone:
            current_side = "left"
        elif curr_x > right_zone:
            current_side = "right"
        else:
            current_side = "center"  # In the buffer zone
        
        direction = None
        
        # Check for crossing: only trigger when moving from one clear side to the other clear side
        if fish_state.last_side == "left" and current_side == "right":
            direction = "Downstream"
        elif fish_state.last_side == "right" and current_side == "left":
            direction = "Upstream"
        
        # Update the fish's last known side (only when in a clear zone, not center)
        if current_side in ["left", "right"]:
            fish_state.last_side = current_side
            
        return direction
    
    def process_detection(self, 
                         track_id: int,
                         bbox: Tuple[int, int, int, int],
                         species: str,
                         confidence: float,
                         frame_number: int,
                         adipose_status: Optional[str] = None) -> Tuple[FishState, Optional[str]]:
        """
        Process a single detection and update tracking state.
        
        Args:
            track_id: Unique track identifier
            bbox: Bounding box as (x1, y1, x2, y2)
            species: Detected species
            confidence: Detection confidence
            frame_number: Current frame number
            adipose_status: Optional adipose fin status
            
        Returns:
            Tuple of (fish_state, direction) where direction is crossing direction or None
        """
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Get or create fish state
        fish_state = self.state_manager.get_or_create_state(track_id, center_x, center_y)
        
        # Calculate length
        length = self._calculate_fish_length(x1, y1, x2, y2)
        fish_state.length_inches = length
        fish_state.last_confidence = confidence
        
        # Update voting queues
        fish_state.add_species_vote(species)
        if adipose_status:
            fish_state.add_adipose_vote(adipose_status)
        
        # Detect direction
        direction = self.determine_direction(fish_state, center_x)

        # Update position and trail
        fish_state.update_position(center_x, center_y)
        self.state_manager.update_trail(track_id, center_x, center_y)
        
        return fish_state, direction
    
    def can_count_crossing(self, fish_state: FishState, direction: str, frame_number: int) -> bool:
        """
        Check if a crossing can be counted (respects cooldown and anti-oscillation).
        
        Args:
            fish_state: Fish state object
            direction: Direction of crossing
            frame_number: Current frame number
            
        Returns:
            True if crossing can be counted
        """
        return fish_state.should_count_crossing(direction, frame_number, self.config.count_cooldown_frames)
    
    def record_crossing(self, fish_state: FishState, direction: str, frame_number: int):
        """Record a crossing event for a fish."""
        fish_state.record_count(frame_number, direction)
    
    def cleanup_inactive_tracks(self, active_track_ids: Set[int]):
        """Clean up tracking state for inactive tracks."""
        self.state_manager.cleanup_inactive_tracks(active_track_ids)
    
    def get_fish_state(self, track_id: int) -> Optional[FishState]:
        """Get fish state for a specific track ID."""
        return self.state_manager.get_state(track_id)
    
    def get_trail_points(self, track_id: int) -> list:
        """Get trail points for visualization."""
        return self.state_manager.get_trail(track_id)
    
    def get_all_states(self) -> Dict[int, FishState]:
        """Get all current fish states."""
        return self.state_manager.get_all_states()
    
    def _calculate_fish_length(self, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate fish length from bounding box diagonal."""
        diagonal_pixels = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        # Use config pixels_per_inch if available, otherwise use default
        pixels_per_inch = getattr(self.config, 'pixels_per_inch', 25.253)
        return diagonal_pixels / pixels_per_inch
    
    @property
    def center_line_position(self) -> Optional[int]:
        """Get the center line position in pixels."""
        return self.center_line