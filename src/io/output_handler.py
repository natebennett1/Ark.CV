"""
Abstract base class for output handling in fish counting pipeline.

This module defines the minimal interface that all output handlers must implement,
allowing the pipeline to work with different output destinations (local files,
cloud databases, etc.) without changing the core logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class OutputHandler(ABC):
    """
    Abstract base class for handling fish counting outputs.
    
    Implementations can write to local files, cloud databases, or any other
    destination while maintaining a consistent interface for the pipeline.
    """
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the output handler and prepare for writing.
        
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def record_count(self, species: str, direction: str, 
                    frame_number: int, track_id: int, confidence: float,
                    x_percent: float, y_percent: float, length_inches: float,
                    video_timestamp: str = None) -> bool:
        """
        Record a single fish count.
        
        This is the core method that all handlers must implement. Local handlers
        may write to CSV immediately, while cloud handlers may just accumulate counts.
        
        Args:
            species: Classified species
            direction: Movement direction ("Upstream" or "Downstream")
            frame_number: Frame number
            track_id: Fish track ID
            confidence: Detection confidence
            x_percent: X position as percentage of frame width
            y_percent: Y position as percentage of frame height
            length_inches: Fish length in inches
            video_timestamp: Optional formatted timestamp (for local CSV)
            
        Returns:
            True if record successful, False otherwise
        """
        pass
    
    @abstractmethod
    def write_final_counts(self) -> bool:
        """
        Write final aggregated counts to the output destination.
        
        For local handlers, this may be a no-op since counts are written immediately.
        For cloud handlers, this writes the accumulated counts to DynamoDB.
        
        Returns:
            True if write successful, False otherwise
        """
        pass

    @abstractmethod
    def get_species_counts(self) -> Dict[str, Dict[str, int]]:
        """
        Get current species count statistics.
        
        Returns:
            Dictionary mapping species to direction counts
            Example: {"Chinook": {"Upstream": 5, "Downstream": 2}}
        """
        pass
    
    @abstractmethod
    def finalize(self) -> bool:
        """
        Finalize output handling and flush any pending writes.
        
        For local handlers: close files, write summaries
        For cloud handlers: write aggregated counts to DynamoDB
        
        Returns:
            True if finalization successful, False otherwise
        """
        pass
    
    def format_video_timestamp(self, frame_number: int, fps: float) -> str:
        """
        Format frame number as timestamp string.
        
        Default implementation - override if needed.
        
        Args:
            frame_number: Current frame number
            fps: Video frames per second
            
        Returns:
            Formatted timestamp string
        """
        if fps <= 0:
            return "00:00:00.000"
        
        total_seconds = frame_number / fps
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    
    def print_final_summary(self):
        """
        Print final summary to console.
        
        Default implementation - override if needed.
        """
        counts = self.get_species_counts()
        total_upstream = sum(c["Upstream"] for c in counts.values())
        total_downstream = sum(c["Downstream"] for c in counts.values())
        
        print(f"Printing final counts.")
        for species, dirs in sorted(counts.items()):
            print(f"{species}: Up={dirs['Upstream']} Down={dirs['Downstream']}")
        print(f"{'TOTAL'}: Up={total_upstream} Down={total_downstream}")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.initialize():
            raise RuntimeError("Failed to initialize output handler")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.finalize()