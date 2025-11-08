"""
Local file-based output handler for fish counting pipeline.

This implementation writes results to local CSV files and generates
summary reports, suitable for local development and testing.
"""

import csv
import os
from typing import Dict, Any
from datetime import datetime
from collections import defaultdict

from ..config.settings import IOConfig
from .output_handler import OutputHandler


class LocalOutputHandler(OutputHandler):
    """
    Handles output to local CSV files and summary reports.
    
    This is the default output handler for local development and testing.
    It writes fish count results to CSV files and generates text summaries.
    """
    
    def __init__(self, io_config: IOConfig):
        self.io_config = io_config
        self.csv_path = io_config.csv_output_path
        self.csv_file = None
        self.csv_writer = None
        
        # Statistics tracking
        self.species_counts = defaultdict(lambda: {"Upstream": 0, "Downstream": 0})
        self.total_detections = 0
        self.processing_start_time = datetime.now()
    
    def initialize(self) -> bool:
        """
        Open CSV file for writing with headers.
        
        Returns:
            True if opened successfully, False otherwise
        """
        try:
            self.csv_file = open(self.csv_path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.writer(self.csv_file)
            
            # Write header row
            self.csv_writer.writerow([
                "VideoTimestamp", "Frame", "TrackID", "Species", "Confidence",
                "Direction", "X_Percent", "Y_Percent", "Length_Inches",
                "Location", "Date"
            ])
            
            print(f"✔ CSV output opened: {self.csv_path}")
            return True
            
        except Exception as e:
            print(f"✖ Failed to open CSV file: {e}")
            return False
    
    def record_count(self, species: str, direction: str,
                    frame_number: int, track_id: int, confidence: float,
                    x_percent: float, y_percent: float, length_inches: float,
                    video_timestamp: str = None) -> bool:
        """
        Record a fish count by writing to CSV.
        
        Args:
            species: Classified species
            direction: Movement direction
            frame_number: Frame number
            track_id: Fish track ID
            confidence: Detection confidence
            x_percent: X position as percentage of frame width
            y_percent: Y position as percentage of frame height
            length_inches: Fish length in inches
            video_timestamp: Formatted timestamp string (HH:MM:SS.fff)
            
        Returns:
            True if write successful, False otherwise
        """
        if self.csv_writer is None:
            print("✖ CSV writer not initialized")
            return False
        
        try:
            self.csv_writer.writerow([
                video_timestamp or "",
                frame_number,
                track_id,
                species,
                f"{confidence:.2f}",
                direction,
                f"{x_percent:.1f}",
                f"{y_percent:.1f}",
                f"{length_inches:.1f}",
                self.io_config.location,
                self.io_config.date_str
            ])
            
            # Update statistics
            self.species_counts[species][direction] += 1
            self.total_detections += 1
            
            return True
            
        except Exception as e:
            print(f"✖ Error writing CSV record: {e}")
            return False
    
    def write_final_counts(self):
        """No-op for local handler since counts are written immediately."""
        return True

    def get_species_counts(self) -> Dict[str, Dict[str, int]]:
        """Get current species count statistics."""
        return dict(self.species_counts)
    
    def finalize(self) -> bool:
        """
        Close CSV file and write summary report.
        
        Returns:
            True if finalization successful, False otherwise
        """
        success = True
        
        self.print_final_summary()

        # Close CSV file
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
        
        # Write summary report
        if not self._write_summary_report():
            success = False
        
        return success
    
    def _write_summary_report(self) -> bool:
        """
        Write a summary report file alongside the CSV output.
        
        Returns:
            True if summary written successfully, False otherwise
        """
        try:
            # Place summary in the same directory as the CSV
            csv_dir = os.path.dirname(self.csv_path)
            csv_basename = os.path.basename(self.csv_path)
            summary_path = os.path.join(csv_dir, csv_basename.replace('.csv', '_summary.txt'))
            
            processing_time = (datetime.now() - self.processing_start_time).total_seconds()
            total_upstream = sum(counts["Upstream"] for counts in self.species_counts.values())
            total_downstream = sum(counts["Downstream"] for counts in self.species_counts.values())
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Fish Counting Summary Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'=' * 50}\n\n")
                
                f.write(f"Location: {self.io_config.location}\n")
                f.write(f"Date: {self.io_config.date_str}\n")
                f.write(f"Processing Time: {processing_time:.1f} seconds\n\n")
                
                f.write(f"Total Fish Counted: {self.total_detections}\n")
                f.write(f"Upstream: {total_upstream}\n")
                f.write(f"Downstream: {total_downstream}\n\n")
                
                f.write("Species Breakdown:\n")
                f.write("-" * 30 + "\n")
                
                for species, counts in sorted(self.species_counts.items()):
                    total_species = counts['Upstream'] + counts['Downstream']
                    f.write(f"{species:20s} Total: {total_species:3d} (Up: {counts['Upstream']:3d}, Down: {counts['Downstream']:3d})\n")
                
                f.write(f"\nDetailed results: {os.path.basename(self.csv_path)}\n")
            
            print(f"✔ Summary report written: {summary_path}")
            return True
            
        except Exception as e:
            print(f"✖ Error writing summary report: {e}")
            return False