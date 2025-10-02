"""
Output writing utilities for CSV and results management.

This module handles writing fish count results to CSV files and managing
output data in a format suitable for analysis and cloud storage.
"""

import csv
import os
from typing import Dict, Any
from datetime import datetime
from collections import defaultdict

from ..config.settings import IOConfig


class OutputWriter:
    """
    Manages output writing for fish counting results.
    
    This class handles CSV output, summary statistics, and result formatting
    in a way that's compatible with cloud storage and batch processing.
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
    
    def open_csv(self) -> bool:
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
    
    def write_count_record(self,
                          video_timestamp: str,
                          frame_number: int,
                          track_id: int,
                          species: str,
                          confidence: float,
                          direction: str,
                          x_percent: float,
                          y_percent: float,
                          length_inches: float) -> bool:
        """
        Write a single count record to the CSV file.
        
        Args:
            video_timestamp: Formatted timestamp string (HH:MM:SS.fff)
            frame_number: Frame number
            track_id: Fish track ID
            species: Classified species
            confidence: Detection confidence
            direction: Movement direction
            x_percent: X position as percentage of frame width
            y_percent: Y position as percentage of frame height
            length_inches: Fish length in inches
            
        Returns:
            True if write successful, False otherwise
        """
        if self.csv_writer is None:
            print("✖ CSV writer not initialized")
            return False
        
        try:
            self.csv_writer.writerow([
                video_timestamp,
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
    
    def get_species_counts(self) -> Dict[str, Dict[str, int]]:
        """Get current species count statistics."""
        return dict(self.species_counts)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for the processing session.
        
        Returns:
            Dictionary containing processing statistics
        """
        processing_time = (datetime.now() - self.processing_start_time).total_seconds()
        
        total_upstream = sum(counts["Upstream"] for counts in self.species_counts.values())
        total_downstream = sum(counts["Downstream"] for counts in self.species_counts.values())
        
        return {
            "processing_time_seconds": processing_time,
            "total_fish_counted": self.total_detections,
            "total_upstream": total_upstream,
            "total_downstream": total_downstream,
            "species_breakdown": dict(self.species_counts),
            "location": self.io_config.location,
            "date": self.io_config.date_str,
            "output_csv": self.csv_path
        }
    
    def write_summary_report(self) -> bool:
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
            stats = self.get_summary_stats()
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Fish Counting Summary Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'=' * 50}\n\n")
                
                f.write(f"Location: {stats['location']}\n")
                f.write(f"Date: {stats['date']}\n")
                f.write(f"Processing Time: {stats['processing_time_seconds']:.1f} seconds\n\n")
                
                f.write(f"Total Fish Counted: {stats['total_fish_counted']}\n")
                f.write(f"Upstream: {stats['total_upstream']}\n")
                f.write(f"Downstream: {stats['total_downstream']}\n\n")
                
                f.write("Species Breakdown:\n")
                f.write("-" * 30 + "\n")
                
                for species, counts in sorted(stats['species_breakdown'].items()):
                    total_species = counts['Upstream'] + counts['Downstream']
                    f.write(f"{species:20s} Total: {total_species:3d} (Up: {counts['Upstream']:3d}, Down: {counts['Downstream']:3d})\n")
                
                f.write(f"\nDetailed results: {os.path.basename(self.csv_path)}\n")
            
            print(f"✔ Summary report written: {summary_path}")
            return True
            
        except Exception as e:
            print(f"✖ Error writing summary report: {e}")
            return False
    
    def format_video_timestamp(self, frame_number: int, fps: float) -> str:
        """
        Format frame number as HH:MM:SS.fff timestamp.
        
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
    
    def close_csv(self):
        """Close the CSV file and clean up resources."""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print(f"✔ CSV file closed: {self.csv_path}")
    
    def print_final_summary(self):
        """Print final summary to console."""
        stats = self.get_summary_stats()
        
        print(f"\n--- Processing Complete ---")
        print(f"Total time: {stats['processing_time_seconds']:.1f}s")
        print(f"\n--- Results for {stats['location']} on {stats['date']} ---")
        
        for species, counts in sorted(stats['species_breakdown'].items()):
            print(f"  {species:30s} Up={counts['Upstream']} Down={counts['Downstream']}")
        
        print(f"  {'TOTAL':30s} Up={stats['total_upstream']} Down={stats['total_downstream']}")
        print(f"Counts saved to {self.csv_path}")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.open_csv():
            raise RuntimeError("Failed to open CSV file")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.close_csv()
        self.write_summary_report()