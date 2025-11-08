"""
Video processing utilities for reading frames and managing video I/O.

This module handles video input/output operations and provides a clean interface
for frame-by-frame processing that's compatible with both local files and cloud storage.
"""

import cv2
from typing import Optional, Tuple, Generator
import numpy as np

from ..config.settings import VideoConfig, IOConfig


class VideoProcessor:
    """
    Handles video input processing with support for both local and cloud storage.
    
    This class provides a unified interface for video processing that can work
    with local files or cloud-based video streams.
    """
    
    def __init__(self, io_config: IOConfig, video_config: VideoConfig):
        self.io_config = io_config
        self.video_config = video_config
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        
        # Video properties
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.fps: Optional[float] = None
        self.total_frames: Optional[int] = None
        self.current_frame: int = 0
    
    def open_video(self) -> bool:
        """
        Open the video file for reading.
        
        Returns:
            True if video opened successfully, False otherwise
        """
        if not self.io_config.video_path:
            raise ValueError("Video path not configured")
        
        self.cap = cv2.VideoCapture(self.io_config.video_path)
        if not self.cap.isOpened():
            print(f"✖ Could not open video: {self.io_config.video_path}")
            return False
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {self.width}x{self.height} @ {self.fps:.1f}fps, {self.total_frames} frames")
        return True
    
    def setup_output_video(self) -> bool:
        """
        Set up output video writer if video saving is enabled.
        
        Returns:
            True if setup successful or not needed, False on error
        """
        if not self.video_config.save_output_video:
            return True
        
        if self.width is None or self.height is None or self.fps is None:
            print("✖ Cannot setup output video: input video not opened")
            return False
        
        fourcc = cv2.VideoWriter_fourcc(*self.video_config.output_video_codec)
        
        self.video_writer = cv2.VideoWriter(self.io_config.video_output_path, fourcc, self.fps, (self.width, self.height))
        
        if not self.video_writer.isOpened():
            print(f"✖ Failed to create output video: {self.io_config.video_output_path}")
            return False
        
        print(f"✔ Will save annotated video to: {self.io_config.video_output_path}")
        return True
    
    def read_frames(self) -> Generator[Tuple[bool, Optional[np.ndarray], int, float], None, None]:
        """
        Generator that yields video frames with metadata.
        
        Yields:
            Tuple of (success, frame, frame_number, timestamp_seconds)
        """
        if self.cap is None:
            raise RuntimeError("Video not opened. Call open_video() first.")
        
        frame_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Calculate timestamp
            timestamp_sec = (frame_count / self.fps) if self.fps > 0 else 0
            
            yield True, frame, frame_count, timestamp_sec
        
        yield False, None, frame_count, 0.0
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the output video.
        
        Args:
            frame: Frame to write
            
        Returns:
            True if write successful or writer not configured, False on error
        """
        if self.video_writer is None:
            return True  # No writer configured, which is fine
        
        try:
            self.video_writer.write(frame)
            return True
        except Exception as e:
            print(f"✖ Error writing frame: {e}")
            return False
    
    def display_frame(self, frame: np.ndarray, window_name: str = "Fish Counter") -> bool:
        """
        Display a frame if display is enabled.
        
        Args:
            frame: Frame to display
            window_name: Window name for display
            
        Returns:
            True if should continue, False if user requested quit
        """
        if not self.video_config.enable_display:
            return True
        
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord('q')
    
    def cleanup(self):
        """Clean up video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        if self.video_config.enable_display:
            cv2.destroyAllWindows()
        
        print("✔ Video resources cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        if not self.open_video():
            raise RuntimeError("Failed to open video")
        if not self.setup_output_video():
            raise RuntimeError("Failed to setup output video")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()