"""
Preprocessing job to identify and extract fish-containing segments from videos.
Optimized for cost savings by filtering out empty footage before expensive GPU processing.
"""

import os
import sys
import logging
import json
import tempfile
import cv2
import boto3
import subprocess
import shutil
from typing import List, Dict, Tuple
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

@dataclass
class VideoSegment:
    """Represents a time segment in the video."""
    start_sec: float
    end_sec: float
    detection_count: int = 0
    
    @property
    def duration_sec(self) -> float:
        return self.end_sec - self.start_sec


class VideoPreprocessor:
    """
    Preprocesses videos to extract only segments containing fish.
    Uses lightweight detection to minimize costs.
    """
    
    def __init__(self,
                 sample_interval: int = 5,
                 buffer_before_sec: float = 3,
                 buffer_after_sec: float = 3,
                 merge_gap_sec: float = 60):
        """
        Args:
            sample_interval: Check every Nth frame
            buffer_before_sec: Seconds to include before first fish detection
            buffer_after_sec: Seconds to include after last fish detection  
            merge_gap_sec: Merge segments if gap between them is less than this
        """
        self.sample_interval = sample_interval
        self.buffer_before_sec = buffer_before_sec
        self.buffer_after_sec = buffer_after_sec
        self.merge_gap_sec = merge_gap_sec
        
        self.s3_client = boto3.client('s3')
 
        # Initialize background subtractor
        # history: number of frames to learn background model
        # varThreshold: threshold on squared Mahalanobis distance for pixel-model match
        # detectShadows: disabled to improve performance
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=16,
            detectShadows=False
        )
        
        # Area filter: self.min_area - self.max_area pixels²"
        self.min_area = 500
        self.max_area = 50000

    def download_video_from_s3(self, bucket: str, key: str, local_path: str) -> bool:
        """Download video from S3 to local storage."""
        try:
            logger.info("Downloading s3://%s/%s to %s...", bucket, key, local_path)
            self.s3_client.download_file(bucket, key, local_path)
            
            file_size_gb = os.path.getsize(local_path) / (1024**3)
            logger.info("Downloaded %.2f GB", file_size_gb)
            return True
            
        except Exception as e:
            logger.exception("Failed to download video: %s", e)
            return False
    
    def detect_fish_segments(self, video_path: str) -> List[VideoSegment]:
        """
        Scan video and identify time segments containing fish.
        
        Returns:
            List of VideoSegment objects with fish detections
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info("Scanning video for fish presence. Sample interval: every %d frames", self.sample_interval)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0

        logger.info("Video: %.1f hours (%d frames @ %.1f fps)", duration_sec/3600, total_frames, fps)

        # Learning period - let background subtractor see sediment
        logger.info("Learning background model (processing first 500 frames)...")
        for i in range(min(500, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (640, 480))
            self.bg_subtractor.apply(resized, learningRate=0.01)

        # Reset to start for actual detection
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        logger.info("Background model learned. Starting detection...")

        # Track detections with list of (timestamp_sec, detection_count)
        detection_timestamps = []
        
        frame_idx = 0
        frames_checked = 0
        total_detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process sampled frames
            if frame_idx % self.sample_interval == 0:
                timestamp_sec = frame_idx / fps
                
                # Run background subtraction (resize for speed)
                resized = cv2.resize(frame, (640, 480))

                # Apply background subtraction
                fg_mask = self.bg_subtractor.apply(resized)

                # Morphological operations to reduce noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

                # Find contours
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Filter by size
                num_detections = 0
                detected_areas = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if self.min_area < area < self.max_area:
                        num_detections += 1
                        detected_areas.append(area)

                if num_detections > 0:
                    detection_timestamps.append((timestamp_sec, num_detections))
                    total_detections += num_detections
                
                frames_checked += 1
                
                # Progress update
                if frames_checked % 100 == 0:
                    progress = (frame_idx / total_frames) * 100
                    logger.info("Progress: %.1f%% | Detections: %d", progress, total_detections)
            
            frame_idx += 1
        
        cap.release()

        logger.info("Scan complete: %d moving objects detected across %d sampled frames", total_detections, len(detection_timestamps))
        logger.info("Frames checked: %d / %d (%.1f%%)", frames_checked, total_frames, frames_checked/total_frames*100)

        # Convert detection timestamps into segments with buffers
        segments = self._create_segments_from_detections(detection_timestamps, duration_sec)
        
        return segments
    
    def _create_segments_from_detections(self, 
                                        detection_timestamps: List[Tuple[float, int]], 
                                        video_duration_sec: float) -> List[VideoSegment]:
        """
        Convert detection timestamps into buffered and merged segments.
        
        Args:
            detection_timestamps: List of (timestamp_sec, detection_count)
            video_duration_sec: Total video duration
            
        Returns:
            List of merged VideoSegment objects
        """
        if not detection_timestamps:
            logger.info("No fish detected in video")
            return []
        
        # Sort by timestamp
        detection_timestamps.sort(key=lambda x: x[0])
        
        # Create initial segments with buffers
        raw_segments = []
        current_segment = None
        
        for timestamp_sec, det_count in detection_timestamps:
            if current_segment is None:
                # Start new segment
                current_segment = VideoSegment(
                    start_sec=max(0, timestamp_sec - self.buffer_before_sec),
                    end_sec=timestamp_sec + self.buffer_after_sec,
                    detection_count=det_count
                )
            else:
                # Check if we should extend current segment or start new one
                if timestamp_sec <= current_segment.end_sec:
                    # Extend current segment
                    current_segment.end_sec = max(current_segment.end_sec, 
                                                 timestamp_sec + self.buffer_after_sec)
                    current_segment.detection_count += det_count
                else:
                    # Close current segment and start new one
                    raw_segments.append(current_segment)
                    current_segment = VideoSegment(
                        start_sec=max(0, timestamp_sec - self.buffer_before_sec),
                        end_sec=timestamp_sec + self.buffer_after_sec,
                        detection_count=det_count
                    )
        
        # Don't forget the last segment
        if current_segment is not None:
            raw_segments.append(current_segment)
        
        # Merge segments that are close together
        merged_segments = self._merge_close_segments(raw_segments)
        
        # Only the last segment can exceed video duration due to buffer_after_sec
        if merged_segments:
            merged_segments[-1].end_sec = min(merged_segments[-1].end_sec, video_duration_sec)

        logger.info("Segment Summary: raw=%d, merged=%d", len(raw_segments), len(merged_segments))
        
        total_duration = sum(seg.duration_sec for seg in merged_segments)
        reduction_pct = 100 * (1 - total_duration / video_duration_sec)
        
        logger.info("Total processing duration: %.2f hours (%.1f%% reduction)", 
            total_duration/3600, reduction_pct)
        
        return merged_segments
    
    def _merge_close_segments(self, segments: List[VideoSegment]) -> List[VideoSegment]:
        """
        Merge segments that are within merge_gap_sec of each other.
        
        This prevents creating many small clips when fish are consistently present.
        """
        if not segments:
            return []
        
        merged = []
        current = segments[0]
        
        for next_seg in segments[1:]:
            gap = next_seg.start_sec - current.end_sec
            
            if gap <= self.merge_gap_sec:
                # Merge: extend current segment to include next
                current.end_sec = next_seg.end_sec
                current.detection_count += next_seg.detection_count
            else:
                # Gap too large, finalize current and start new
                merged.append(current)
                current = next_seg
        
        # Add the last segment
        merged.append(current)
        
        return merged
    
    def extract_video_clips(self, 
                           source_video: str, 
                           segments: List[VideoSegment],
                           output_dir: str,
                           video_name: str) -> List[Dict]:
        """
        Extract video clips for each segment using ffmpeg.
        
        Returns:
            List of clip metadata dictionaries
        """
        os.makedirs(output_dir, exist_ok=True)
        
        clip_metadata = []
        
        for i, seg in enumerate(segments):
            clip_filename = f"{video_name}_clip_{i:03d}.mp4"
            clip_path = os.path.join(output_dir, clip_filename)

            logger.info("Extracting clip %d/%d: %s (%.1fs - %.1fs, %.1fs duration)", 
                       i+1, len(segments), clip_filename, seg.start_sec, seg.end_sec, seg.duration_sec)

            # Use ffmpeg for fast extraction
            success = self._extract_clip_ffmpeg(source_video, seg.start_sec, seg.duration_sec, clip_path)
            
            if success:
                clip_size_mb = os.path.getsize(clip_path) / (1024**2)
                logger.info("Created: %.1f MB", clip_size_mb)
                
                clip_metadata.append({
                    'clip_filename': clip_filename,
                    'clip_path': clip_path,
                    'start_sec': seg.start_sec,
                    'end_sec': seg.end_sec,
                    'duration_sec': seg.duration_sec,
                    'detection_count': seg.detection_count,
                    'clip_index': i
                })
            else:
                logger.error("Failed to extract clip %d", i)
        
        return clip_metadata
    
    def _extract_clip_ffmpeg(self, source: str, start_sec: float, duration_sec: float, output: str) -> bool:
        """Extract video clip using ffmpeg (fast, no re-encoding)."""
        try:
            cmd = [
                'ffmpeg',
                '-i', source,
                '-ss', str(start_sec),
                '-t', str(duration_sec),
                # Re-encode video for precision
                '-c:v', 'libx264',
                # Fast encoding
                '-preset', 'veryfast',
                # Quality (lower = better, 0 would be lossless but can cause issues)
                '-crf', '18',
                '-an',
                '-avoid_negative_ts', 'make_zero',
                '-y',
                output
            ]
            
            result = subprocess.run(cmd, 
                                   capture_output=True, 
                                   text=True,
                                   timeout=300)
            
            return result.returncode == 0
            
        except Exception as e:
            logger.exception("ffmpeg error: %s", e)
            return False
    
    def upload_clips_to_s3(self, 
                          clips: List[Dict], 
                          bucket: str, 
                          s3_prefix: str) -> List[Dict]:
        """
        Upload extracted clips to S3.
        
        Returns:
            Updated clip metadata with S3 keys
        """
        logger.info("Uploading %d clips to s3://%s/%s/", len(clips), bucket, s3_prefix)
        
        for clip in clips:
            local_path = clip['clip_path']
            s3_key = f"{s3_prefix}/{clip['clip_filename']}"
            
            logger.info("Uploading %s...", clip['clip_filename'])
            
            try:
                self.s3_client.upload_file(local_path, bucket, s3_key)
                clip['s3_key'] = s3_key
                clip['s3_bucket'] = bucket
                logger.info("Successfully uploaded to s3://%s/%s", bucket, s3_key)
                
                # Clean up local file to save space
                os.remove(local_path)
                
            except Exception as e:
                logger.exception("Upload failed: %s", e)

        return clips
    
    def send_clips_to_processing_queue(self, 
                                  clips: List[Dict],
                                  queue_url: str,
                                  location: str,
                                  ladder: str,
                                  date_str: str,
                                  time_str: str):
        """Send SQS messages for each clip to trigger full processing."""
        sqs_client = boto3.client('sqs')

        logger.info("Sending %d messages to processing queue...", len(clips))

        for clip in clips:
            message = {
                'video_s3_bucket': clip['s3_bucket'],
                'video_s3_key': clip['s3_key'],
                'location': location,
                'ladder': ladder,
                'date_str': date_str,
                'time_str': time_str
            }
            
            try:
                response = sqs_client.send_message(
                    QueueUrl=queue_url,
                    MessageBody=json.dumps(message)
                )
                logger.info("Sent clip %d: %s...", clip['clip_index'], response['MessageId'][:8])
                
            except Exception as e:
                logger.exception("Failed to send message for clip %d: %s", clip['clip_index'], e)


def parse_filename(filename: str) -> Tuple[str, str, str, str]:
    """
    Parse location, ladder, date, and time from filename.
    Expected format: {location}-{ladder}-{YYYYMMDD}-{HHMM}.mp4
    Example: wells-east-20251025-0800.mp4
    
    Returns:
        Tuple of (location, ladder, date_str, time_str)
    """
    basename = os.path.basename(filename).replace('.mp4', '')
    parts = basename.split('-')
    
    if len(parts) >= 4:
        location = parts[0]
        ladder = parts[1]
        date_raw = parts[2]
        time_str = parts[3]
        
        # Convert date format: 20251025 -> 2025-10-25
        date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
        
        return location, ladder, date_str, time_str
    else:
        raise ValueError(
            f"Cannot parse filename: {filename}. "
            f"Expected format: location-ladder-YYYYMMDD-HHMM.mp4"
        )


def main():
    """Main preprocessing job entry point - reads from SQS queue."""
    logger.info("Begin Fish Video Preprocessing Job")

    sqs_message = os.getenv('SQS_MESSAGE')
    processing_queue_url = os.getenv('PROCESSING_QUEUE_URL')

    if not sqs_message or not processing_queue_url:
        logger.error("Missing required environment variables. Exiting.")
        return 1

    body = json.loads(sqs_message)
    
    video_s3_bucket = body['video_s3_bucket']
    video_s3_key = body['video_s3_key']

    logger.info("Processing: s3://%s/%s", video_s3_bucket, video_s3_key)

    # Parse video metadata from filename
    try:
        video_filename = os.path.basename(video_s3_key)
        location, ladder, date_str, time_str = parse_filename(video_filename)
        logger.info("Location: %s, Ladder: %s, Date: %s, Time: %s", location, ladder, date_str, time_str)
    except ValueError as e:
        logger.error("Invalid filename format: %s", e)
        return 1
    
    try:
        # Initialize preprocessor
        preprocessor = VideoPreprocessor()
        
        # Download video
        video_local_path = os.path.join(tempfile.gettempdir(), 'input_video.mp4')
        if not preprocessor.download_video_from_s3(video_s3_bucket, video_s3_key, video_local_path):
            return 1
        
        # Detect fish segments
        segments = preprocessor.detect_fish_segments(video_local_path)
        
        if not segments:
            logger.info("No fish detected in video - no processing needed")
            return 0
        
        # Extract clips
        output_dir = os.path.join(tempfile.gettempdir(), 'clips')
        video_name = video_filename.replace('.mp4', '')
        clips = preprocessor.extract_video_clips(video_local_path, segments, output_dir, video_name)
        
        if not clips:
            logger.error("No clips extracted. Exiting.")
            return 1
        
        # Upload clips to S3 (same bucket, different prefix)
        s3_prefix = f"trimmed-clips/{video_name}"
        clips = preprocessor.upload_clips_to_s3(clips, video_s3_bucket, s3_prefix)
        
        # Send to processing queue
        preprocessor.send_clips_to_processing_queue(
            clips=clips,
            queue_url=processing_queue_url,
            location=location,
            ladder=ladder,
            date_str=date_str,
            time_str=time_str
        )

        logger.info("Preprocessing complete! %d clips created from %s, ready for GPU processing.", len(clips), video_s3_key)

        return 0
    except Exception as e:
        logger.exception("Preprocessing failed: %s", e)
        return 1
    finally:
        # Clean up local files
        if os.path.exists(video_local_path):
            os.remove(video_local_path)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


# # This version of main() is for local testing only
# def main():
#     """Main preprocessing job entry point - uses local files."""
#     logger.info("Begin Fish Video Preprocessing Job")
    
#     video_filename = "wells-east-20251025-0800.mp4"
#     location, ladder, date_str, time_str = parse_filename(video_filename)
#     logger.info("Location: %s, Ladder: %s, Date: %s, Time: %s", location, ladder, date_str, time_str)

#     try:
#         preprocessor = VideoPreprocessor()
        
#         video_local_path = 'path-to-your-video.mp4'
        
#         segments = preprocessor.detect_fish_segments(video_local_path)
        
#         if not segments:
#             logger.info("No fish detected in video - no processing needed")
#             return 0
        
#         output_dir = 'path-to-your-output-directory'
#         video_name = video_filename.replace('.mp4', '')
#         clips = preprocessor.extract_video_clips(video_local_path, segments, output_dir, video_name)
        
#         if not clips:
#             logger.error("No clips extracted. Exiting.")
#             return 1
        
#         logger.info("Preprocessing complete! %d clips created, ready for GPU processing.", len(clips))
        
#         return 0
        
#     except Exception as e:
#         logger.exception("Preprocessing failed: %s", e)
#         return 1


if __name__ == "__main__":
    sys.exit(main())