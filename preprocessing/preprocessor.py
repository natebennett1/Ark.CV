"""
Preprocessing job to identify and extract fish-containing segments from videos.
Optimized for cost savings by filtering out empty footage before expensive GPU processing.
"""

import os
import sys
import json
import cv2
import boto3
import subprocess
from typing import List, Dict, Tuple
from dataclasses import dataclass

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
        
        self.min_area = 500
        self.max_area = 50000

        print("Initialized background subtractor for motion detection")
        print(f"  Area filter: {self.min_area} - {self.max_area} pixels²")

    def download_video_from_s3(self, bucket: str, key: str, local_path: str) -> bool:
        """Download video from S3 to local storage."""
        try:
            print(f"Downloading s3://{bucket}/{key} to {local_path}...")
            self.s3_client.download_file(bucket, key, local_path)
            
            file_size_gb = os.path.getsize(local_path) / (1024**3)
            print(f"✓ Downloaded {file_size_gb:.2f} GB")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download video: {e}")
            return False
    
    def detect_fish_segments(self, video_path: str) -> List[VideoSegment]:
        """
        Scan video and identify time segments containing fish.
        
        Returns:
            List of VideoSegment objects with fish detections
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print(f"\nScanning video for fish presence...")
        print(f"Sample interval: every {self.sample_interval} frames")
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0
        
        print(f"Video: {duration_sec/3600:.1f} hours ({total_frames:,} frames @ {fps:.1f} fps)")

        # Learning period - let background subtractor see sediment
        print("Learning background model (processing first 500 frames)...")
        for i in range(min(500, total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (640, 480))
            self.bg_subtractor.apply(resized, learningRate=0.01)  # Slow learning

        # Reset to start for actual detection
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print("Background model learned. Starting detection...")

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
                    print(f"Progress: {progress:.1f}% | Detections: {total_detections}")
            
            frame_idx += 1
        
        cap.release()
        
        print(f"✓ Scan complete: {total_detections} moving objects detected across {len(detection_timestamps)} sampled frames")
        print(f"  Frames checked: {frames_checked:,} / {total_frames:,} ({frames_checked/total_frames*100:.1f}%)")
        
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
            print("No fish detected in video")
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

        # Print segment summary
        print(f"\nSegment Summary:")
        print(f"  Raw segments: {len(raw_segments)}")
        print(f"  Merged segments: {len(merged_segments)}")
        
        total_duration = sum(seg.duration_sec for seg in merged_segments)
        reduction_pct = 100 * (1 - total_duration / video_duration_sec)
        
        print(f"  Total processing duration: {total_duration/3600:.2f} hours ({reduction_pct:.1f}% reduction)")
        print(f"\n  Segments:")
        for i, seg in enumerate(merged_segments):
            print(f"    [{i}] {seg.start_sec/60:.1f}m - {seg.end_sec/60:.1f}m "
                  f"(duration: {seg.duration_sec/60:.1f}m, detections: {seg.detection_count})")
        
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
                print(f"  Merging segments (gap: {gap:.0f}s)")
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
            
            print(f"\nExtracting clip {i+1}/{len(segments)}: {clip_filename}")
            print(f"  Time range: {seg.start_sec:.1f}s - {seg.end_sec:.1f}s ({seg.duration_sec:.1f}s)")
            
            # Use ffmpeg for fast, lossless extraction
            success = self._extract_clip_ffmpeg(source_video, seg.start_sec, seg.duration_sec, clip_path)
            
            if success:
                clip_size_mb = os.path.getsize(clip_path) / (1024**2)
                print(f"  ✓ Created: {clip_size_mb:.1f} MB")
                
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
                print(f"  ✗ Failed to extract clip {i}")
        
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
            print(f"ffmpeg error: {e}")
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
        print(f"\nUploading {len(clips)} clips to s3://{bucket}/{s3_prefix}/")
        
        for clip in clips:
            local_path = clip['clip_path']
            s3_key = f"{s3_prefix}/{clip['clip_filename']}"
            
            print(f"  Uploading {clip['clip_filename']}...")
            
            try:
                self.s3_client.upload_file(local_path, bucket, s3_key)
                clip['s3_key'] = s3_key
                clip['s3_bucket'] = bucket
                print(f"    ✓ s3://{bucket}/{s3_key}")
                
                # Clean up local file to save space
                os.remove(local_path)
                
            except Exception as e:
                print(f"    ✗ Upload failed: {e}")
        
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
        
        print(f"\nSending {len(clips)} messages to processing queue...")
        
        for clip in clips:
            message = {
                'video_s3_bucket': clip['s3_bucket'],
                'video_s3_key': clip['s3_key'],
                'location': location,
                'date_str': date_str
            }
            
            try:
                response = sqs_client.send_message(
                    QueueUrl=queue_url,
                    MessageBody=json.dumps(message)
                )
                print(f"  ✓ Sent clip {clip['clip_index']}: {response['MessageId'][:8]}...")
                
            except Exception as e:
                print(f"  ✗ Failed to send message for clip {clip['clip_index']}: {e}")


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
        location = parts[0].title()  # wells -> Wells
        ladder = parts[1].title()    # east -> East
        date_raw = parts[2]          # 20251025
        time_str = parts[3]          # 0800
        
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
    print("Fish Video Preprocessing Job")
    print("=" * 50)
    
    # Read message from SQS queue (set by ECS task environment)
    # TODO: Uncomment
    # preprocessing_queue_url = os.getenv('PREPROCESSING_QUEUE_URL')
    # processing_queue_url = os.getenv('PROCESSING_QUEUE_URL')
    
    # if not preprocessing_queue_url or not processing_queue_url:
    #     print("Missing required environment variables. Exiting.")
    #     return 1
    # TODO: End Uncomment
    
    # Receive message from preprocessing queue
    # TODO: Uncomment
    # sqs_client = boto3.client('sqs')
    
    # print(f"Polling preprocessing queue: {preprocessing_queue_url}")
    # response = sqs_client.receive_message(
    #     QueueUrl=preprocessing_queue_url,
    #     MaxNumberOfMessages=1,
    #     WaitTimeSeconds=10  # Long polling
    # )
    
    # if 'Messages' not in response:
    #     print("No messages in queue")
    #     return 0
    
    # message = response['Messages'][0]
    # receipt_handle = message['ReceiptHandle']
    # body = json.loads(message['Body'])
    
    # video_s3_bucket = body['video_s3_bucket']
    # video_s3_key = body['video_s3_key']
    
    # print(f"Processing: s3://{video_s3_bucket}/{video_s3_key}")
    # TODO: End Uncomment
    
    # Parse video metadata from filename
    try:
        # TODO: Switch back
        # video_filename = os.path.basename(video_s3_key)
        video_filename = "wells-east-20251025-0800.mp4"
        location, ladder, date_str, time_str = parse_filename(video_filename)
        print(f"Location: {location}")
        print(f"Ladder: {ladder}")
        print(f"Date: {date_str}")
        print(f"Time: {time_str}")
    except ValueError as e:
        print(f"Error: {e}")
        # Delete message from queue (invalid format)
        # TODO: Uncomment
        # sqs_client.delete_message(
        #     QueueUrl=preprocessing_queue_url,
        #     ReceiptHandle=receipt_handle
        # )
        # TODO: End Uncomment
        return 1
    
    try:
        # Initialize preprocessor
        preprocessor = VideoPreprocessor()
        
        # Download video
        # TODO: Uncomment
        # video_local_path = '/tmp/input_video.mp4'
        # if not preprocessor.download_video_from_s3(video_s3_bucket, video_s3_key, video_local_path):
        #     return 1
        video_local_path = 'C:\\Users\\alexqian\\OneDrive - Microsoft\\Documents\\! MY DOCUMENTS !\\ArkInputs\\Sparse-Test-Video.mp4'
        # TODO: End Uncomment
        
        # Detect fish segments
        segments = preprocessor.detect_fish_segments(video_local_path)
        
        if not segments:
            print("\nNo fish detected in video - no processing needed")
            # Delete message and exit successfully
            # TODO: Uncomment
            # sqs_client.delete_message(
            #     QueueUrl=preprocessing_queue_url,
            #     ReceiptHandle=receipt_handle
            # )
            # TODO: End Uncomment
            return 0
        
        # Extract clips
        # TODO: Switch back
        # output_dir = '/tmp/clips'
        output_dir = 'C:\\Users\\alexqian\\OneDrive - Microsoft\\Documents\\! MY DOCUMENTS !\\ArkInputs\\tmp'
        video_name = video_filename.replace('.mp4', '')
        clips = preprocessor.extract_video_clips(video_local_path, segments, output_dir, video_name)
        
        if not clips:
            print("No clips extracted. Exiting.")
            return 1
        
        # Upload clips to S3 (same bucket, different prefix)
        # TODO: Uncomment
        # s3_prefix = f"processed_clips/{video_name}"
        # clips = preprocessor.upload_clips_to_s3(clips, video_s3_bucket, s3_prefix)
        
        # Send to processing queue
        # preprocessor.send_clips_to_processing_queue(
        #     clips=clips,
        #     queue_url=processing_queue_url,
        #     location=location,
        #     ladder=ladder,
        #     date_str=date_str,
        #     time_str=time_str
        # )
        # TODO: End Uncomment

        print("\nPreprocessing complete!")
        # TODO: Uncomment
        # print(f"   Original video: {video_s3_key}")
        # TODO: End Uncomment
        print(f"   Clips created: {len(clips)}")
        print(f"   Ready for GPU processing")
        
        # Delete message from queue (success)
        # TODO: Uncomment
        # sqs_client.delete_message(
        #     QueueUrl=preprocessing_queue_url,
        #     ReceiptHandle=receipt_handle
        # )
        # TODO: End Uncomment
        
        return 0
        
    except Exception as e:
        print(f"\nPreprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        # Don't delete message - it will be retried or go to DLQ
        return 1


if __name__ == "__main__":
    sys.exit(main())