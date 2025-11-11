import os
import sys
import time
import cv2
import traceback
from datetime import datetime
from typing import Dict, Any

# Add src to path for importing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.config import PipelineConfig, ConfigLoader
from src.detection import FishDetector, AdiposeDetector
from src.tracking import TrackingManager
from src.classification import SpeciesClassifier
from src.quality import ManualReviewCollector
from src.io import VideoProcessor, OutputHandler, LocalOutputHandler, CloudOutputHandler

# Open CV uses BGR, not RGB
COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_LIGHT_BLUE = (255, 165, 0)


class FishCountingPipeline:
    """
    Main pipeline orchestrator that coordinates all components.
    """
    
    def __init__(self, config: PipelineConfig, is_cloud: bool):
        print("Initializing Fish Counting Pipeline...")

        self.config = config
        self.config.validate()
        
        self.is_cloud = is_cloud

        # Parse date
        input_date = datetime.strptime(self.config.io.date_str, "%Y-%m-%d").date()
        
        # Initialize detectors
        self.detector = FishDetector(self.config.model, self.config.botsort)
        self.adipose_detector = AdiposeDetector(self.config.model)
        
        # Initialize tracking
        self.tracking_manager = TrackingManager(self.config.counting)
        
        # Initialize classification
        self.species_classifier = SpeciesClassifier(
            self.config.classification, 
            self.config.io.location, 
            input_date
        )
        
        # Manual Review collector will be initialized after video is opened
        # (needs video properties like FPS and dimensions)
        self.manual_review_collector = None
        
        # Initialize I/O
        self.video_processor = VideoProcessor(self.config.io, self.config.video)
        self.output_handler = CloudOutputHandler(self.config.io) if is_cloud else LocalOutputHandler(self.config.io)
        
        # Processing statistics
        self.start_time = None
        self.frames_processed = 0
    
    def _draw_frame_annotations(self, frame, center_line, species_counts):
        """Draw center line and live counts on frame."""
        height, width = frame.shape[:2]
        
        # Draw center line
        cv2.line(frame, (center_line, 0), (center_line, height), COLOR_GREEN, 3)
        
        # Draw live counts
        y_offset = 60
        cv2.putText(frame, "Live Counts:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)
        
        for species, counts in sorted(species_counts.items()):
            y_offset += 20
            count_text = f"{species}: Up {counts['Upstream']} Down {counts['Downstream']}"
            cv2.putText(frame, count_text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)
    
    def _draw_detection_annotations(self, frame, bbox, species, confidence, 
                                  length, direction, center_line, crossing_count):
        """Draw detection annotations on frame."""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        
        # Choose color based on position relative to center line and upstream direction
        # YELLOW = approaching/before count, LIGHT_BLUE = already counted/after crossing
        if self.config.counting.upstream_direction == "right_to_left":
            color = COLOR_LIGHT_BLUE if center_x < center_line else COLOR_YELLOW
        else:
            color = COLOR_LIGHT_BLUE if center_x > center_line else COLOR_YELLOW
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        crossing_info = f" (x{crossing_count})" if crossing_count > 0 else ""
        direction_info = f" ({direction})" if direction else ""
        label = f"{species} {length:.1f}in {confidence:.2f}{crossing_info}{direction_info}"
        
        # Draw label
        cv2.putText(frame, label, (x1, max(15, y1 - 10)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _draw_trails(self, frame, track_id, center_line):
        """Draw fish trail on frame."""
        trail = self.tracking_manager.get_trail_points(track_id)
        if len(trail) < 2:
            return
        
        # Choose color based on position relative to center line and upstream direction
        # YELLOW = approaching/before count, LIGHT_BLUE = already counted/after crossing
        current_x = trail[-1][0] if trail else 0
        if self.config.counting.upstream_direction == "right_to_left":
            color = COLOR_LIGHT_BLUE if current_x < center_line else COLOR_YELLOW
        else:
            color = COLOR_LIGHT_BLUE if current_x > center_line else COLOR_YELLOW
        
        # Draw trail lines
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i-1], trail[i], color, 2)
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete fish counting pipeline.
        
        Returns:
            Dictionary containing processing results and statistics
        """
        self.start_time = time.time()
        
        try:
            # Load models
            print("Loading models...")
            self.detector.load_model()
            self.adipose_detector.load_model()
            
            # Use context managers for proper resource cleanup
            with self.video_processor as video_proc, self.output_handler as output:
                return self._process_video(video_proc, output)
                
        except Exception as e:
            print(f"✖ Pipeline failed: {e}")
            traceback.print_exc()
            raise
        finally:
            self._cleanup()
    
    def _process_video(self, video_proc: VideoProcessor, output: OutputHandler) -> Dict[str, Any]:
        """Process the video frame by frame."""
        # Initialize tracking with video dimensions
        self.tracking_manager.initialize_frame_info(video_proc.width)
        
        # Initialize Manual Review collector with video properties
        self.manual_review_collector = ManualReviewCollector(
            config=self.config.hitl,
            location=self.config.io.location,
            ladder=self.config.io.ladder,
            date_str=self.config.io.date_str,
            time_str=self.config.io.time_str,
            video_fps=video_proc.fps,
            frame_width=video_proc.width,
            frame_height=video_proc.height,
            upstream_direction=self.config.counting.upstream_direction
        )
        
        # Set center line for manual review collector
        center_line = self.tracking_manager.center_line
        self.manual_review_collector.set_center_line_position(center_line)
        
        print(f"Starting video processing...")
        print(f"Center line at pixel {center_line}")
        
        # Process frames
        for success, frame, frame_number, timestamp_sec in video_proc.read_frames():
            if not success:
                break
            
            self.frames_processed = frame_number
            
            # Run detection and tracking
            results = self.detector.detect_and_track(frame)
            
            # Draw frame annotations
            self._draw_frame_annotations(frame, center_line, output.get_species_counts())
            
            # Extract detections
            boxes, track_ids, confidences, class_ids = self.detector.extract_detections(results)
            
            # Collect detections for manual review (track_id, bbox pairs)
            frame_detections = []
            
            if boxes is not None and track_ids is not None:
                # Clean up inactive tracks
                active_track_ids = set(track_ids)
                self.tracking_manager.cleanup_inactive_tracks(active_track_ids)
                
                # Collect detections for occlusion detection
                for bbox, track_id in zip(boxes, track_ids):
                    frame_detections.append((int(track_id), tuple(map(int, bbox))))
                
                # Process each detection
                for bbox, track_id, confidence, class_id in zip(boxes, track_ids, confidences, class_ids):
                    self._process_detection(
                        frame, bbox, track_id, confidence, class_id,
                        frame_number, timestamp_sec, video_proc.fps,
                        output, center_line, video_proc
                    )
            
            # Process frame for occlusion detection and clip recording
            self.manual_review_collector.process_frame(
                frame,
                frame_number,
                timestamp_sec, 
                os.path.basename(self.config.io.video_path),
                frame_detections
            )
            
            # Write frame to the annotated video output (only happens if running locally)
            video_proc.write_frame(frame)
        
        # Finalize manual review collector to save any remaining peak occlusions
        self.manual_review_collector.finalize_processing()

        # Upload QA clips to S3 if running in cloud mode
        qa_clips_s3_url = None
        
        if self.is_cloud:
            s3_bucket = os.getenv('QA_CLIPS_S3_BUCKET')
            if not s3_bucket:
                raise ValueError("QA_CLIPS_S3_BUCKET environment variable must be set in cloud mode.")
            s3_prefix = f"{self.config.io.location}/{self.config.io.ladder}/{self.config.io.date_str}/{self.config.io.time_str}"
            clips_dir = self.manual_review_collector.clips_dir
            if isinstance(output, CloudOutputHandler):
                qa_clips_s3_url = output.upload_qa_clips_to_s3(clips_dir, s3_bucket, s3_prefix)
        
        # Write final fish counts
        output.write_final_counts(qa_clips_s3_url=qa_clips_s3_url)

        # Return results
        elapsed_total = time.time() - self.start_time
        return {
            "frames_processed": self.frames_processed,
            "processing_time": elapsed_total
        }
    
    def _process_detection(self, frame, bbox, track_id, confidence, class_id,
                          frame_number, timestamp_sec, fps, output: OutputHandler, 
                          center_line, video_proc: VideoProcessor):
        """Process a single detection through the full pipeline."""
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Get raw species from model
        raw_species = self.detector.get_class_name(class_id)
        
        # Classify species with business rules
        classified_species, classification_reason = self.species_classifier.classify_detection(raw_species, confidence)
        
        # Detect adipose status for this frame (if applicable)
        current_adipose_status = None
        if self.adipose_detector.is_loaded and "_U" in classified_species:
            current_adipose_status, adipose_conf = self.adipose_detector.infer_adipose_status(
                frame, bbox
            )
        
        # Update tracking state (this adds adipose vote to the queue)
        fish_state, direction = self.tracking_manager.process_detection(
            track_id, bbox, classified_species, confidence, 
            adipose_status=current_adipose_status
        )
        
        # Add current confidence to history for QA tracking
        fish_state.add_confidence(confidence)
        
        # Get stable species from temporal voting
        stable_species = fish_state.get_stable_species() or "Unknown"
        
        # Apply adipose refinement using temporal voting (if we have enough votes)
        stable_adipose = fish_state.get_stable_adipose()
        if stable_adipose and stable_adipose in {"Present", "Absent"} and "_U" in stable_species:
            stable_species = self.species_classifier.apply_adipose_refinement(stable_species, stable_adipose)
        
        # Check for counting
        can_count = (direction is not None and 
                    #TODO (keep?): stable_species != "Unknown" and
                    self.tracking_manager.can_count_crossing(fish_state, direction, frame_number))
        
        if can_count:
            # Record the count
            self.tracking_manager.record_crossing(fish_state, direction, frame_number)
            
            # Use the maximum confidence from history to catch fish that may have had low confidence
            # consistently during tracking, not just at the crossing moment
            max_confidence = fish_state.get_max_confidence()
            if max_confidence is None:
                max_confidence = confidence  # Fallback to current confidence

            video_timestamp = output.format_video_timestamp(frame_number, fps)
            frame_width, frame_height = video_proc.width, video_proc.height
            x_percent = (center_x / frame_width) * 100 if frame_width > 0 else 0
            y_percent = (center_y / frame_height) * 100 if frame_height > 0 else 0
            
            # Write count record
            output.record_count(
                video_timestamp=video_timestamp,
                frame_number=frame_number,
                track_id=track_id,
                species=stable_species,
                confidence=max_confidence,
                direction=direction,
                x_percent=x_percent,
                y_percent=y_percent,
                length_inches=fish_state.length_inches
            )
            
            print(f"COUNTED: Track {track_id} ({stable_species}, {fish_state.length_inches:.1f}in) "
                  f"{direction} at {video_timestamp} - Crossing #{fish_state.crossing_count}")

            # Report crossing event. This captures low-confidence, unknown species, and bull trout crossings
            self.manual_review_collector.report_crossing_event(
                track_id=track_id,
                bbox=bbox,
                frame_idx=frame_number,
                timestamp_sec=timestamp_sec,
                video_name=os.path.basename(self.config.io.video_path),
                species=stable_species,
                confidence=max_confidence,
                direction=direction
            )
        
        # Draw annotations
        self._draw_detection_annotations(
            frame, bbox, stable_species, confidence,
            fish_state.length_inches, direction, center_line, fish_state.crossing_count
        )
        self._draw_trails(frame, track_id, center_line)
    
    def _cleanup(self):
        """Clean up resources."""
        if self.detector:
            self.detector.cleanup()
        if self.adipose_detector:
            self.adipose_detector.cleanup()
        print("✔ Pipeline cleanup complete")



def main():
    """Main entry point."""
    print("🐟 Fish Counting Pipeline v2.0 (Refactored)")
    
    try:
        sqs_message = os.getenv('SQS_MESSAGE')
        model_s3_bucket = os.getenv('MODEL_S3_BUCKET')
        model_s3_key = os.getenv('MODEL_S3_KEY')

        is_cloud = sqs_message is not None

        if is_cloud:
            print("Running in cloud mode. Loading configuration from SQS message.")

            if not model_s3_bucket or not model_s3_key:
                raise ValueError("MODEL_S3_BUCKET and MODEL_S3_KEY environment variables must be set in cloud mode.")

            config = ConfigLoader.load_config_from_sqs_message(sqs_message, model_s3_bucket, model_s3_key)
        else:
            print("Running in local mode. Loading configuration from: configs/local.json.")
            config = ConfigLoader.load_config_from_file("configs/alex.json")

        # Create and run pipeline
        pipeline = FishCountingPipeline(config, is_cloud)
        results = pipeline.run()
        
        print("\n🎉 Pipeline completed successfully!")
        print(f"Processed {results['frames_processed']} frames in {results['processing_time']:.1f}s")
        
    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())