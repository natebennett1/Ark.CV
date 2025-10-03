#!/usr/bin/env python3
import os
import cv2
import time
import csv
import numpy as np
import torch
import traceback
from datetime import datetime
from ultralytics import YOLO
from collections import defaultdict

# === PYTORCH 2.6 COMPATIBILITY FIX ===
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

def run_adipose_counter(
    model_path=r"C:\Users\20ben\Downloads\weights621.pt",
    video_path=r"C:\Users\20ben\WEST1_HQ_20240715_100000.mp4",
    location="Wells Dam",
    date_str="2024-07-15",
    confidence_threshold=0.1,
    unknown_threshold=0.5,
    adipose_lock_threshold=0.5,
    center_line_position=0.50,
    trail_max_length=30,
    enable_display=False,
    save_output_video=True,
    pixels_per_inch=26.04
):
    global writer, w, h, fps, fish_states, track_history, species_counts, INACTIVE_FRAMES_THRESHOLD, ADIPOSE_LOCK_THRESHOLD, CENTER_LINE, TRAIL_MAX_LENGTH, ENABLE_DISPLAY, SAVE_OUTPUT_VIDEO, PIXELS_PER_INCH, LOCATION, DATE_STR, model
    
    # Initialize global variables
    writer = None
    w = 0
    h = 0
    fps = 0
    fish_states = {}
    track_history = defaultdict(list)
    species_counts = defaultdict(lambda: {"Upstream": 0, "Downstream": 0})
    INACTIVE_FRAMES_THRESHOLD = 75
    ADIPOSE_LOCK_THRESHOLD = adipose_lock_threshold
    CENTER_LINE_POSITION = center_line_position
    TRAIL_MAX_LENGTH = trail_max_length
    ENABLE_DISPLAY = enable_display
    SAVE_OUTPUT_VIDEO = save_output_video
    PIXELS_PER_INCH = pixels_per_inch
    LOCATION = location
    DATE_STR = date_str

    # === Load Model & Video ===
    model = YOLO(model_path)
    model.model.eval()
    torch.load = original_load
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.overrides["conf"] = confidence_threshold
    model.overrides["iou"] = 0.25
    model.overrides["tracker"] = "bytetrack.yaml"

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if w <= 0 or h <= 0:
        print("✖ Invalid video dimensions. Check video file.")
        return None, None
    if fps <= 0:
        print("⚠ Warning: Invalid FPS detected, defaulting to 30 FPS")
        fps = 30.0

    CENTER_LINE = int(w * CENTER_LINE_POSITION)
    
    # Define output paths
    csv_path = f"fish_count_{LOCATION.replace(' ', '_')}_{DATE_STR}.csv"
    output_video_path = f"output_{LOCATION.replace(' ', '_')}_{DATE_STR}.mp4"
    
    video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)) if SAVE_OUTPUT_VIDEO else None

    frame_count = 0
    start_time = time.time()

    print(f"Starting fish counting with Maximum Length measurement only")
    print(f"Adipose lock threshold: {ADIPOSE_LOCK_THRESHOLD}")
    print(f"Species requiring wild/hatchery classification: Chinook, Sockeye, Steelhead")

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "VideoTimestamp", "Frame", "TrackID", "Species", "Confidence", "Direction", 
            "X_Percent", "Y_Percent", "Max_Length_Inches", "Location", "Date", "Detection_Count"
        ])
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 2 == 0:
                continue
            if frame_count % 300 == 0:
                cleanup_inactive_tracks(frame_count)
            results = model.track(source=frame, persist=True, device=device, verbose=False)[0]
            cv2.line(frame,(CENTER_LINE,0),(CENTER_LINE,h),(0,255,0),3)
            oy = 60
            cv2.putText(frame,"Live Counts (Max Length):",(10,oy),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            for sp,cnt in sorted(species_counts.items()):
                oy+=20
                cv2.putText(frame,f"{sp}: Up {cnt['Upstream']} Down {cnt['Downstream']}",(10,oy),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)
            process_frame_with_enhanced_classification(results, frame, frame_count)
            if ENABLE_DISPLAY:
                cv2.imshow("Fish Counter", frame)
                if cv2.waitKey(1) == ord("q"): 
                    break
            if SAVE_OUTPUT_VIDEO and video_writer:
                video_writer.write(frame)
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed if elapsed > 0 else 0
                print(f"Frame {frame_count}: {fps_current:.1f} FPS")
    cap.release()
    if ENABLE_DISPLAY: 
        cv2.destroyAllWindows()
    if SAVE_OUTPUT_VIDEO and video_writer: 
        video_writer.release()
    print_final_report()
    return csv_path, output_video_path

def is_in_season(name, loc, dt):
    if "Chinook" in name:
        for key in ["chinook_spring","chinook_summer","chinook_fall"]:
            for start, end in SEASONAL_RANGES[loc].get(key, []):
                if start <= dt <= end:
                    return True
        return False
    for species, ranges in SEASONAL_RANGES[loc].items():
        if species.startswith(name.lower()):
            for start, end in ranges:
                if start <= dt <= end:
                    return True
    return True

def calculate_fish_horizontal_length(x1, y1, x2, y2):
    """
    Calculate horizontal length of fish bounding box in inches.
    This is more biologically accurate than diagonal measurement.
    """
    if PIXELS_PER_INCH <= 0:
        return 0
    horizontal_pixels = abs(x2 - x1)  # Width of bounding box
    return horizontal_pixels / PIXELS_PER_INCH

def extract_adipose_status(species_name):
    """Extract wild/hatchery/unknown status from species name"""
    species_lower = species_name.lower()
    if "wild" in species_lower:
        return "wild"
    elif "hatchery" in species_lower:
        return "hatchery"
    elif "unknown" in species_lower:
        return "unknown"
    else:
        return "unknown"  # Default if no adipose info is present

def extract_base_species(name):
    """Extract the base species from the full classification name"""
    for sp in ["Chinook", "Coho", "Sockeye", "Steelhead", "Pike Minnow", "Bull Trout", "Pink Salmon", "Lamprey", "Suckerfish"]:
        if sp in name:
            return sp
    return name

def classify_by_size_and_adipose(base_species, length_inches, adipose_status="unknown"):
    """Classify fish by size and adipose status using wild/hatchery system"""
    # Define size thresholds (you may need to adjust these based on your requirements)
    SIZE_THRESHOLDS = {
        "Chinook": {"adult": 20.0, "jack": 12.0},
        "Coho": {"adult": 16.0, "jack": 10.0},
        "Bull Trout": {"adult": 14.0}
    }
    
    if base_species in ["Lamprey", "Pike Minnow", "Pink Salmon", "Suckerfish"]:
        return base_species
    
    if base_species == "Chinook":
        if length_inches >= SIZE_THRESHOLDS["Chinook"]["adult"]:
            size_class = "Adult"
        elif length_inches >= SIZE_THRESHOLDS["Chinook"]["jack"]:
            size_class = "Jack"
        else:
            return "Chinook Mini Jack"
        
        if adipose_status in ["wild", "hatchery"]:
            return f"Chinook {size_class} {adipose_status.title()}"
        else:
            return f"Chinook {size_class} Unknown"
    
    if base_species == "Coho":
        if length_inches >= SIZE_THRESHOLDS["Coho"]["adult"]:
            return "Coho Adult"
        elif length_inches >= SIZE_THRESHOLDS["Coho"]["jack"]:
            return "Coho Jack"
        else:
            return "Coho Juvenile"
    
    if base_species == "Sockeye":
        if adipose_status in ["wild", "hatchery"]:
            return f"Sockeye {adipose_status.title()}"
        else:
            return "Sockeye Unknown"
    
    if base_species == "Steelhead":
        if adipose_status in ["wild", "hatchery"]:
            return f"Steelhead {adipose_status.title()}"
        else:
            return "Steelhead Unknown"
    
    if base_species == "Bull Trout":
        return "Bull Trout Adult" if length_inches >= SIZE_THRESHOLDS["Bull Trout"]["adult"] else "Bull Trout Subadult"
    
    return base_species

def determine_direction(prev_x, curr_x, center_line):
    # Handle the case where prev_x is None (first detection)
    if prev_x is None:
        return None
    if prev_x < center_line and curr_x >= center_line:
        return "Downstream"
    elif prev_x >= center_line and curr_x < center_line:
        return "Upstream"
    return None

def cleanup_inactive_tracks(frame_count):
    """Remove fish states and track history for tracks that haven't been seen recently"""
    global fish_states, track_history
    
    inactive_tracks = []
    for tid, fish_state in fish_states.items():
        if fish_state.detections:
            last_frame = fish_state.detections[-1][0]  # Frame number of last detection
            if frame_count - last_frame > INACTIVE_FRAMES_THRESHOLD:
                inactive_tracks.append(tid)
    
    for tid in inactive_tracks:
        if tid in fish_states:
            del fish_states[tid]
        if tid in track_history:
            del track_history[tid]
    
    if inactive_tracks:
        print(f"Cleaned up {len(inactive_tracks)} inactive tracks at frame {frame_count}")

class FishState:
    def __init__(self, track_id):
        self.track_id = track_id
        self.detections = []
        self.frames = []
        self.last_x = None
        self.crossing_count = 0
        self.best_species = None
        self.best_confidence = 0
        self.adipose_status = "unknown"
        self.adipose_confidence = 0
        self.adipose_locked = False  # Once we get a high-confidence wild/hatchery, lock it in
        self.base_species = None
        self.max_length = 0  # Track only the maximum length

    def add_detection(self, frame_count, bbox, species_conf_dict, frame):
        self.detections.append((frame_count, bbox, species_conf_dict))
        self.frames.append(frame)
        
        # Update maximum length
        x1, y1, x2, y2 = bbox
        current_length = calculate_fish_horizontal_length(x1, y1, x2, y2)
        self.max_length = max(self.max_length, current_length)
        
        # Find the best species classification
        for species, conf in species_conf_dict.items():
            if conf > self.best_confidence:
                self.best_confidence = conf
                self.best_species = species
                self.base_species = extract_base_species(species)
                
                # Handle adipose status with locking mechanism
                current_adipose = extract_adipose_status(species)
                
                # If we haven't locked in an adipose status yet
                if not self.adipose_locked:
                    # If this is a high-confidence wild or hatchery classification, lock it in
                    if current_adipose in ["wild", "hatchery"] and conf >= ADIPOSE_LOCK_THRESHOLD:
                        self.adipose_status = current_adipose
                        self.adipose_confidence = conf
                        self.adipose_locked = True
                        print(f"LOCKED: Track {self.track_id} adipose status as '{current_adipose}' with confidence {conf:.3f}")
                    # Otherwise, update with the current classification (including unknown)
                    elif conf > self.adipose_confidence:
                        self.adipose_status = current_adipose
                        self.adipose_confidence = conf
                # If already locked, keep the locked status but update confidence if it's the same status
                elif self.adipose_status == current_adipose and conf > self.adipose_confidence:
                    self.adipose_confidence = conf

    def get_max_length(self):
        """Return the maximum horizontal length across all detections."""
        return self.max_length

    def get_detection_count(self):
        """Return the total number of detections for this fish."""
        return len(self.detections)

    def needs_adipose_determination(self):
        """Check if we still need to determine wild/hatchery status"""
        return (self.base_species in ["Chinook", "Sockeye", "Steelhead"] and 
                not self.adipose_locked and 
                self.adipose_status == "unknown")

    def get_final_classification(self):
        """
        Get final classification using MAXIMUM horizontal length.
        """
        if not self.detections: 
            return "Unknown"
        
        # Use maximum horizontal length for classification
        return classify_by_size_and_adipose(self.base_species or "Unknown", self.max_length, self.adipose_status)

    def get_confidence_score(self):
        return self.best_confidence

    def get_classification_completeness(self):
        """Return how complete our classification is (1.0 = fully classified)"""
        if self.base_species not in ["Chinook", "Sockeye", "Steelhead"]:
            return 1.0  # Species that don't need adipose classification are always complete
        
        if self.adipose_locked or self.adipose_status != "unknown":
            return 1.0  # We have adipose info (either locked or current)
        else:
            return 0.5  # We're missing adipose information

def get_adipose_classification_summary(fish_states):
    """Generate summary of wild/hatchery classifications"""
    summary = defaultdict(lambda: {"Wild": 0, "Hatchery": 0, "Unknown": 0, "Total": 0})
    
    for state in fish_states.values():
        sp = state.base_species
        if sp in ["Chinook", "Sockeye", "Steelhead"]:
            status = state.adipose_status.title()  # Convert to title case
            summary[sp][status] += 1
            summary[sp]["Total"] += 1
    
    return summary

def process_frame_with_enhanced_classification(results, frame, frame_count):
    global writer, w, h, fps, fish_states, track_history, species_counts, model
    
    if results.boxes is None or results.boxes.id is None:
        return
    
    boxes = results.boxes.xyxy.cpu().numpy().astype(int)
    ids = results.boxes.id.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    active_ids = set(ids)

    # Clean up track history for inactive tracks
    for tid in list(track_history):
        if tid not in active_ids:
            del track_history[tid]

    for (x1, y1, x2, y2), tid, conf, cid in zip(boxes, ids, confs, cls_ids):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        raw_species = model.names[cid] if cid < len(model.names) else f"class_{cid}"
        species_conf_dict = {raw_species: conf}
        
        # Initialize or update fish state
        if tid not in fish_states:
            fish_states[tid] = FishState(tid)
        
        fish_state = fish_states[tid]
        fish_state.add_detection(frame_count, (x1, y1, x2, y2), species_conf_dict, frame)
        
        # Update track history for visualization
        trail = track_history[tid]
        trail.append((cx, cy))
        if len(trail) > TRAIL_MAX_LENGTH:
            trail.pop(0)
        
        # Check for direction crossing
        direction = determine_direction(fish_state.last_x, cx, CENTER_LINE)
        if direction:
            final_species = fish_state.get_final_classification()
            confidence_score = fish_state.get_confidence_score()
            completeness = fish_state.get_classification_completeness()
            
            # Count the fish if we have sufficient classification
            if completeness >= 0.75 or not fish_state.needs_adipose_determination():
                # Use maximum horizontal length for final measurement
                max_length_inches = fish_state.get_max_length()
                detection_count = fish_state.get_detection_count()
                
                # Calculate video timestamp
                video_timestamp_seconds = (frame_count * 2) / fps if fps > 0 else 0
                minutes = int(video_timestamp_seconds // 60) if video_timestamp_seconds > 0 else 0
                hours = int(minutes // 60) if minutes > 0 else 0
                minutes = minutes % 60
                seconds = video_timestamp_seconds % 60 if video_timestamp_seconds > 0 else 0
                video_timestamp_formatted = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
                
                # Calculate position percentages
                x_percent = (cx / w) * 100 if w > 0 else 0
                y_percent = (cy / h) * 100 if h > 0 else 0
                
                # Update counts and write to CSV
                species_counts[final_species][direction] += 1
                fish_state.crossing_count += 1
                
                if writer:
                    writer.writerow([
                        video_timestamp_formatted, frame_count, tid, final_species,
                        f"{confidence_score:.2f}", direction, f"{x_percent:.1f}",
                        f"{y_percent:.1f}", f"{max_length_inches:.1f}", LOCATION, DATE_STR,
                        f"{detection_count}"
                    ])
                
                # Generate status info for logging
                adipose_info = ""
                if fish_state.base_species in ["Chinook", "Sockeye", "Steelhead"]:
                    lock_status = "LOCKED" if fish_state.adipose_locked else "CURRENT"
                    adipose_info = f" [Adipose: {fish_state.adipose_status}, {lock_status}, Conf: {fish_state.adipose_confidence:.2f}]"
                
                print(f"COUNTED: Track {tid} ({final_species}) going {direction} at {video_timestamp_formatted}{adipose_info} [Max Length: {max_length_inches:.1f}\"] - Completeness: {completeness:.1f}")
            else:
                print(f"DELAYED: Track {tid} crossed but waiting for better adipose classification (completeness: {completeness:.1f})")
        
        fish_state.last_x = cx
        
        # Visualization - show max length in label
        color = (255, 165, 0) if cx < CENTER_LINE else (0, 255, 255)
        crossing_info = f" (x{fish_state.crossing_count})" if fish_state.crossing_count > 0 else ""
        max_length = fish_state.get_max_length()
        label = f"{fish_state.get_final_classification()} {conf:.2f} ({max_length:.1f}\"){crossing_info}"
        
        if fish_state.needs_adipose_determination():
            label += " [Need Adipose]"
        elif fish_state.adipose_locked:
            label += " [LOCKED]"
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw tracking trail
        for i in range(1, len(trail)):
            cv2.line(frame, trail[i - 1], trail[i], color, 2)

def print_final_report():
    global fish_states, species_counts
    
    print(f"\n--- Results for {LOCATION} on {DATE_STR} ---")
    total_upstream = total_downstream = 0
    
    for sp, cnt in sorted(species_counts.items()):
        upstream, downstream = cnt['Upstream'], cnt['Downstream']
        total_upstream += upstream
        total_downstream += downstream
        print(f"  {sp:40s} Up={upstream:3d} Down={downstream:3d}")
    
    print(f"  {'TOTAL':40s} Up={total_upstream:3d} Down={total_downstream:3d}")
    
    print(f"\n--- Wild/Hatchery Analysis ---")
    adipose_summary = get_adipose_classification_summary(fish_states)
    
    for species, stats in adipose_summary.items():
        if stats['Total'] > 0:
            print(f"  {species}:")
            for status in ["Wild", "Hatchery", "Unknown"]:
                pct = (stats[status] / stats["Total"]) * 100 if stats["Total"] > 0 else 0
                print(f"    {status:8}: {stats[status]:3d} ({pct:5.1f}%)")
            print(f"    Total:      {stats['Total']:3d}")
    
    # Classification quality metrics
    complete = sum(1 for fs in fish_states.values() if fs.get_classification_completeness() >= 0.75)
    locked = sum(1 for fs in fish_states.values() if fs.adipose_locked)
    total_fish = len(fish_states)
    
    if total_fish > 0:
        print(f"\n--- Classification Quality ---")
        print(f"  Complete classifications: {complete}/{total_fish} ({(complete/total_fish)*100:.1f}%)")
        print(f"  Locked wild/hatchery:     {locked}/{total_fish} ({(locked/total_fish)*100:.1f}%)")
    
    # Max length summary
    if fish_states:
        print(f"\n--- Maximum Length Analysis ---")
        max_lengths = [fs.get_max_length() for fs in fish_states.values() if fs.get_max_length() > 0]
        
        if max_lengths:
            avg_max_length = sum(max_lengths) / len(max_lengths)
            largest_fish = max(max_lengths)
            smallest_fish = min(max_lengths)
            
            print(f"  Total fish measured:     {len(max_lengths)}")
            print(f"  Average maximum length:  {avg_max_length:.1f} inches")
            print(f"  Largest fish recorded:   {largest_fish:.1f} inches")
            print(f"  Smallest fish recorded:  {smallest_fish:.1f} inches")

if __name__ == "__main__":
    run_adipose_counter()