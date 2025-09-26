#!/usr/bin/env python3
import os
import cv2
import time
import csv
import numpy as np
import torch
import traceback
from datetime import datetime, date
from ultralytics import YOLO
from collections import defaultdict, deque
from pathlib import Path
import hashlib

# =========================
# 0) PATHS & BASIC SETTINGS
# =========================
MODEL_PATH  = os.path.expanduser(r"C:\Users\20ben\Downloads\weights93.pt")
VIDEO_PATH  = os.path.expanduser(r"C:\Users\20ben\Downloads\20240711_800_100.mp4")
LOCATION    = "Wells Dam"
DATE_STR    = "2025-08-31"  # YYYY-MM-DD

# Optional second-pass adipose model (leave empty to disable gracefully)
ADIPOSE_MODEL_PATH = ""  # e.g., r"C:\Users\20ben\Downloads\adipose_head.pt"

# Validate paths
if not os.path.isfile(MODEL_PATH):
    print(f"✖ Model file not found at: {MODEL_PATH}")
    raise SystemExit(1)
print(f"✔ Using model weights: {MODEL_PATH}")

if not os.path.isfile(VIDEO_PATH):
    print(f"✖ Video file not found: {VIDEO_PATH}")
    raise SystemExit(1)
print(f"✔ Using video file: {VIDEO_PATH}")

try:
    input_date = datetime.strptime(DATE_STR, "%Y-%m-%d").date()
except ValueError:
    print("✖ Invalid date format in DATE_STR constant. Use YYYY-MM-DD.")
    raise SystemExit(1)

# ===================
# 1) CONFIG CONSTANTS
# ===================
file_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_PATH             = f"fish_counts_{file_timestamp}.csv"
CONFIDENCE_THRESHOLD = 0.10  # detector confidence floor
UNKNOWN_THRESHOLD    = 0.45   # ← raised so low-conf becomes *_U
CENTER_LINE_POSITION = 0.50
TRAIL_MAX_LENGTH     = 30
ENABLE_DISPLAY       = False
SAVE_OUTPUT_VIDEO    = True
OUTPUT_VIDEO_PATH    = f"annotated_video_{file_timestamp}.mp4"

# Per-class minimum acceptance (base-species level)
MIN_CLASS_CONF = {
    "BullTrout": 0.80  # stricter to stop over-diagnosis
}

# HITL / Active Learning
HITL_OUT_DIR          = "./hitl_queue"
LOWCONF_THRESHOLD     = 0.80
COUNT_REVIEW_THRESHOLD = 0.80  # flag counted fish under this confidence for HITL review
HITL_EXPAND_RATIO     = 0.15
HITL_TRACK_GAP_FRAMES = 45
HITL_DEDUP            = True

# Size calibration
PIXELS_PER_INCH = 25.253

SIZE_THRESHOLDS = {
    "Chinook":  {"adult": 22, "jack": 12, "mini_jack": 0},
    "Coho":     {"adult": 18, "jack": 12},
    "Sockeye":  {"adult": 20},
    "Steelhead":{"adult": 24},
    "BullTrout":{"adult": 12}
}

# ============================
# 2) SPECIES & SEASONALITY MAP
# ============================
SPECIES_BY_LOCATION = {
    "Wells Dam": [
        "Chinook_AA","Chinook_AP","Chinook_U",
        "Coho_AA","Coho_AP","Coho_U",
        "Sockeye_AA","Sockeye_AP","Sockeye_U",
        "Steelhead_AA","Steelhead_AP","Steelhead_U",
        "Lamprey","Pike","BullTrout","Suckerfish","ResidentFish","Pink"
    ],
}

# Season windows (default); Coho override handled below
def wells_coho_allowed(d: date) -> bool:
    # Coho not counted before Sept 10 at Wells
    return d >= date(d.year, 9, 10) and d <= date(d.year, 11, 30)

SEASONAL_RANGES = {
    "Wells Dam": {
        "chinook_spring": [(date(input_date.year,5,1),  date(input_date.year,6,28))],
        "chinook_summer": [(date(input_date.year,6,29), date(input_date.year,8,28))],
        "chinook_fall":   [(date(input_date.year,8,29), date(input_date.year,11,15))],
        "sockeye_run":    [(date(input_date.year,6,1),  date(input_date.year,9,30))],
        "steelhead_run":  [(date(input_date.year,3,1),  date(input_date.year,5,31)),
                           (date(input_date.year,9,1),  date(input_date.year,11,30))],
        "lamprey_run":    [(date(input_date.year,6,1),  date(input_date.year,9,30))],
    }
}

def extract_base_species(label: str) -> str:
    if "_" in label:
        return label.split("_", 1)[0]
    s = label.strip()
    if s == "Pike Minnow": return "Pike"
    if s == "Bull Trout":  return "BullTrout"
    if s == "Resident Fish - sp.": return "ResidentFish"
    if s == "Pink Salmon": return "Pink"
    for k in ["Chinook","Coho","Sockeye","Steelhead","Pike","BullTrout","Pink","Lamprey","Suckerfish","ResidentFish"]:
        if k in s:
            return k
    return s

def normalize_legacy_output(label: str) -> str:
    s = label.strip()
    def mk(base, s_): 
        return f"{base}_{'AP' if 'Present' in s_ else 'AA' if 'Absent' in s_ else 'U'}"
    if "Chinook"   in s and "Adipose" in s: return mk("Chinook", s)
    if "Sockeye"   in s and "Adipose" in s: return mk("Sockeye", s)
    if "Steelhead" in s and "Adipose" in s: return mk("Steelhead", s)
    if "Coho"      in s and "Adipose" in s: return mk("Coho", s)
    if s == "Pike Minnow": return "Pike"
    if s == "Bull Trout":  return "BullTrout"
    if s == "Resident Fish - sp.": return "ResidentFish"
    if s == "Pink Salmon": return "Pink"
    return s

def is_in_season(full_label: str, loc: str, d: date) -> bool:
    base = extract_base_species(full_label)
    seasons = SEASONAL_RANGES.get(loc, {})
    if base == "Chinook":
        for key in ["chinook_spring","chinook_summer","chinook_fall"]:
            for start, end in seasons.get(key, []):
                if start <= d <= end:
                    return True
        return False
    if base == "Coho":
        if loc == "Wells Dam":
            return wells_coho_allowed(d)  # Sept 10 start
        return date(d.year,8,1) <= d <= date(d.year,11,30)
    if base == "Sockeye":
        return any(s <= d <= e for s,e in seasons.get("sockeye_run", []))
    if base == "Steelhead":
        return any(s <= d <= e for s,e in seasons.get("steelhead_run", []))
    if base == "Lamprey":
        return any(s <= d <= e for s,e in seasons.get("lamprey_run", []))
    return True

def adipose_tag_from_words(status: str) -> str:
    return {"Present":"AP","Absent":"AA"}.get(status, "U")

def classify_by_size_and_adipose(base_species: str, adipose_status="Unknown"):
    if base_species in {"Chinook","Sockeye","Steelhead","Coho"}:
        return f"{base_species}_{adipose_tag_from_words(adipose_status)}"
    return base_species

def calculate_fish_length(x1,y1,x2,y2):
    diagonal_pixels = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
    return diagonal_pixels / PIXELS_PER_INCH

# =====================================
# 3) PYTORCH LOAD PATCH (compatibility)
# =====================================
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

print(f"Attempting to load model from: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    model.model.eval()
    torch.load = original_load
    print("✔ Custom model loaded successfully")
except Exception as e:
    print(f"✖ Failed to load model: {e}")
    traceback.print_exc()
    torch.load = original_load
    raise SystemExit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"✔ Model moved to {device}")

# Inference settings
model.overrides["conf"] = CONFIDENCE_THRESHOLD
model.overrides["iou"]  = 0.25           # ← favor recall on overlaps
model.overrides["max_det"]  = 200
# We'll pass a dict tracker config directly in model.track(...) below

# ===============================
# 4) OPTIONAL ADIPOSE 2ND-PASS
# ===============================
adipose_model = None
if ADIPOSE_MODEL_PATH and os.path.isfile(ADIPOSE_MODEL_PATH):
    try:
        adipose_model = YOLO(ADIPOSE_MODEL_PATH).to(device)
        adipose_model.model.eval()
        print(f"✔ Adipose model loaded: {ADIPOSE_MODEL_PATH}")
    except Exception as e:
        print(f"⚠ Could not load adipose model: {e}")
        adipose_model = None
else:
    print("ℹ No adipose model provided; skipping second-pass refinement.")

def _expand_box(x1,y1,x2,y2,w,h,ratio=0.20):
    bw, bh = (x2-x1), (y2-y1)
    dx, dy = int(bw*ratio), int(bh*ratio)
    return max(0,x1-dx), max(0,y1-dy), min(w-1,x2+dx), min(h-1,y2+dy)

def infer_adipose_status(frame, box, expand=0.20, min_conf=0.50):
    if adipose_model is None:
        return "Unknown", 0.0
    x1,y1,x2,y2 = box
    h, w = frame.shape[:2]
    ex1,ey1,ex2,ey2 = _expand_box(x1,y1,x2,y2,w,h,expand)
    crop = frame[ey1:ey2, ex1:ex2]
    if crop.size == 0:
        return "Unknown", 0.0
    r = adipose_model.predict(source=crop, verbose=False)[0]
    # classifier head path
    if hasattr(r, "probs") and r.probs is not None:
        probs = r.probs.data.float().cpu().numpy()
        idx = int(np.argmax(probs))
        names = getattr(adipose_model, "names", {0:"Absent",1:"Present"})
        label = names.get(idx, "Unknown")
        return (label if probs[idx] >= min_conf else "Unknown", float(probs[idx]))
    # tiny detector path
    if r.boxes is None or len(r.boxes)==0:
        return "Unknown", 0.0
    confs = r.boxes.conf.float().cpu().numpy()
    cids  = r.boxes.cls.int().cpu().numpy()
    best  = int(np.argmax(confs))
    names = getattr(adipose_model, "names", {0:"Absent",1:"Present"})
    label = names.get(int(cids[best]), "Unknown")
    return (label if confs[best] >= min_conf else "Unknown", float(confs[best]))

# =============================
# 5) HITL LOW-CONF COLLECTOR
# =============================
class HITLCollector:
    def __init__(self, out_root, lowconf_threshold, expand_ratio, track_gap_frames, dedup=True):
        self.out_root = out_root
        self.lowconf_threshold = float(lowconf_threshold)
        self.expand_ratio = float(expand_ratio)
        self.track_gap_frames = int(track_gap_frames)
        self.dedup = dedup
        Path(self.out_root).mkdir(parents=True, exist_ok=True)
        self.meta_csv_path = os.path.join(self.out_root, "metadata.csv")
        if not os.path.exists(self.meta_csv_path):
            with open(self.meta_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "saved_at_utc","location","date_str","video_name",
                    "frame_idx","timestamp_sec","track_id",
                    "pred_class","pred_conf",
                    "bbox_x1","bbox_y1","bbox_x2","bbox_y2",
                    "crop_path","frame_path","direction","notes"
                ])
        self.tracks = {}
        self._seen_hashes = set()

    @staticmethod
    def _lap_var(img):
        if img is None or img.size == 0: return 0.0
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    @staticmethod
    def _sha1(img):
        small = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
        return hashlib.sha1(small.tobytes()).hexdigest()

    def _candidate_score(self, crop, area, conf):
        sharp = self._lap_var(crop)
        proximity = -(self.lowconf_threshold - conf)
        return (sharp, area, proximity)

    def _dir_for(self, date_str, location, pred_cls_name, category=None):
        date_dir = date_str.replace("-", "")
        species_dir = (pred_cls_name or "unknown").replace(" ", "_").lower()
        location_dir = location.replace(" ", "_")
        if category:
            return os.path.join(self.out_root, category, date_dir, location_dir, species_dir)
        return os.path.join(self.out_root, date_dir, location_dir, species_dir)

    def _write_row(self, row):
        with open(self.meta_csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    def observe(self, frame, frame_idx, timestamp_sec, video_name,
                location, date_str, x1, y1, x2, y2,
                pred_cls_name, pred_conf, track_id, direction=None):
        if track_id is None: return
        h, w = frame.shape[:2]
        st = self.tracks.get(track_id, {"last_seen": frame_idx, "high_conf_seen": False, "best_candidate": None})
        st["last_seen"] = frame_idx
        conf = float(pred_conf)
        if conf >= self.lowconf_threshold:
            st["high_conf_seen"] = True
            st["best_candidate"] = None
            self.tracks[track_id] = st
            return
        if st.get("high_conf_seen"): 
            self.tracks[track_id] = st
            return
        x1i,y1i,x2i,y2i = int(x1),int(y1),int(x2),int(y2)
        ex1,ey1,ex2,ey2 = _expand_box(x1i,y1i,x2i,y2i,w,h,HITL_EXPAND_RATIO)
        crop = frame[ey1:ey2, ex1:ex2]
        if crop is None or crop.size == 0:
            self.tracks[track_id] = st
            return
        if HITL_DEDUP:
            sig = self._sha1(crop)
            if sig in self._seen_hashes:
                self.tracks[track_id] = st
                return
        area = max(1, (ex2-ex1)*(ey2-ey1))
        score = self._candidate_score(crop, area, conf)
        cand = {"crop":crop,"frame":frame.copy(),"bbox":(x1i,y1i,x2i,y2i),"frame_idx":int(frame_idx),
                "timestamp_sec":float(timestamp_sec),"video_name":video_name,"location":location,"date_str":date_str,
                "pred_cls_name":pred_cls_name,"pred_conf":conf,"direction":direction,"score":score}
        best = st.get("best_candidate")
        if (best is None) or (cand["score"] > best["score"]):
            st["best_candidate"] = cand
            if HITL_DEDUP: self._seen_hashes.add(sig)
        self.tracks[track_id] = st

    def _flush_track(self, tid):
        st = self.tracks.get(tid)
        if not st or st.get("high_conf_seen"): return
        cand = st.get("best_candidate")
        if not cand: return
        out_dir = self._dir_for(cand["date_str"], cand["location"], cand["pred_cls_name"])
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base = f"{ts}_f{cand['frame_idx']:06d}_t{tid}_{cand['pred_cls_name']}_{cand['pred_conf']:.2f}"
        crop_path  = os.path.join(out_dir, base + "_crop.jpg")
        frame_path = os.path.join(out_dir, base + "_frame.jpg")
        cv2.imwrite(crop_path,  cand["crop"],  [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(frame_path, cand["frame"], [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        x1,y1,x2,y2 = cand["bbox"]
        self._write_row([
            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            cand["location"], cand["date_str"], cand["video_name"],
            cand["frame_idx"], f"{cand['timestamp_sec']:.3f}", tid,
            cand["pred_cls_name"], f"{cand['pred_conf']:.4f}",
            int(x1), int(y1), int(x2), int(y2),
            crop_path, frame_path,
            cand["direction"] if cand["direction"] is not None else "",
            "low_conf_track_best"
        ])

    def gc_inactive(self, current_frame_idx):
        to_remove = []
        for tid, st in self.tracks.items():
            if current_frame_idx - st["last_seen"] > HITL_TRACK_GAP_FRAMES:
                self._flush_track(tid); to_remove.append(tid)
        for tid in to_remove:
            self.tracks.pop(tid, None)

    def flush_all(self):
        for tid in list(self.tracks.keys()):
            self._flush_track(tid)
            self.tracks.pop(tid, None)

    def flag_count_event(self, frame, frame_idx, timestamp_sec, video_name,
                         location, date_str, x1, y1, x2, y2,
                         pred_cls_name, pred_conf, track_id, direction=None):
        if frame is None:
            return
        x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)
        x1i, y1i = max(0, x1i), max(0, y1i)
        h, w = frame.shape[:2]
        x2i, y2i = min(w, max(x1i + 1, x2i)), min(h, max(y1i + 1, y2i))
        crop = frame[y1i:y2i, x1i:x2i]
        if crop is None or crop.size == 0:
            return
        annotated = frame.copy()
        cv2.rectangle(annotated, (x1i, y1i), (x2i, y2i), (0, 0, 255), 2)
        out_dir = self._dir_for(date_str, location, pred_cls_name, category="count_review")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base = f"{ts}_f{frame_idx:06d}_t{track_id}_{pred_cls_name}_{pred_conf:.2f}"
        crop_path = os.path.join(out_dir, base + "_crop.jpg")
        frame_path = os.path.join(out_dir, base + "_frame.jpg")
        cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        cv2.imwrite(frame_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        self._write_row([
            datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            location,
            date_str,
            video_name,
            frame_idx,
            f"{timestamp_sec:.3f}",
            track_id,
            pred_cls_name,
            f"{pred_conf:.4f}",
            int(x1i),
            int(y1i),
            int(x2i),
            int(y2i),
            crop_path,
            frame_path,
            direction if direction is not None else "",
            "count_below_threshold"
        ])

collector = HITLCollector(HITL_OUT_DIR, LOWCONF_THRESHOLD, HITL_EXPAND_RATIO, HITL_TRACK_GAP_FRAMES, HITL_DEDUP)

# ======================================
# 6) TEMPORAL STABILITY & DEBOUNCE RULES
# ======================================
STABILITY_WINDOW = 3     # frames per track for majority vote (↑)
ADIPOSE_WINDOW   = 3
CROSS_DELTA_PCT  = 0.03  # require ~3% width past center to count (↑)
COUNT_COOLDOWN   = 0     # frames before same track can count again

def determine_direction(prev_x, curr_x, center_line, w):
    # Debounced: only if we cross and land beyond center +/- delta
    delta = int(w * CROSS_DELTA_PCT)
    if prev_x < (center_line - 0) and curr_x >= (center_line + delta):
        return "Downstream"
    if prev_x > (center_line + 0) and curr_x <= (center_line - delta):
        return "Upstream"
    return None

# Prepare an inline ByteTrack configuration (prefer this; otherwise use a tuned bytetrack.yaml)
tracker_cfg = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.1,
    "track_low_thresh": 0.05,
    "new_track_thresh": 0.1,
    "match_thresh": 0.8,
    "track_buffer": 30,      # ← important: ride out brief occlusions
    "max_time_lost": 30
}

# ===============================
# 7) OPEN VIDEO AND MAIN PIPELINE
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"✖ Could not open video: {VIDEO_PATH}")
    raise SystemExit(1)

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = float(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
CENTER_LINE = int(w * CENTER_LINE_POSITION)

print(f"Video info: {w}x{h} @ {fps:.1f}fps, {total_frames} frames")
print(f"Size calibration: {PIXELS_PER_INCH:.2f} px/in")

video_writer = None
if SAVE_OUTPUT_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
    print(f"✔ Will save annotated video to: {OUTPUT_VIDEO_PATH}")

fish_states    = {}
track_history  = defaultdict(list)
species_counts = defaultdict(lambda: {"Upstream":0,"Downstream":0})

frame_count = 0
start_time = time.time()

def majority_vote(q: deque):
    if not q: return None
    # break ties by most recent
    counts = defaultdict(int)
    for x in q: counts[x] += 1
    best = max(counts.items(), key=lambda kv: (kv[1], list(q)[::-1].index(kv[0])))
    return best[0]

with open(CSV_PATH, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["VideoTimestamp","Frame","TrackID","Species","Confidence","Direction","X_Percent","Y_Percent","Length_Inches","Location","Date"])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n▶️ End of video.")
            break
        frame_count += 1

        # NO frame skipping (we want every crossing)
        video_ts_sec_now = (frame_count / fps) if fps > 0 else 0

        # Run tracker with inline config (falls back to default if dict unsupported)
        try:
            results = model.track(
                source=frame,
                persist=True,
                device=device,
                verbose=False,
                tracker=tracker_cfg
            )[0]
        except TypeError:
            # Some older Ultralytics builds don’t accept dict; use default YAML
            results = model.track(
                source=frame,
                persist=True,
                device=device,
                verbose=False,
                tracker="bytetrack.yaml"
            )[0]

        # Draw center line + live counts
        cv2.line(frame, (CENTER_LINE,0), (CENTER_LINE,h), (0,255,0), 3)
        oy = 60
        cv2.putText(frame,"Live Counts:",(10,oy),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        for sp,cnt in sorted(species_counts.items()):
            oy+=20
            cv2.putText(frame,f"{sp}: Up {cnt['Upstream']} Down {cnt['Downstream']}",(10,oy),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)

        if results.boxes is None or results.boxes.id is None:
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {frame_count} frames in {elapsed:.1f}s")
            if ENABLE_DISPLAY:
                cv2.imshow("Fish Counter", frame)
                if cv2.waitKey(1) == ord("q"): break
            if SAVE_OUTPUT_VIDEO and video_writer:
                video_writer.write(frame)
            collector.gc_inactive(frame_count)
            continue

        boxes   = results.boxes.xyxy.cpu().numpy().astype(int)
        ids     = results.boxes.id.cpu().numpy().astype(int)
        confs   = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        active_ids = set(ids)

        # Cleanup trails
        for tid in list(track_history):
            if tid not in active_ids:
                del track_history[tid]

        for (x1,y1,x2,y2), tid, conf, cid in zip(boxes, ids, confs, cls_ids):
            cx, cy = (x1+x2)//2, (y1+y2)//2
            length_inches = calculate_fish_length(x1,y1,x2,y2)

            # Raw label from model (composite if trained that way)
            raw_species = model.names[cid] if cid < len(model.names) else f"class_{cid}"
            raw_species = normalize_legacy_output(raw_species)

            # Base species & initial acceptance by thresholds
            species_final = "Unknown"
            base = extract_base_species(raw_species)

            # Apply per-class minimums first
            min_req_conf = max(UNKNOWN_THRESHOLD, MIN_CLASS_CONF.get(base, UNKNOWN_THRESHOLD))

            if conf >= min_req_conf:
                # Season & locality filters
                if raw_species in SPECIES_BY_LOCATION[LOCATION] and is_in_season(raw_species, LOCATION, input_date):
                    species_final = raw_species
                else:
                    # try to map to an expected label with same base (e.g., choose _U if others disallowed)
                    fallback = None
                    for expected in SPECIES_BY_LOCATION[LOCATION]:
                        if extract_base_species(expected) == base and is_in_season(expected, LOCATION, input_date):
                            fallback = expected
                            break
                    species_final = fallback if fallback else "Unknown"
            else:
                species_final = "Unknown"

            # Optional adipose refinement: only for salmonids and only if suffix is U
            adipose_species = {"Chinook","Sockeye","Steelhead","Coho"}
            if extract_base_species(species_final) in adipose_species:
                suffix = species_final.split("_",1)[1] if "_" in species_final else "U"
                if suffix == "U":
                    status_word, status_conf = infer_adipose_status(frame, (x1,y1,x2,y2), expand=0.20, min_conf=0.50)
                    if status_word in {"Present","Absent"}:
                        species_final = f"{extract_base_species(species_final)}_{adipose_tag_from_words(status_word)}"

            # HITL observation
            collector.observe(
                frame=frame,
                frame_idx=frame_count,
                timestamp_sec=video_ts_sec_now,
                video_name=os.path.basename(VIDEO_PATH),
                location=LOCATION,
                date_str=DATE_STR,
                x1=x1, y1=y1, x2=x2, y2=y2,
                pred_cls_name=raw_species,
                pred_conf=conf,
                track_id=tid,
                direction=None
            )

            # Trails
            trail = track_history[tid]
            trail.append((cx,cy))
            if len(trail) > TRAIL_MAX_LENGTH:
                trail.pop(0)

            # State init/update
            st = fish_states.get(tid)
            if st is None:
                st = fish_states[tid] = {
                    "last_x": cx,
                    "species_votes": deque(maxlen=STABILITY_WINDOW),
                    "adipose_votes": deque(maxlen=ADIPOSE_WINDOW),
                    "length": length_inches,
                    "last_count_frame": -10**9,
                    "crossing_count": 0,
                    "last_conf": 0.0
                }

            # Temporal voting
            st["species_votes"].append(species_final)
            if "_" in species_final:
                st["adipose_votes"].append(species_final.split("_",1)[1])

            st["last_conf"] = float(conf)

            stable_species = majority_vote(st["species_votes"]) or "Unknown"

            # Direction with debounce + cooldown
            direction = determine_direction(st["last_x"], cx, CENTER_LINE, w)
            can_count = direction is not None and (frame_count - st["last_count_frame"] >= COUNT_COOLDOWN)

            if can_count and stable_species != "Unknown":
                video_timestamp_seconds = (frame_count / fps) if fps > 0 else 0
                video_timestamp_formatted = f"{int(video_timestamp_seconds//3600):02d}:{int((video_timestamp_seconds%3600)//60):02d}:{video_timestamp_seconds%60:06.3f}"
                x_percent = (cx / w) * 100 if w > 0 else 0
                y_percent = (cy / h) * 100 if h > 0 else 0

                species_counts[stable_species][direction] += 1
                st["crossing_count"] += 1
                st["last_count_frame"] = frame_count

                writer.writerow([
                    video_timestamp_formatted,
                    frame_count,
                    tid,
                    stable_species,
                    f"{conf:.2f}",
                    direction,
                    f"{x_percent:.1f}",
                    f"{y_percent:.1f}",
                    f"{st['length']:.1f}",
                    LOCATION,
                    DATE_STR
                ])
                print(f"COUNTED: Track {tid} ({stable_species}, {st['length']:.1f}in) {direction} at {video_timestamp_formatted} - Crossing #{st['crossing_count']}")

                count_confidence = st.get("last_conf", float(conf))
                if count_confidence < COUNT_REVIEW_THRESHOLD:
                    collector.flag_count_event(
                        frame=frame,
                        frame_idx=frame_count,
                        timestamp_sec=video_ts_sec_now,
                        video_name=os.path.basename(VIDEO_PATH),
                        location=LOCATION,
                        date_str=DATE_STR,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        pred_cls_name=stable_species,
                        pred_conf=count_confidence,
                        track_id=tid,
                        direction=direction
                    )
                    print(f"⚠️  FLAGGED for review: Track {tid} counted as {stable_species} with confidence {count_confidence:.2f} (<{COUNT_REVIEW_THRESHOLD:.2f})")

            # Update last_x and draw
            st["last_x"] = cx
            color = (255,165,0) if cx < CENTER_LINE else (0,255,255)
            crossing_info = f" (x{st.get('crossing_count',0)})" if st.get('crossing_count',0) > 0 else ""
            draw_label = f"{stable_species} {length_inches:.1f}in {conf:.2f}{crossing_info}" + (f" ({direction})" if direction else "")
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
            cv2.putText(frame,draw_label,(x1,max(15,y1-10)),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            for i in range(1,len(trail)):
                cv2.line(frame, trail[i-1], trail[i], color, 2)

        # display/save & GC
        if ENABLE_DISPLAY:
            cv2.imshow("Fish Counter", frame)
            if cv2.waitKey(1) == ord("q"):
                print("User requested quit."); break
        if SAVE_OUTPUT_VIDEO and video_writer:
            video_writer.write(frame)
        collector.gc_inactive(frame_count)

        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps_current = frame_count / elapsed if elapsed > 0 else 0
            total_fish = sum(sum(v.values()) for v in species_counts.values())
            print(f"Frame {frame_count}: {fps_current:.1f} FPS | Total fish: {total_fish}")

# ======================
# 8) CLEANUP & REPORTING
# ======================
cap.release()
if ENABLE_DISPLAY:
    cv2.destroyAllWindows()
if SAVE_OUTPUT_VIDEO and video_writer:
    video_writer.release()
    print(f"✔ Annotated video saved to: {OUTPUT_VIDEO_PATH}")

collector.flush_all()
print(f"✔ HITL crops & metadata saved under: {os.path.abspath(HITL_OUT_DIR)}")

elapsed_total = time.time() - start_time
print(f"\n--- Processing Complete ---")
print(f"Total frames processed: {frame_count}")
print(f"Total time: {elapsed_total:.1f}s")   
print(f"Average FPS: {frame_count/elapsed_total:.1f}")

print(f"\n--- Results for {LOCATION} on {DATE_STR} ---")
for sp,cnt in sorted(species_counts.items()):
    print(f"  {sp:30s} Up={cnt['Upstream']} Down={cnt['Downstream']}")
print(f"  TOTAL                        Up={sum(c['Upstream'] for c in species_counts.values())} Down={sum(c['Downstream'] for c in species_counts.values())}")
print(f"Counts saved to {CSV_PATH}")
