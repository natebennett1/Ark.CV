# Ark.CV Fish Counting Pipeline

This repository contains a single-script computer vision pipeline for counting and
classifying fish passages using an Ultralytics YOLO model. The script ingests a
video, tracks detections across frames, applies domain-specific species logic,
and exports both an annotated video and CSV-based analytics.

## Repository Structure

- `model0903.py` – Main entry point that loads the YOLO model, performs
  detection, tracks fish with BoT-SORT, applies seasonal and location-based
  logic, triggers human-in-the-loop (HITL) exports, and writes per-track
  measurements.
- `README.md` – This guide.

## Requirements

- Python 3.9+
- GPU optional (CUDA acceleration improves throughput).
- Recommended Python dependencies:
  ```bash
  pip install ultralytics==8.2.0 opencv-python torch torchvision torchaudio
  pip install filterpy scikit-image lap scipy
  ```
  The additional packages (`filterpy`, `scikit-image`, `lap`, `scipy`) ensure
  BoT-SORT has the motion and appearance modules it expects.

## Quick Start

1. **Update local paths** in `model0903.py`:
   - `MODEL_PATH` – Absolute path to your trained YOLO weights (`.pt`).
   - `VIDEO_PATH` – Absolute path to the input video.
   - `LOCATION`, `DATE_STR` – Metadata for seasonal logic and reporting.

2. **Run the script**:
   ```bash
   python model0903.py
   ```

3. **Outputs**:
   - `fish_counts_<timestamp>.csv` – Per-track species counts with direction,
     confidence, and positional metadata.
   - `annotated_video_<timestamp>.mp4` – Visualization with bounding boxes,
     class decisions, and direction overlays (if `SAVE_OUTPUT_VIDEO = True`).
   - `hitl_queue/` – Crops queued for analyst review when detections are
     low-confidence or ambiguous.

## BoT-SORT Tracking

The pipeline now uses **BoT-SORT** instead of ByteTrack for multi-object
tracking. BoT-SORT augments motion cues with appearance embeddings and global
motion compensation, making it more robust to occlusions, variable lighting, and
non-linear fish trajectories. Configuration defaults are defined inline in
`model0903.py`, and the script automatically falls back to Ultralytics'
`botsort.yaml` if the running Ultralytics build does not accept dictionary-based
tracker overrides.

If you need to tune the tracker further, adjust the `tracker_cfg` dictionary in
`model0903.py` or point the fallback to a custom YAML file.

## Human-in-the-Loop Workflow

The `HITLCollector` class buffers low-confidence tracks, exporting cropped frame
snippets for analyst review. Fine-tune the thresholds via:

- `LOWCONF_THRESHOLD` – Minimum detection confidence to be considered "safe".
- `HITL_EXPAND_RATIO` – Padding ratio applied to crops.
- `HITL_TRACK_GAP_FRAMES` – Frames to wait before finalizing a track bundle.

Resulting CSVs (`hitl_manifest.csv`) and crops/images are saved in
`hitl_queue/<track_id>/` folders.

## Recommended Next Steps for the Fish Counting Model

1. **Dataset Expansion & Active Learning**
   - Prioritize labeling the low-confidence crops emitted by the HITL pipeline
     to balance under-represented species and lighting conditions.
   - Incorporate hard negatives (e.g., debris, bubbles) to reduce false
     positives.

2. **Appearance Embeddings for Re-ID**
   - Integrate a lightweight ReID backbone (such as OSNet or a MobileNet-based
     embedder) to enhance BoT-SORT's appearance matching, especially in crowded
     scenes or when fish reverse direction.

3. **Temporal Consistency Checks**
   - Apply a temporal smoothing layer on class confidences (e.g., exponential
     moving average) to complement the majority-vote mechanism, reducing label
     flicker on noisy frames.

4. **Adaptive Thresholding**
   - Consider dynamic confidence thresholds based on scene brightness or camera
     gain metadata. This can prevent over-suppression during night scenes while
     keeping daytime detections precise.

5. **Automated QA & Monitoring**
   - Add unit tests or smoke tests that validate key configuration invariants
     (e.g., species lists per location) and run sample frames through the model
     to catch regressions before deployment.

6. **Model Architecture Exploration**
   - Evaluate YOLOv8-seg or transformer-based detectors to capture fine-grained
     fin features that influence adipose classification, or incorporate a
     two-stage head where a specialized classifier refines species-specific
     cues.

## Support

For questions or assistance, please document the environment, Ultralytics
version, GPU availability, and attach sample frames exhibiting the issue. This
context helps triage tracking or classification anomalies quickly.
