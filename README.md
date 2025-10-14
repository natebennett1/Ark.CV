# Ark.CV Fish Counting Pipeline

This repository contains a computer vision pipeline for counting and
classifying fish passages using an Ultralytics YOLO model. The pipeline ingests a
video, tracks detections across frames, applies domain-specific species logic,
and exports both an annotated video and CSV-based analytics.

## Requirements

- Python 3.12
- GPU optional (CUDA acceleration improves throughput).
- Noteworthy Python dependencies (this README will walk you through installing them later):
  - `ultralytics==8.2.0`
  - `opencv-python`
  - `torch`, `torchvision`,`torchaudio`
  - (`filterpy`, `scikit-image`, `lap`, `scipy`) ensure
  BoT-SORT has the motion and appearance modules it expects.

## Repository Structure
```
configs/
├── wells_dam_test.json             # Default pipeline configuration
└── template.json                   # Example template for overriding configurations

src/
├── classification/                 # Species classification with business rules
│   ├── species_classifier.py       # Main class that performs species classifications
│   └── species_rules.py            # Business logic for species/seasons
│
├── config/                         # All configuration management
│   ├── config_loader.py            # Loads pipeline config from .json file
│   └── settings.py                 # Centralized config with validation
│
├── detection/                      # YOLO model management
│   ├── adipose_detector.py         # Secondary adipose fin detection
│   └── detector.py                 # Main fish detector with tracking
│
├── io/                             # Video processing and output management
│   ├── output_writer.py            # CSV output and statistics
│   └── video_processor.py          # Video I/O with cloud support
│
├── quality/                        # Human-in-the-loop data collection
│   ├── clip_recorder.py            # Handles the recording of video clips around QA events
│   ├── manual_review_collector.py  # Orchestrates capturing of all QA events
│   ├── occlusion_detector.py       # Performs occlusion detection using graph theory
│   └── quality_event.py            # QA event data class
│
└── tracking/                       # Fish tracking and state management
    ├── fish_state.py               # Individual fish state tracking
    └── tracker.py                  # Direction detection and crossing logic

fish_counter.py                     # Pipeline entry point and main orchestrator
```

### Quick Start

1. **Setup Environment**:
  - This example is for Windows.
    ```bash
    # Navigate to your project directory
    cd "path\to\your\repo"

    # Create your virtual environment (if you don't have one already)
    python -m venv .venv

    # Activate your virtual environment
    .\.venv\Scripts\Activate.ps1

    # Install dependencies
    pip install -r requirements.txt
    ```

2. **Configure Your Pipeline**:
  - Use the provided config file: `configs/wells_dam_test.json`
  - Or create your own config file based on the template
  - In either case, make sure to update paths for your model weights and input video, and also the location and date string:
    ```json
    {
      "model": {
        "model_path": "path\to\your\weights.pt"
      },
      "io": {
        "video_path": "path\to\your\video.mp4",
        "location": "location",
        "date_str": "YYYY-MM-DD"
      }
    }
    ```

3. **Run the Pipeline**:
  - By default, the `configs/wells_dam_test.json` will be used.
    ```bash
    python fish_counter.py
    ```
  - To override with your own config file, run:
    ```bash
    python fish_counter.py configs/your_config_file.json
    ```

4. **Outputs** (automatically organized in timestamped folders):
    ```
    output_20251002_143521/
    ├── manual_review/                            # QA clips for human review
    ├── annotated_video_20251002_143521.mp4       # Video with bounding boxes
    ├── fish_counts_20251002_143521.csv           # Detection & classification data
    └── fish_counts_20251002_143521_summary.txt   # Summary
    ```

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
