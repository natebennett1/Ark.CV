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
  - `Pillow` (image loading/augmentation for crop training)
  - (`filterpy`, `scikit-image`, `lap`, `scipy`) ensure
  BoT-SORT has the motion and appearance modules it expects.

## Repository Structure
```
configs/
└── local.json                      # Configurations used if running locally

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

### Fish Counter Quick Start

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
    pip install -r docker/requirements-fishcounter.txt
    ```

2. **Configure Your Pipeline**:
  - Use the provided config file: `configs/local.json`
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
  - Set your video and model paths, along with any other parameters of your choosing, in `configs/local.json`.
    ```bash
    python .\fish_counter.\fish_counter.py
    ```

4. **Outputs** (automatically organized in timestamped folders):
    ```
    output_20251002_143521/
    ├── manual_review/                            # QA clips for human review
    ├── annotated_video_20251002_143521.mp4       # Video with bounding boxes
    ├── fish_counts_20251002_143521.csv           # Detection & classification data
    └── fish_counts_20251002_143521_summary.txt   # Summary
    ```

5. **Using Docker**
  - If you want to containerize the fish counter program and test it locally, do the following:
    - Copy over whatever video you want to test into the root of this repository (Ark.CV directory).
    - Edit Dockerfile.fishcounter to create an input directory and copy a test video into the docker container. It should look something like this:
      ```bash
      ...

      # Not actually needed for cloud deployment, but useful for local testing
      COPY configs/ ./configs/

      # Add these two lines
      RUN mkdir -p /app/input
      COPY your-test-video.mp4 /app/input/

      ...
      ```
    - Add the following lines in config_loader.py's load_config_from_file method, just before "config.\_\_post_init\_\_()" is called:
      ```bash
      config.io.video_path = '/app/input/your-test-video.mp4'
      config.model.model_path = '/app/weights/species-weights.pt'
      config.model.adipose_model_path = '/app/weights/adipose-weights.pt'
      ```
    - From the root directory of the repository, build the docker image:
      ```bash
      docker build -f .\docker\Dockerfile.fishcounter -t fishcounter-latest .
      ```
    - Run the docker container:
      ```bash
      docker run --name fishcounter-container fishcounter-latest
      ```
    - To get the output after the container has finished running, run this command and look in the temp_app directory for the output folder:
      ```bash
      docker cp fishcounter-container:/app ./temp_app
      ```
    - IMPORTANT: Be sure to undo all changes made to both the Dockerfile and config_loader.py after you are done testing.

### Preprocessor Quick Start

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
    pip install -r docker/requirements-preprocessing.txt
    ```

2. **Use Local Version**
  - Use the version of main in preprocessor.py that is meant for local execution, and replace the input video and output directory paths.

3. **Run the Preprocessor**
  - Run:
    ```bash
    python preprocessing/preprocessor.py
    ```

4. **Using Docker**
  - If you want to containerize preprocessor.py and test it locally, do the following:
    - Copy over whatever video you want to test into the root of this repository (Ark.CV directory).
    - Edit Dockerfile.preprocessing to create input and output directories and copy a test video into the docker container. It should look something like this:
      ```bash
      ...

      # Copy only the preprocessing module
      COPY preprocessing/preprocessor.py .

      # Add these two lines
      RUN mkdir -p /app/input /app/output
      COPY your-test-video.mp4 /app/input/

      ...
      ```
    - Ensure that the video path in preprocessor.py matches '/app/input/your-test-video.mp4'.
    - Ensure that the output path in preprocessor.py matches '/app/output'.
    - From the root directory of the repository, build the docker image:
      ```bash
      docker build -f .\docker\Dockerfile.preprocessing -t preprocessing-latest .
      ```
    - Run the docker container:
      ```bash
      docker run --name preprocessing-container preprocessing-latest
      ```
    - To get the output after the container has finished running:
      ```bash
      docker cp preprocessing-container:/app/output ./output
      ```
    - IMPORTANT: Be sure to undo all changes made to both the Dockerfile and preprocessor.py after you are done testing.

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

## Image-only cascade training workflow

The steps below focus solely on preparing datasets and training models from an
image export. They do **not** require running the video-counting pipeline.
Point the commands at the Roboflow dataset folder on your workstation (e.g.
`C:\Users\20ben\Downloads\Enhanced Wells Dam.v7i.coco-mmdetection`).

### 1. Prepare fish-only detection labels and crop metadata

`dataset_tools/prepare_datasets.py` converts your `[species]_[adipose]` labels into
a single-class COCO detector dataset and multi-task crop metadata. Point it at
the Roboflow export root and it will automatically pick up the `train/`,
`valid/`, and `test/` splits without touching the original images.

```powershell
python -m dataset_tools.prepare_datasets `
  --dataset-root "C:\Users\20ben\Downloads\Enhanced Wells Dam.v7i.coco-mmdetection" `
  --output-dir "C:\Users\20ben\Documents\ark_cv_cascade" `
  --pad-ratio 0.15 `
  --min-crop-size 8
```

Outputs of interest:

- `anns_det_fish_only_{train,val,test}.json` – single-class COCO annotations.
- `crops/` – padded crops per detection split (default 15% context padding).
- `crops_meta_{split}.csv` – CSV metadata for the multi-task crop trainer.

### 2. Train the RT-DETR fish detector in MMDetection

Copy `configs/rtdetr_r50_fish.py` into your MMDetection workspace and point the
dataset entries at the files produced above (use absolute paths on Windows). The config sets
`short-side` resizing (960–1280px), light photometric augmentations, and a
single-class head.

```powershell
python tools/train.py configs/rtdetr_r50_fish.py `
  --cfg-options `
    train_dataloader.dataset.data_root="C:/Users/20ben/Documents/ark_cv_cascade/" `
    val_dataloader.dataset.data_root="C:/Users/20ben/Documents/ark_cv_cascade/" `
    test_dataloader.dataset.data_root="C:/Users/20ben/Documents/ark_cv_cascade/" `
    train_dataloader.dataset.ann_file="C:/Users/20ben/Documents/ark_cv_cascade/anns_det_fish_only_train.json" `
    val_dataloader.dataset.ann_file="C:/Users/20ben/Documents/ark_cv_cascade/anns_det_fish_only_val.json" `
    test_dataloader.dataset.ann_file="C:/Users/20ben/Documents/ark_cv_cascade/anns_det_fish_only_val.json"
```

Adjust batch size/learning schedule per your GPU. When training completes, copy
the best checkpoint path into your pipeline configuration as the primary
detector.

### 3. Train the multi-task crop classifier

Use `dataset_tools/train_classifier.py` to train an EfficientNet-based classifier
with shared backbone and species/adipose heads. The script consumes the crop
metadata CSVs generated in step 1 and reads crops from the output directory you
supplied earlier.

```powershell
python -m dataset_tools.train_classifier `
  --train-csv "C:\Users\20ben\Documents\ark_cv_cascade\crops_meta_train.csv" `
  --val-csv "C:\Users\20ben\Documents\ark_cv_cascade\crops_meta_val.csv" `
  --root-dir "C:\Users\20ben\Documents\ark_cv_cascade" `
  --output-dir "C:\Users\20ben\Documents\ark_cv_classifier" `
  --model efficientnet_b2 `
  --image-size 256 `
  --epochs 45 `
  --adipose-weight 0.6
```

The training loop applies horizontal flips, ±5° rotations, color jitter, random
gamma, mild blur, and cutout to build robustness. After training, thresholds of
≈0.3 (species) and 0.45 (adipose) for the softmax maxima work well as
“unknown” triggers. Export the averaged logits per track in your runtime to
stabilize predictions across frames.

## Support

For questions or assistance, please document the environment, Ultralytics
version, GPU availability, and attach sample frames exhibiting the issue. This
context helps triage tracking or classification anomalies quickly.

## Keeping local experiments off `main`

If you would like to keep the cascade-training utilities (or any other
experiments) separate from your production `main` branch, work on a feature
branch and push that branch to GitHub instead of merging immediately. A typical
flow looks like this:

```powershell
# Make sure your local main matches GitHub
git checkout main
git pull origin main

# Create and switch to a dedicated branch for the training workflow
git checkout -b feature/local-cascade-training

# Do your work, commit as needed, then publish the branch
git push -u origin feature/local-cascade-training
```

As long as you keep working on `feature/local-cascade-training`, nothing touches
`main`. When you are ready to review the changes, open a pull request targeting
`main` (or another branch of your choosing). You can continue iterating on the
feature branch without merging until you are satisfied.
