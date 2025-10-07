"""
Pipeline configuration settings.

This module centralizes all configuration parameters for the fish counting pipeline,
making it easy to modify behavior without changing core logic.
"""

import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """Configuration for YOLO models."""
    # Primary detection model
    model_path: str = ""
    confidence_threshold: float = 0.10
    iou_threshold: float = 0.25
    max_detections: int = 200
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # Optional adipose fin detection model
    adipose_model_path: str = ""
    adipose_min_confidence: float = 0.50
    adipose_expand_ratio: float = 0.20


@dataclass
class BotsortConfig:
    """Configuration for BoT-SORT tracker."""
    tracker_type: str = "botsort"
    track_buffer: int = 30
    match_thresh: float = 0.8
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.25
    gmc_method: str = "sparseOptFlow"
    new_track_thresh: float = 0.1
    track_high_thresh: float = 0.1
    track_low_thresh: float = 0.05
    max_time_lost: int = 30


@dataclass
class ClassificationConfig:
    """Configuration for species classification."""
    unknown_threshold: float = 0.25
    min_class_confidence: Dict[str, float] = field(default_factory=lambda: {
        "BullTrout": 0.80
    })
    size_thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Chinook": {"adult": 22, "jack": 12, "mini_jack": 0},
        "Coho": {"adult": 18, "jack": 12},
        "Sockeye": {"adult": 20},
        "Steelhead": {"adult": 24},
        "BullTrout": {"adult": 12}
    })


@dataclass
class CountingConfig:
    """Configuration for fish counting and direction detection."""
    pixels_per_inch: float = 25.253
    center_line_position: float = 0.50  # Fraction of frame width
    trail_max_length: int = 30
    stability_window: int = 3  # Frames for majority vote
    adipose_window: int = 3
    count_cooldown_frames: int = 0  # Frames before same track can count again


@dataclass
class ManualReviewConfig:
    """Configuration for Manual Review data collection."""
    output_dir: str = "./manual_review"
    lowconf_threshold: float = 0.80
    count_review_threshold: float = 0.80
    expand_ratio: float = 0.15
    track_gap_frames: int = 45
    enable_deduplication: bool = True


@dataclass
class VideoConfig:
    """Configuration for video processing."""
    enable_display: bool = False
    save_output_video: bool = True
    output_video_codec: str = "mp4v"
    frame_skip: int = 0  # Process every nth frame (0 = no skipping)


@dataclass
class IOConfig:
    """Configuration for input/output paths and formats."""
    # Input paths
    video_path: str = ""
    
    # Output paths (will be auto-generated if empty)
    csv_output_path: str = ""
    video_output_path: str = ""
    
    # Metadata
    location: str = ""
    date_str: str = ""  # YYYY-MM-DD format
    
    # Cloud storage (for future AWS deployment)
    s3_input_bucket: Optional[str] = None
    s3_output_bucket: Optional[str] = None
    s3_input_key: Optional[str] = None
    use_cloud_storage: bool = False


@dataclass 
class PipelineConfig:
    """Main configuration class that combines all sub-configurations."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    botsort: BotsortConfig = field(default_factory=BotsortConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    counting: CountingConfig = field(default_factory=CountingConfig)
    hitl: ManualReviewConfig = field(default_factory=ManualReviewConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    io: IOConfig = field(default_factory=IOConfig)
    
    # Computed properties
    _file_timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def __post_init__(self):
        """Auto-generate output paths if not provided."""
        # Create timestamped output directory
        output_dir = f"output_{self._file_timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.io.csv_output_path:
            self.io.csv_output_path = os.path.join(output_dir, f"fish_counts_{self._file_timestamp}.csv")
        
        if not self.io.video_output_path:
            self.io.video_output_path = os.path.join(output_dir, f"annotated_video_{self._file_timestamp}.mp4")
        
        # Update Manual Review output directory to be inside the timestamped folder
        if self.hitl.output_dir == "./manual_review":
            self.hitl.output_dir = os.path.join(output_dir, "manual_review")
            
        if not self.io.date_str:
            self.io.date_str = datetime.now().strftime("%Y-%m-%d")
    
    def validate(self) -> None:
        """Validate configuration settings."""
        errors = []
        
        # Check required paths
        if not self.model.model_path:
            errors.append("Model path is required")
        elif not os.path.isfile(self.model.model_path):
            errors.append(f"Model file not found: {self.model.model_path}")
            
        if not self.io.video_path:
            errors.append("Video path is required")
        elif not self.io.use_cloud_storage and not os.path.isfile(self.io.video_path):
            errors.append(f"Video file not found: {self.io.video_path}")
        
        # Validate date format
        if self.io.date_str:
            try:
                datetime.strptime(self.io.date_str, "%Y-%m-%d")
            except ValueError:
                errors.append(f"Invalid date format '{self.io.date_str}'. Use YYYY-MM-DD.")
        
        # Validate numeric ranges
        if not 0 <= self.counting.center_line_position <= 1:
            errors.append("Center line position must be between 0 and 1")
            
        if not 0 <= self.model.confidence_threshold <= 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"- {e}" for e in errors))
    
    @classmethod
    def from_legacy_constants(cls, **overrides) -> 'PipelineConfig':
        """Create config from the legacy constants in the original script."""
        config = cls()
        
        # Apply any overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Try to find nested attribute
                parts = key.split('.')
                if len(parts) == 2:
                    section, attr = parts
                    if hasattr(config, section):
                        setattr(getattr(config, section), attr, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'model': self.model.__dict__,
            'botsort': self.botsort.__dict__, 
            'classification': self.classification.__dict__,
            'counting': self.counting.__dict__,
            'hitl': self.hitl.__dict__,
            'video': self.video.__dict__,
            'io': self.io.__dict__
        }


def create_default_config() -> PipelineConfig:
    """Create a default configuration with sensible defaults."""
    return PipelineConfig()


def load_config_from_env() -> PipelineConfig:
    """Load configuration from environment variables."""
    config = create_default_config()
    
    # Model configuration
    if model_path := os.getenv('FISH_MODEL_PATH'):
        config.model.model_path = model_path
    if adipose_path := os.getenv('FISH_ADIPOSE_MODEL_PATH'):
        config.model.adipose_model_path = adipose_path
    
    # I/O configuration
    if video_path := os.getenv('FISH_VIDEO_PATH'):
        config.io.video_path = video_path
    if location := os.getenv('FISH_LOCATION'):
        config.io.location = location
    if date_str := os.getenv('FISH_DATE'):
        config.io.date_str = date_str
    
    # Cloud configuration
    if s3_input := os.getenv('FISH_S3_INPUT_BUCKET'):
        config.io.s3_input_bucket = s3_input
        config.io.use_cloud_storage = True
    if s3_output := os.getenv('FISH_S3_OUTPUT_BUCKET'):
        config.io.s3_output_bucket = s3_output
    
    return config