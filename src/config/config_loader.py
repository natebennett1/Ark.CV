"""
Configuration loader for the fish counting pipeline.
"""

import json
from . import PipelineConfig


class ConfigLoader:
    """Simple configuration loader class."""
    
    @staticmethod
    def load_config_from_file(config_path: str) -> PipelineConfig:
        """Load configuration from a JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        config = PipelineConfig()
        
        # Update configuration from file data
        for section, settings in config_data.items():
            if hasattr(config, section):
                section_obj = getattr(config, section)
                for key, value in settings.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        
        # Set up timestamped output paths
        config.__post_init__()
        
        return config