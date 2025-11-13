"""
Configuration loader for the fish counting pipeline.
"""

import os
import logging
import json
import tempfile
import boto3
from . import PipelineConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    
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

        config.__post_init__()
        
        return config
    
    @staticmethod
    def load_config_from_sqs_message(sqs_message: str) -> PipelineConfig:
        """
        Load configuration from an SQS message for cloud deployment.
        
        Args:
            sqs_message: JSON string containing the SQS message body
            
        Expected SQS message format:
        {
            "video_s3_bucket": "bucket-name",
            "video_s3_key": "path/to/video.mp4",
            "location": "wells",
            "ladder": "east",
            "date_str": "2025-08-31",
            "time_str": "0800"
        }
        
        Returns:
            Configured PipelineConfig object
        """
        # Parse SQS message
        message_body = json.loads(sqs_message)
        
        video_s3_bucket = message_body['video_s3_bucket']
        video_s3_key = message_body['video_s3_key']
        location = message_body['location']
        ladder = message_body['ladder']
        date_str = message_body['date_str']
        time_str = message_body['time_str']
        
        local_video_path = ConfigLoader._download_from_s3(
            video_s3_bucket, 
            video_s3_key
        )
        
        config = PipelineConfig()
        
        # Set model paths to models in Docker image
        config.model.model_path = '/app/weights/species-weights.pt'
        config.model.adipose_model_path = '/app/weights/adipose-weights.pt'

        #TODO: Set CountingConfig upstream direction based on location and ladder

        # Set video config
        config.video.save_output_video = False
        
        # Set I/O config
        config.io.video_path = local_video_path
        config.io.location = location
        config.io.ladder = ladder
        config.io.date_str = date_str
        config.io.time_str = time_str
        
        config.__post_init__()
        
        return config
    
    @staticmethod
    def _download_from_s3(bucket: str, key: str) -> str:
        """
        Download file from S3 to a local temporary path.
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            
        Returns:
            Local file path to the downloaded file
        """
        s3_client = boto3.client('s3')
        
        # Create temp file with proper extension
        filename = os.path.basename(key)
        local_path = os.path.join(tempfile.gettempdir(), filename)
        
        logger.info("Downloading s3://%s/%s to %s...", bucket, key, local_path)
        
        try:
            s3_client.download_file(bucket, key, local_path)
            
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info("Downloaded %.2f MB", file_size_mb)
            
            return local_path
            
        except Exception as e:
            logger.exception("Failed to download from S3: s3://%s/%s", bucket, key)
            raise