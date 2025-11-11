"""
Cloud-based output handler for fish counting pipeline.

This implementation accumulates species counts and writes them to DynamoDB
at the end of processing.
"""

import os
import boto3
from typing import Dict
from datetime import datetime
from collections import defaultdict

from ..config.settings import IOConfig
from .output_handler import OutputHandler


class CloudOutputHandler(OutputHandler):
    """
    Handles output to DynamoDB for cloud deployment.
    
    Only tracks aggregated species counts, then writes to DynamoDB on finalize.
    """
    
    def __init__(self, io_config: IOConfig):
        self.location = io_config.location
        self.ladder = io_config.ladder
        self.date_str = io_config.date_str
        self.time_str = io_config.time_str
        
        # Only track counts - no individual records
        self.species_counts = defaultdict(lambda: {"Upstream": 0, "Downstream": 0})
    
    def initialize(self) -> bool:
        """Initialize (nothing needed for cloud)."""
        print(f"✔ Cloud output handler initialized for {self.location}/{self.ladder} on {self.date_str}_{self.time_str}")
        return True
    
    def record_count(self, species: str, direction: str,
                    frame_number: int, track_id: int, confidence: float,
                    x_percent: float, y_percent: float, length_inches: float,
                    video_timestamp: str = None) -> bool:
        """
        Record a fish count by incrementing species counts.
        
        Cloud handler ignores individual record details and only tracks totals.
        """
        self.species_counts[species][direction] += 1
        return True
    
    def write_final_counts(self, qa_clips_s3_url: str = None) -> bool:
        """
        Atomically update aggregated counts in DynamoDB.
        """
        try:
            pk = f"{self.location}#{self.ladder}#{self.date_str}#{self.time_str}"
            print(f"Updating DynamoDB atomically with PK={pk}")

            client = boto3.client('dynamodb')
            timestamp = datetime.now().isoformat()

            # Base SET expressions (row metadata + ensure species_counts exists)
            set_clauses = [
                "location = :loc",
                "ladder = :lad",
                "date = :date",
                "hour = :hour",
                "last_updated = :ts",
                "species_counts = if_not_exists(species_counts, :empty_map)"
            ]

            expr_attr_names = {}
            expr_attr_values = {
                ':loc': {'S': self.location},
                ':lad': {'S': self.ladder},
                ':date': {'S': self.date_str},
                ':hour': {'S': self.time_str},
                ':ts': {'S': timestamp},
                ':empty_map': {'M': {}},
                ':zero': {'N': '0'},
            }

            if qa_clips_s3_url:
                set_clauses.append("qa_clips_s3_url = :qa_url")
                expr_attr_values[':qa_url'] = {'S': qa_clips_s3_url}

            # Add SET increment expressions for each species
            for idx, (species, directions) in enumerate(self.species_counts.items()):
                sp = f"#sp{idx}"
                expr_attr_names[sp] = species

                up = f":u{idx}"
                down = f":d{idx}"
                expr_attr_values[up] = {'N': str(directions['Upstream'])}
                expr_attr_values[down] = {'N': str(directions['Downstream'])}

                # SET species_counts.<species> = if_not_exists(...)
                set_clauses.append(
                    f"species_counts.{sp} = if_not_exists(species_counts.{sp}, :empty_map)"
                )

                # SET species_counts.<species>.Upstream = if_not_exists(...) + increment
                set_clauses.append(
                    f"species_counts.{sp}.Upstream = if_not_exists(species_counts.{sp}.Upstream, :zero) + {up}"
                )
                set_clauses.append(
                    f"species_counts.{sp}.Downstream = if_not_exists(species_counts.{sp}.Downstream, :zero) + {down}"
                )

            # Combine everything into a single UpdateExpression
            update_expression = "SET " + ", ".join(set_clauses)

            # Example final UpdateExpression:
            # SET location = :loc,
            #     ladder = :lad,
            #     date = :date,
            #     hour = :hour,
            #     last_updated = :ts,
            #     species_counts = if_not_exists(species_counts, :empty_map),
            #     species_counts.#sp0 = if_not_exists(species_counts.#sp0, :empty_map),
            #     species_counts.#sp0.Upstream = if_not_exists(species_counts.#sp0.Upstream, :zero) + :u0,
            #     species_counts.#sp0.Downstream = if_not_exists(species_counts.#sp0.Downstream, :zero) + :d0,
            #     species_counts.#sp1 = if_not_exists(species_counts.#sp1, :empty_map),
            #     species_counts.#sp1.Upstream = if_not_exists(species_counts.#sp1.Upstream, :zero) + :u1,
            #     species_counts.#sp1.Downstream = if_not_exists(species_counts.#sp1.Downstream, :zero) + :d1
            client.update_item(
                TableName='fish-counts',
                Key={'pk': {'S': pk}},
                UpdateExpression=update_expression,
                ExpressionAttributeNames=expr_attr_names,
                ExpressionAttributeValues=expr_attr_values
            )

            print("✔ DynamoDB counts updated successfully (atomic, single call, using + increments)")
            return True

        except Exception as e:
            print(f"✖ Error updating DynamoDB: {e}")
            import traceback
            traceback.print_exc()
            return False

    def upload_qa_clips_to_s3(self, clips_dir: str, s3_bucket: str, s3_prefix: str) -> str:
        """
        Upload all QA clips from local directory to S3.
        
        Args:
            clips_dir: Local directory containing QA clips
            s3_bucket: S3 bucket name
            s3_prefix: S3 prefix (folder path) for the clips
            
        Returns:
            True if successful, False otherwise
        """
        try:
            s3_client = boto3.client('s3')
            uploaded_count = 0
            
            # Upload all video files directly in clips_dir
            for filename in os.listdir(clips_dir):
                if filename.endswith('.mp4'):
                    local_path = os.path.join(clips_dir, filename)
                    
                    s3_key = f"{s3_prefix}/{filename}"
                    
                    print(f"Uploading QA clip: {filename} -> s3://{s3_bucket}/{s3_key}")
                    s3_client.upload_file(local_path, s3_bucket, s3_key)
                    uploaded_count += 1
            
            if uploaded_count > 0:
                print(f"Uploaded {uploaded_count} QA clip files to S3")
                
                # Return the S3 folder URL
                s3_url = f"s3://{s3_bucket}/{s3_prefix}/"
                return s3_url
            else:
                print("No QA clip files found to upload.")
                return None
            
        except Exception as e:
            print(f"✖ Error uploading QA clips to S3: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_species_counts(self) -> Dict[str, Dict[str, int]]:
        """Get current species count statistics."""
        return dict(self.species_counts)
    
    def finalize(self) -> bool:
        """
        Write aggregated counts to DynamoDB.
        
        PK will be: location#ladder#date#hour
        """
        self.print_final_summary()
        return True