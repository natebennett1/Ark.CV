"""
Cloud-based output handler for fish counting pipeline.

This implementation accumulates species counts and writes them to DynamoDB
at the end of processing.
"""

import boto3
from typing import Dict
from datetime import datetime
from collections import defaultdict

from .output_handler import OutputHandler


class CloudOutputHandler(OutputHandler):
    """
    Handles output to DynamoDB for cloud deployment.
    
    Only tracks aggregated species counts, then writes to DynamoDB on finalize.
    """
    
    def __init__(self, location: str, ladder: str, date_str: str, time_str: str):
        self.location = location
        self.ladder = ladder
        self.date_str = date_str
        self.time_str = time_str
        
        # Only track counts - no individual records
        self.species_counts = defaultdict(lambda: {"Upstream": 0, "Downstream": 0})
        self.processing_start_time = datetime.now()
    
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
    
    def write_final_counts(self) -> bool:
        """
        Atomically update aggregated counts in DynamoDB.
        
        This implementation uses UpdateItem + if_not_exists to prevent lost updates
        when multiple writers run concurrently.
        """
        try:
            pk = f"{self.location}#{self.ladder}#{self.date_str}#{self.time_str}"
            print(f"Updating DynamoDB atomically with PK={pk}")

            client = boto3.client('dynamodb')
            timestamp = datetime.now().isoformat()

            # Ensure the base record exists (no-op if already exists)
            client.update_item(
                TableName='fish-counts',
                Key={'pk': {'S': pk}},
                UpdateExpression="""
                    SET location = :loc,
                        ladder = :lad,
                        date = :date,
                        hour = :hour,
                        last_updated = :ts
                """,
                ExpressionAttributeValues={
                    ':loc': {'S': self.location},
                    ':lad': {'S': self.ladder},
                    ':date': {'S': self.date_str},
                    ':hour': {'S': self.time_str},
                    ':ts': {'S': timestamp}
                }
            )

            # Atomically increment counts for each species
            for species, directions in self.species_counts.items():
                client.update_item(
                    TableName='fish-counts',
                    Key={'pk': {'S': pk}},
                    UpdateExpression="""
                        SET species_counts.#species.M.Upstream =
                            if_not_exists(species_counts.#species.M.Upstream, :zero) + :up_inc,
                            species_counts.#species.M.Downstream =
                            if_not_exists(species_counts.#species.M.Downstream, :zero) + :down_inc,
                            last_updated = :ts
                    """,
                    ExpressionAttributeNames={
                        '#species': species
                    },
                    ExpressionAttributeValues={
                        ':zero': {'N': '0'},
                        ':up_inc': {'N': str(directions['Upstream'])},
                        ':down_inc': {'N': str(directions['Downstream'])},
                        ':ts': {'S': timestamp}
                    }
                )

            print("✔ DynamoDB counts updated successfully (atomic)")
            return True

        except Exception as e:
            print(f"✖ Error updating DynamoDB: {e}")
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