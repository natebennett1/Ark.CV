"""
Species classification orchestration with context-aware decision making.

This module coordinates the classification workflow by combining model outputs,
confidence thresholds, and business rules to make final species determinations.
"""

from datetime import date
from typing import Optional, Tuple

from ..config.settings import ClassificationConfig
from .species_rules import SpeciesRules


class SpeciesClassifier:
    """
    Orchestrates species classification with configuration and context.
    
    This class manages the classification workflow by:
    - Holding location/date context
    - Applying confidence thresholds from configuration
    - Coordinating calls to stateless business rules
    - Making final classification decisions
    - Providing convenient access to contextual queries
    """
    
    def __init__(self, classification_config: ClassificationConfig, location: str, check_date: date):
        """
        Initialize the classifier with context.
        
        Args:
            classification_config: Configuration with thresholds and size rules
            location: Location name for species validation
            check_date: Date for seasonal validation
        """
        self.config = classification_config
        self.location = location
        self.check_date = check_date
    
    def classify_detection(self, raw_species: str, confidence: float) -> Tuple[str, str]:
        """
        Classify a detection into final species category.
        
        This is the main classification workflow that:
        1. Normalizes raw model output
        2. Checks confidence thresholds
        3. Validates against location/seasonal rules
        
        Args:
            raw_species: Raw species name from the model
            confidence: Detection confidence score (0.0 to 1.0)
            
        Returns:
            Tuple of (final_species, reasoning) where reasoning explains the classification
        """
        # Step 1: Normalize the raw model output
        normalized_species = SpeciesRules.normalize_legacy_output(raw_species)
        
        # Step 2: Apply confidence thresholds
        min_confidence = self._get_minimum_confidence(normalized_species)
        if confidence < min_confidence:
            return "Unknown", f"Confidence {confidence:.3f} below threshold {min_confidence:.3f}"
        
        # Step 3: Validate species label against location and seasonal rules
        is_valid, result_species = SpeciesRules.validate_species_label(
            normalized_species, 
            self.location, 
            self.check_date
        )
        
        if is_valid:
            return result_species, "Classification successful"

        return "Unknown", f"Species '{normalized_species}' not allowed at {self.location} on {self.check_date}"
    
    def _get_minimum_confidence(self, species: str) -> float:
        """
        Get the minimum confidence threshold for a species.
        
        Uses species-specific thresholds from config, falling back to the
        general unknown_threshold if no specific threshold is defined.
        
        Args:
            species: Species name
            
        Returns:
            Minimum confidence threshold (0.0 to 1.0)
        """
        return max(
            self.config.unknown_threshold,
            self.config.min_class_confidence.get(species, self.config.unknown_threshold)
        )

    def apply_adipose_refinement(self, detected_species: str, adipose_status: str) -> str:
        """
        Apply adipose fin refinement to a detected species.
        
        This adds or updates the adipose fin suffix for salmonids.
        Non-salmonids are returned unchanged.
        
        Args:
            detected_species: Detected species name
            adipose_status: Adipose fin status ("Present", "Absent", or ""Unknown")
            
        Returns:
            Species name with appropriate adipose suffix
        """
        return SpeciesRules.apply_adipose_suffix(detected_species, adipose_status)
    
    def classify_by_size(self, species: str, length_inches: float) -> Optional[str]:
        """
        Classify fish by size categories (adult, jack, etc.).
        
        Args:
            species: Species name
            length_inches: Fish length in inches
            
        Returns:
            Size classification or None if no size thresholds defined
        """
        thresholds = self.config.size_thresholds.get(species)
        if not thresholds:
            return None
        
        # Sort thresholds by size (largest first)
        sorted_categories = sorted(thresholds.items(), key=lambda x: x[1], reverse=True)
        
        for category, min_length in sorted_categories:
            if length_inches >= min_length:
                return category
        
        # If no category matches, return the smallest category
        return sorted_categories[-1][0] if sorted_categories else None