"""
Species classification logic that combines detection results with business rules.

This module handles the complex logic for determining final species classifications
based on model predictions, confidence thresholds, seasonal rules, and adipose fin detection.
"""

from datetime import date
from typing import Optional, Tuple

from ..config.settings import ClassificationConfig
from .species_rules import SpeciesRules


class SpeciesClassifier:
    """
    Handles species classification with business rule application.
    
    This class combines raw model predictions with domain-specific business rules
    including seasonal availability, location-based species lists, and confidence thresholds.
    """
    
    def __init__(self, classification_config: ClassificationConfig, location: str, check_date: date):
        self.config = classification_config
        self.location = location
        self.check_date = check_date
        self.species_rules = SpeciesRules()
    
    def classify_detection(self, 
                          raw_species: str, 
                          confidence: float,
                          adipose_status: Optional[str] = None) -> Tuple[str, str]:
        """
        Classify a detection into final species category.
        
        Args:
            raw_species: Raw species name from the model
            confidence: Detection confidence score
            adipose_status: Optional adipose fin status ("Present", "Absent", "Unknown")
            
        Returns:
            Tuple of (final_species, reasoning) where reasoning explains the classification
        """
        # Step 1: Normalize the raw model output
        normalized_species = self.species_rules.normalize_legacy_output(raw_species)
        base_species = self.species_rules.extract_base_species(normalized_species)
        
        # Step 2: Apply confidence thresholds
        min_confidence = self._get_minimum_confidence(base_species)
        if confidence < min_confidence:
            return "Unknown", f"Confidence {confidence:.3f} below threshold {min_confidence:.3f}"
        
        # Step 3: Check location and seasonal rules
        final_species = self.species_rules.validate_species_classification(
            normalized_species, self.location, self.check_date, confidence, min_confidence
        )
        
        if final_species == "Unknown":
            return "Unknown", f"Species not allowed at {self.location} on {self.check_date}"
        
        # Step 4: Apply adipose fin refinement for salmonids
        if adipose_status and self._is_salmonid(base_species):
            refined_species = self._apply_adipose_refinement(final_species, adipose_status)
            if refined_species != final_species:
                return refined_species, f"Adipose fin status: {adipose_status}"
        
        return final_species, "Classification successful"
    
    def _get_minimum_confidence(self, base_species: str) -> float:
        """Get the minimum confidence threshold for a base species."""
        return max(
            self.config.unknown_threshold,
            self.config.min_class_confidence.get(base_species, self.config.unknown_threshold)
        )
    
    def _is_salmonid(self, base_species: str) -> bool:
        """Check if a species is a salmonid that can have adipose fin classification."""
        salmonids = {"Chinook", "Coho", "Sockeye", "Steelhead"}
        return base_species in salmonids
    
    def _apply_adipose_refinement(self, current_species: str, adipose_status: str) -> str:
        """
        Apply adipose fin refinement to a salmonid classification.
        
        Args:
            current_species: Current species classification
            adipose_status: Adipose fin status ("Present", "Absent", "Unknown")
            
        Returns:
            Refined species classification with adipose suffix
        """
        if "_" not in current_species:
            # No existing adipose classification, add it
            base = current_species
            suffix = self.species_rules.adipose_tag_from_words(adipose_status)
            return f"{base}_{suffix}"
        
        # Extract base and current suffix
        base, current_suffix = current_species.split("_", 1)
        
        # Only refine if current classification is uncertain (U suffix)
        if current_suffix == "U" and adipose_status in {"Present", "Absent"}:
            new_suffix = self.species_rules.adipose_tag_from_words(adipose_status)
            return f"{base}_{new_suffix}"
        
        # Keep current classification
        return current_species
    
    def get_allowed_species(self) -> set:
        """Get the set of species allowed at the current location."""
        return self.species_rules.get_allowed_species(self.location)
    
    def is_species_in_season(self, species: str) -> bool:
        """Check if a species is currently in season."""
        return self.species_rules.is_in_season(species, self.location, self.check_date)
    
    def classify_by_size(self, base_species: str, length_inches: float) -> Optional[str]:
        """
        Classify fish by size categories (adult, jack, etc.).
        
        Args:
            base_species: Base species name
            length_inches: Fish length in inches
            
        Returns:
            Size classification or None if no size thresholds defined
        """
        thresholds = self.config.size_thresholds.get(base_species)
        if not thresholds:
            return None
        
        # Sort thresholds by size (largest first)
        sorted_categories = sorted(thresholds.items(), key=lambda x: x[1], reverse=True)
        
        for category, min_length in sorted_categories:
            if length_inches >= min_length:
                return category
        
        # If no category matches, return the smallest category
        return sorted_categories[-1][0] if sorted_categories else None
    
    def get_classification_confidence(self, species: str, confidence: float) -> str:
        """
        Get a human-readable confidence assessment.
        
        Args:
            species: Classified species
            confidence: Detection confidence
            
        Returns:
            Confidence assessment string
        """
        if species == "Unknown":
            return "Low confidence"
        
        base_species = self.species_rules.extract_base_species(species)
        min_conf = self._get_minimum_confidence(base_species)
        
        if confidence >= 0.9:
            return "Very high confidence"
        elif confidence >= 0.8:
            return "High confidence"
        elif confidence >= min_conf + 0.1:
            return "Medium confidence" 
        else:
            return "Low confidence"