"""
Species classification rules and seasonal logic.

This module contains all the business logic for determining valid fish species
based on location, season, and other domain-specific rules.
"""

from datetime import date
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass


@dataclass 
class SpeciesRules:
    """Encapsulates all species classification business logic."""
    
    # Species allowed by location
    SPECIES_BY_LOCATION: Dict[str, List[str]] = None
    
    # Seasonal ranges by location
    SEASONAL_RANGES: Dict[str, Dict[str, List[Tuple[date, date]]]] = None
    
    def __post_init__(self):
        if self.SPECIES_BY_LOCATION is None:
            self.SPECIES_BY_LOCATION = {
                "Wells Dam": [
                    "Chinook_AA", "Chinook_AP", "Chinook_U",
                    "Coho_AA", "Coho_AP", "Coho_U", 
                    "Sockeye_AA", "Sockeye_AP", "Sockeye_U",
                    "Steelhead_AA", "Steelhead_AP", "Steelhead_U",
                    "Lamprey", "Pike", "BullTrout", "Suckerfish", 
                    "ResidentFish", "Pink"
                ],
            }
    
    def get_seasonal_ranges(self, input_date: date) -> Dict[str, Dict[str, List[Tuple[date, date]]]]:
        """Get seasonal ranges for the given year."""
        if self.SEASONAL_RANGES is None:
            year = input_date.year
            self.SEASONAL_RANGES = {
                "Wells Dam": {
                    "chinook_spring": [(date(year, 5, 1), date(year, 6, 28))],
                    "chinook_summer": [(date(year, 6, 29), date(year, 8, 28))],
                    "chinook_fall": [(date(year, 8, 29), date(year, 11, 15))],
                    "sockeye_run": [(date(year, 6, 1), date(year, 9, 30))],
                    "steelhead_run": [
                        (date(year, 3, 1), date(year, 5, 31)),
                        (date(year, 9, 1), date(year, 11, 30))
                    ],
                    "lamprey_run": [(date(year, 6, 1), date(year, 9, 30))],
                }
            }
        return self.SEASONAL_RANGES
    
    def extract_base_species(self, label: str) -> str:
        """Extract the base species name from a label."""
        if "_" in label:
            return label.split("_", 1)[0]
        
        s = label.strip()
        
        # Handle legacy naming
        if s == "Pike Minnow": 
            return "Pike"
        if s == "Bull Trout":  
            return "BullTrout"
        if s == "Resident Fish - sp.": 
            return "ResidentFish"
        if s == "Pink Salmon": 
            return "Pink"
        
        # Check for known species in the label
        known_species = [
            "Chinook", "Coho", "Sockeye", "Steelhead", 
            "Pike", "BullTrout", "Pink", "Lamprey", 
            "Suckerfish", "ResidentFish"
        ]
        
        for species in known_species:
            if species in s:
                return species
        
        return s
    
    def normalize_legacy_output(self, label: str) -> str:
        """Normalize legacy model output to standard format."""
        s = label.strip()
        
        def make_adipose_label(base: str, status_str: str) -> str:
            if 'Present' in status_str:
                return f"{base}_AP"
            elif 'Absent' in status_str:
                return f"{base}_AA"
            else:
                return f"{base}_U"
        
        # Handle adipose-specific labels
        if "Chinook" in s and "Adipose" in s:
            return make_adipose_label("Chinook", s)
        if "Sockeye" in s and "Adipose" in s:
            return make_adipose_label("Sockeye", s)
        if "Steelhead" in s and "Adipose" in s:
            return make_adipose_label("Steelhead", s)
        if "Coho" in s and "Adipose" in s:
            return make_adipose_label("Coho", s)
        
        # Handle legacy names
        if s == "Pike Minnow":
            return "Pike"
        if s == "Bull Trout":
            return "BullTrout"
        if s == "Resident Fish - sp.":
            return "ResidentFish"
        if s == "Pink Salmon":
            return "Pink"
        
        return s
    
    def wells_coho_allowed(self, d: date) -> bool:
        """Check if Coho is allowed at Wells Dam on the given date."""
        # Coho not counted before Sept 10 at Wells
        return d >= date(d.year, 9, 10) and d <= date(d.year, 11, 30)
    
    def is_in_season(self, full_label: str, location: str, check_date: date) -> bool:
        """Check if a species is in season at the given location and date."""
        base = self.extract_base_species(full_label)
        seasons = self.get_seasonal_ranges(check_date).get(location, {})
        
        if base == "Chinook":
            for season_key in ["chinook_spring", "chinook_summer", "chinook_fall"]:
                for start, end in seasons.get(season_key, []):
                    if start <= check_date <= end:
                        return True
            return False
        
        if base == "Coho":
            if location == "Wells Dam":
                return self.wells_coho_allowed(check_date)
            # Default Coho season
            return date(check_date.year, 8, 1) <= check_date <= date(check_date.year, 11, 30)
        
        if base == "Sockeye":
            return any(start <= check_date <= end 
                      for start, end in seasons.get("sockeye_run", []))
        
        if base == "Steelhead":
            return any(start <= check_date <= end 
                      for start, end in seasons.get("steelhead_run", []))
        
        if base == "Lamprey":
            return any(start <= check_date <= end 
                      for start, end in seasons.get("lamprey_run", []))
        
        # Default: species is always in season
        return True
    
    def adipose_tag_from_words(self, status: str) -> str:
        """Convert adipose status word to tag."""
        return {"Present": "AP", "Absent": "AA"}.get(status, "U")
    
    def classify_by_size_and_adipose(self, base_species: str, adipose_status: str = "Unknown") -> str:
        """Classify species including adipose status for salmonids."""
        salmonids = {"Chinook", "Sockeye", "Steelhead", "Coho"}
        
        if base_species in salmonids:
            tag = self.adipose_tag_from_words(adipose_status)
            return f"{base_species}_{tag}"
        
        return base_species
    
    def get_allowed_species(self, location: str) -> Set[str]:
        """Get the set of allowed species for a location."""
        return set(self.SPECIES_BY_LOCATION.get(location, []))
    
    def find_fallback_species(self, base_species: str, location: str, check_date: date) -> str:
        """Find a fallback species label when the primary classification fails."""
        allowed = self.get_allowed_species(location)
        
        # Try to find an allowed species with the same base that's in season
        for expected in allowed:
            if (self.extract_base_species(expected) == base_species and 
                self.is_in_season(expected, location, check_date)):
                return expected
        
        return "Unknown"
    
    def validate_species_classification(self, species_label: str, location: str, 
                                      check_date: date, confidence: float,
                                      min_confidence: float = 0.45) -> str:
        """
        Validate and potentially correct a species classification.
        
        Returns the final species label after applying all business rules.
        """
        # First pass: confidence check
        if confidence < min_confidence:
            return "Unknown"
        
        # Normalize the label 
        normalized = self.normalize_legacy_output(species_label)
        base = self.extract_base_species(normalized)
        
        # Check if species is allowed at this location
        allowed = self.get_allowed_species(location)
        
        if normalized in allowed and self.is_in_season(normalized, location, check_date):
            return normalized
        
        # Try to find a fallback
        fallback = self.find_fallback_species(base, location, check_date)
        return fallback if fallback != "Unknown" else "Unknown"


def create_default_species_rules() -> SpeciesRules:
    """Create default species rules."""
    return SpeciesRules()