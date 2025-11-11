"""
Species classification rules and seasonal logic.

This module contains stateless business rules and data for fish species validation.
"""

from datetime import date
from typing import Dict, List, Tuple, Set


class SpeciesRules:
    """
    Stateless repository of species classification rules and constants.
    
    This class contains:
    - Species lists by location
    - Seasonal date ranges
    - String normalization utilities
    - Pure business logic functions
    """

    # Known salmonid species that can have adipose fin classifications
    SALMONIDS: Set[str] = {"Chinook", "Coho", "Sockeye", "Steelhead"}

    # Valid species for each location
    SPECIES_BY_LOCATION: Dict[str, List[str]] = {
        "Wells Dam": [
            "Chinook",
            "Coho",
            "Sockeye",
            "Steelhead",
            "Lamprey",
            "Pike",
            "BullTrout",
            "Suckerfish",
            "ResidentFish",
            "Pink"
        ],
    }

    # Seasonal date ranges by location and species
    SEASONAL_RANGES: Dict[str, Dict[str, List[Tuple[int, int, int, int]]]] = {
        "Wells Dam": {
            "chinook_spring": [(5, 1, 6, 28)],      # May 1 - June 28
            "chinook_summer": [(6, 29, 8, 28)],     # June 29 - Aug 28
            "chinook_fall": [(8, 29, 11, 15)],      # Aug 29 - Nov 15
            "sockeye_run": [(6, 1, 9, 30)],         # June 1 - Sept 30
            "steelhead_run": [(3, 1, 5, 31), (9, 1, 11, 30)],  # March-May, Sept-Nov
            "lamprey_run": [(6, 1, 9, 30)],         # June 1 - Sept 30
        }
    }

    @staticmethod
    def get_seasonal_ranges(year: int, location: str) -> Dict[str, List[Tuple[date, date]]]:
        """
        Get seasonal ranges for a specific year and location.
        
        Args:
            year: The year to get ranges for
            location: The location name
            
        Returns:
            Dictionary mapping season names to list of (start_date, end_date) tuples
        """
        location_ranges = SpeciesRules.SEASONAL_RANGES.get(location, {})
        result = {}
        
        for season_name, ranges in location_ranges.items():
            result[season_name] = [
                (date(year, start_month, start_day), date(year, end_month, end_day))
                for start_month, start_day, end_month, end_day in ranges
            ]
        
        return result
    
    @staticmethod
    def normalize_legacy_output(label: str) -> str:
        """
        Normalize legacy model output to standard format.
        
        Converts old naming conventions (e.g., "Pike Minnow", "Bull Trout")
        to current standards (e.g., "Pike", "BullTrout").
        
        Args:
            label: Raw species label from model
            
        Returns:
            Normalized species label
        """
        s = label.strip()
        
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
    
    @staticmethod
    def validate_species_label(species: str, location: str, check_date: date) -> Tuple[bool, str]:
        """
        Validate if a species label is allowed at a location and date.
        
        Args:
            species: Normalized species label
            location: Location name
            check_date: Date to check
            
        Returns:
            Tuple of (is_valid, reason_or_alternative)
            - If valid: (True, species_label)
            - If invalid: (False, best_alternative_or_"Unknown")
        """
        allowed = SpeciesRules.get_allowed_species(location)
        
        # Check if exact label is allowed and in season
        if species in allowed and SpeciesRules.is_in_season(species, location, check_date):
            return True, species
        
        return False, "Unknown"
    
    @staticmethod
    def is_in_season(species: str, location: str, check_date: date) -> bool:
        """
        Check if a species is in season at the given location and date.
        
        Args:
            species: Species name
            location: Location name
            check_date: Date to check
            
        Returns:
            True if the species is in season
        """
        seasons = SpeciesRules.get_seasonal_ranges(check_date.year, location)
        
        if species == "Chinook":
            for season_key in ["chinook_spring", "chinook_summer", "chinook_fall"]:
                for start, end in seasons.get(season_key, []):
                    if start <= check_date <= end:
                        return True
            return False
        
        if species == "Coho":
            if location == "Wells Dam":
                return SpeciesRules.is_coho_allowed_wells_dam(check_date)
            # Default Coho season
            return date(check_date.year, 8, 1) <= check_date <= date(check_date.year, 11, 30)
        
        if species == "Sockeye":
            return any(start <= check_date <= end 
                      for start, end in seasons.get("sockeye_run", []))
        
        if species == "Steelhead":
            return any(start <= check_date <= end 
                      for start, end in seasons.get("steelhead_run", []))
        
        if species == "Lamprey":
            return any(start <= check_date <= end 
                      for start, end in seasons.get("lamprey_run", []))
        
        # Default: species is always in season
        return True
    
    @staticmethod
    def is_coho_allowed_wells_dam(check_date: date) -> bool:
        """
        Check if Coho is allowed at Wells Dam on the given date.
        
        Special rule: Coho not counted before Sept 10 at Wells Dam.
        
        Args:
            check_date: Date to check
            
        Returns:
            True if Coho counting is allowed on this date
        """
        return check_date >= date(check_date.year, 9, 10) and check_date <= date(check_date.year, 11, 30)
    
    @staticmethod
    def get_allowed_species(location: str) -> Set[str]:
        """
        Get the set of allowed species for a location.
        
        Args:
            location: Location name
            
        Returns:
            Set of valid species labels for this location
        """
        return set(SpeciesRules.SPECIES_BY_LOCATION.get(location, []))
    
    @staticmethod
    def adipose_tag_from_status(status: str) -> str:
        """
        Convert adipose fin status word to tag suffix.
        
        Args:
            status: Status word ("Present", "Absent", or "Unknown")
            
        Returns:
            Tag suffix: "AP" (present), "AA" (absent), or "U" (unknown)
        """
        return {"Present": "AP", "Absent": "AA", "Unknown": "U"}.get(status, "U")
    
    @staticmethod
    def apply_adipose_suffix(species: str, adipose_status: str) -> str:
        """
        Apply adipose fin suffix to a species name.
        
        Only applies to salmonids. Non-salmonids are returned unchanged.
        
        Args:
            species: Species name (possibly with existing suffix)
            adipose_status: Adipose status ("Present", "Absent", or "Unknown")
            
        Returns:
            Species name with appropriate adipose suffix
        """
        if species not in SpeciesRules.SALMONIDS:
            return species

        tag = SpeciesRules.adipose_tag_from_status(adipose_status)
        return f"{species}_{tag}"