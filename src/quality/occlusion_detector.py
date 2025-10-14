"""
Occlusion detection for fish tracking quality assurance.

This module detects potential occlusions between fish based on bounding box proximity,
helping identify moments that may need manual verification.

Uses graph-based connected components to group all occluding fish into single events,
preventing redundant event creation when multiple fish are simultaneously occluding.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict

from .quality_event import QualityEvent, EventType


class OcclusionDetector:
    """
    Detects potential occlusions between fish based on bounding box proximity.
    
    An occlusion is detected when multiple fish bounding boxes are close to each other,
    measured by a combination of Intersection over Union (IoU) and centroid distance.
    """
    
    def __init__(self, proximity_threshold: float = 0.3, 
                 iou_weight: float = 0.8, distance_weight: float = 0.2):
        """
        Initialize occlusion detector.
        
        Args:
            proximity_threshold: Minimum combined score to trigger occlusion
            iou_weight: Weight for IoU in proximity calculation
            distance_weight: Weight for distance in proximity calculation
        """
        self.proximity_threshold = proximity_threshold
        self.iou_weight = iou_weight
        self.distance_weight = distance_weight
        
        # Track individual fish currently involved in active occlusion clips
        # This prevents creating overlapping occlusion events when group composition changes
        self.active_occlusion_tracks: Set[int] = set()
    
    def detect_occlusions(self, detections: List[Tuple[int, Tuple[int, int, int, int]]], 
                         center_line: Optional[int], frame_idx: int,
                         timestamp_sec: float) -> List[QualityEvent]:
        """
        Detect occlusions in the current frame using connected components.
        
        Builds a graph where fish are nodes and edges represent occlusions,
        then finds connected components to group all mutually-occluding fish
        into single events.
        
        Args:
            detections: List of (track_id, bbox) tuples
            center_line: Position of center line (occlusions near center line are prioritized)
            frame_idx: Current frame index
            timestamp_sec: Current timestamp in seconds
            
        Returns:
            List of QualityEvent objects for detected occlusion groups
        """
        if len(detections) < 2:
            return []
        
        # Build occlusion graph
        # Key: track_id, Value: set of track_ids it occludes with
        occlusion_graph = defaultdict(set)
        
        # Store proximity scores for edges (for reporting the max score)
        edge_scores: Dict[Tuple[int, int], float] = {}
        
        # Check all pairs of detections to build the graph
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                track1_id, bbox1 = detections[i]
                track2_id, bbox2 = detections[j]

                # Only capture occlusions on or before the center line
                if center_line is not None:
                    center1_x = (bbox1[0] + bbox1[2]) / 2
                    center2_x = (bbox2[0] + bbox2[2]) / 2
                    
                    # If BOTH fish are past (left of) the center line, skip
                    if center1_x < center_line and center2_x < center_line:
                        continue
                
                # Calculate proximity score
                iou = self._calculate_iou(bbox1, bbox2)
                distance_score = self._calculate_distance_score(bbox1, bbox2)
                proximity_score = (self.iou_weight * iou + 
                                 self.distance_weight * distance_score)
                
                # If this pair occludes, add edge to graph
                if proximity_score > self.proximity_threshold:
                    occlusion_graph[track1_id].add(track2_id)
                    occlusion_graph[track2_id].add(track1_id)
                    edge_key = (min(track1_id, track2_id), max(track1_id, track2_id))
                    edge_scores[edge_key] = proximity_score
        
        # Find connected components using DFS
        visited = set()
        occlusion_groups = []
        
        for track_id in occlusion_graph.keys():
            if track_id not in visited:
                # DFS to find all connected fish
                component = self._find_component(track_id, occlusion_graph, visited)
                if len(component) >= 2:  # Only care about groups of 2 or more
                    occlusion_groups.append(component)
        
        # Create events for each occlusion group
        events = []
        detection_map = {track_id: bbox for track_id, bbox in detections}
        
        for group in occlusion_groups:
            # Check if any fish in this group are already involved in an active occlusion clip
            # This prevents creating overlapping events when group composition changes
            if any(track_id in self.active_occlusion_tracks for track_id in group):
                continue
            
            # Mark these tracks as now being in an active occlusion clip
            self.active_occlusion_tracks.update(group)
            
            # Find the maximum proximity score within this group
            max_score = 0.0
            score_details = []
            for i, t1 in enumerate(group):
                for t2 in group[i+1:]:
                    edge_key = (min(t1, t2), max(t1, t2))
                    if edge_key in edge_scores:
                        score = edge_scores[edge_key]
                        max_score = max(max_score, score)
                        iou = self._calculate_iou(detection_map[t1], detection_map[t2])
                        score_details.append(f"{t1}-{t2}: {score:.3f}")
            
            # Gather detections for this group
            group_detections = [(tid, detection_map[tid]) for tid in group if tid in detection_map]
            
            # Create quality event for the entire group
            event = QualityEvent(
                event_type=EventType.OCCLUSION,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                proximity_score=max_score,
                track_ids=sorted(group),
                detections=group_detections,
                notes=f"Occlusion group of {len(group)} fish (max proximity: {max_score:.3f})"
            )
            
            events.append(event)
        
        return events
    
    def _find_component(self, start_track: int, graph: Dict[int, Set[int]], 
                       visited: Set[int]) -> List[int]:
        """
        Find all fish in the same connected component using DFS.
        
        Args:
            start_track: Starting track ID
            graph: Adjacency list representation of occlusion graph
            visited: Set of already-visited track IDs
            
        Returns:
            List of all track IDs in this connected component
        """
        component = []
        stack = [start_track]
        
        while stack:
            track_id = stack.pop()
            if track_id in visited:
                continue
            
            visited.add(track_id)
            component.append(track_id)
            
            # Add all neighbors to stack
            for neighbor in graph[track_id]:
                if neighbor not in visited:
                    stack.append(neighbor)
        
        return component
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                       box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_distance_score(self, box1: Tuple[int, int, int, int], 
                                  box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate normalized distance score between bounding box centers.
        Returns a score from 0 to 1, where 1 means very close, 0 means far apart.
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate centers
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        
        # Calculate Euclidean distance
        distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
        
        # Normalize by average box diagonal (rough measure of fish size)
        diag1 = ((x2_1 - x1_1)**2 + (y2_1 - y1_1)**2)**0.5
        diag2 = ((x2_2 - x1_2)**2 + (y2_2 - y1_2)**2)**0.5
        avg_diag = (diag1 + diag2) / 2
        
        # Convert to a score (closer = higher score)
        # Use exponential decay: score = exp(-distance/avg_diag)
        if avg_diag > 0:
            normalized_distance = distance / avg_diag
            return np.exp(-normalized_distance)
        return 0.0