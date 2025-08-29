import numpy as np
from typing import List, Tuple
import cv2
from scipy import ndimage
from sklearn.cluster import DBSCAN

from .grid_map import GridMap
from ..utils.math_utils import distance_3d, check_point_in_bounds


class FrontierFinder:
    def __init__(self, grid_map: GridMap, cluster_tolerance: float = 1.0, min_frontier_size: int = 5):
        self.grid_map = grid_map
        self.cluster_tolerance = cluster_tolerance
        self.min_frontier_size = min_frontier_size
        
    def find_frontiers(self, robot_position: np.ndarray, search_radius: float = 20.0) -> List[np.ndarray]:
        """Find frontier points for exploration"""
        try:
            # Get occupancy grid in the search area
            occupancy_grid = self._get_local_occupancy_grid(robot_position, search_radius)
            
            if occupancy_grid is None:
                return []
            
            # Find frontier cells
            frontier_cells = self._detect_frontier_cells(occupancy_grid)
            
            if len(frontier_cells) == 0:
                return []
            
            # Cluster frontier cells
            frontier_clusters = self._cluster_frontiers(frontier_cells)
            
            # Convert clusters to world coordinates and filter
            frontiers = self._process_frontier_clusters(frontier_clusters, robot_position, search_radius)
            
            return frontiers
            
        except Exception as e:
            print(f"Error in frontier finding: {e}")
            return []
    
    def _get_local_occupancy_grid(self, center: np.ndarray, radius: float) -> np.ndarray:
        """Get local occupancy grid around the robot"""
        try:
            # Calculate grid bounds
            resolution = self.grid_map.resolution
            grid_radius = int(radius / resolution)
            
            center_grid = self.grid_map.world_to_grid(center)
            
            # Define grid region
            min_x = max(0, center_grid[0] - grid_radius)
            max_x = min(self.grid_map.size_x, center_grid[0] + grid_radius)
            min_y = max(0, center_grid[1] - grid_radius)
            max_y = min(self.grid_map.size_y, center_grid[1] + grid_radius)
            
            if min_x >= max_x or min_y >= max_y:
                return None
                
            # Extract local grid
            local_grid = self.grid_map.occupancy_grid[min_x:max_x, min_y:max_y].copy()
            
            return local_grid
            
        except Exception as e:
            print(f"Error getting local occupancy grid: {e}")
            return None
    
    def _detect_frontier_cells(self, occupancy_grid: np.ndarray) -> List[Tuple[int, int]]:
        """Detect frontier cells in the occupancy grid"""
        frontier_cells = []
        
        height, width = occupancy_grid.shape
        
        # Define 8-connected neighborhood
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                # Check if current cell is free space
                if occupancy_grid[i, j] == 0:  # Free space
                    # Check if any neighbor is unknown
                    is_frontier = False
                    for di, dj in neighbors:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if occupancy_grid[ni, nj] == -1:  # Unknown space
                                is_frontier = True
                                break
                    
                    if is_frontier:
                        frontier_cells.append((i, j))
        
        return frontier_cells
    
    def _cluster_frontiers(self, frontier_cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Cluster frontier cells using DBSCAN"""
        if len(frontier_cells) < self.min_frontier_size:
            return []
        
        # Convert to numpy array for clustering
        points = np.array(frontier_cells)
        
        # Use DBSCAN clustering
        eps = self.cluster_tolerance / self.grid_map.resolution  # Convert to grid units
        clustering = DBSCAN(eps=eps, min_samples=self.min_frontier_size).fit(points)
        
        clusters = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
                
            cluster_points = points[clustering.labels_ == label]
            if len(cluster_points) >= self.min_frontier_size:
                clusters.append([tuple(point) for point in cluster_points])
        
        return clusters
    
    def _process_frontier_clusters(self, clusters: List[List[Tuple[int, int]]], 
                                 robot_position: np.ndarray, search_radius: float) -> List[np.ndarray]:
        """Convert frontier clusters to world coordinates"""
        frontiers = []
        
        for cluster in clusters:
            # Calculate cluster centroid
            centroid_grid = np.mean(cluster, axis=0)
            
            # Convert to world coordinates - add z coordinate
            centroid_grid_3d = np.array([centroid_grid[0], centroid_grid[1], self.grid_map.size_z//2])
            centroid_world = self.grid_map.grid_to_world(centroid_grid_3d)
            
            # Check if within search radius
            if distance_3d(robot_position, centroid_world) <= search_radius:
                # Check if accessible (not too close to obstacles)
                if self._is_frontier_accessible(centroid_world):
                    frontiers.append(centroid_world)
        
        return frontiers
    
    def _is_frontier_accessible(self, frontier: np.ndarray) -> bool:
        """Check if a frontier is accessible (safe to approach)"""
        try:
            # Check minimum distance to obstacles
            min_clearance = 1.0  # meters
            
            # Sample points around the frontier
            for dx in [-min_clearance, 0, min_clearance]:
                for dy in [-min_clearance, 0, min_clearance]:
                    for dz in [-min_clearance/2, 0, min_clearance/2]:
                        check_point = frontier + np.array([dx, dy, dz])
                        
                        # Check if point is in bounds
                        if not check_point_in_bounds(check_point,
                                                   self.grid_map.bounds_min,
                                                   self.grid_map.bounds_max):
                            continue
                        
                        # Check occupancy
                        grid_coord = self.grid_map.world_to_grid(check_point)
                        if (0 <= grid_coord[0] < self.grid_map.size_x and 
                            0 <= grid_coord[1] < self.grid_map.size_y):
                            
                            if self.grid_map.occupancy_grid[grid_coord[0], grid_coord[1]] == 1:
                                return False  # Too close to obstacle
            
            return True
            
        except Exception as e:
            print(f"Error checking frontier accessibility: {e}")
            return False
    
    def visualize_frontiers(self, frontiers: List[np.ndarray]) -> np.ndarray:
        """Create visualization of frontiers on the occupancy grid"""
        try:
            vis_grid = self.grid_map.occupancy_grid.copy().astype(np.float32)
            
            # Normalize to 0-1 range
            vis_grid[vis_grid == -1] = 0.5  # Unknown as gray
            vis_grid[vis_grid == 1] = 0.0   # Occupied as black
            # Free space remains as 1.0 (white)
            
            # Mark frontiers in red
            for frontier in frontiers:
                grid_coord = self.grid_map.world_to_grid(frontier)
                if (0 <= grid_coord[0] < self.grid_map.size_x and 
                    0 <= grid_coord[1] < self.grid_map.size_y):
                    # Create a small circle around frontier
                    cv2.circle(vis_grid, (grid_coord[1], grid_coord[0]), 3, 1.0, -1)
            
            return vis_grid
            
        except Exception as e:
            print(f"Error visualizing frontiers: {e}")
            return self.grid_map.occupancy_grid.astype(np.float32)