import numpy as np
from typing import Tuple, List, Optional
import cv2


class GridMap:
    def __init__(self, bounds_min: np.ndarray, bounds_max: np.ndarray, resolution: float = 0.1):
        self.bounds_min = bounds_min
        self.bounds_max = bounds_max  
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.size_x = int((bounds_max[0] - bounds_min[0]) / resolution)
        self.size_y = int((bounds_max[1] - bounds_min[1]) / resolution) 
        self.size_z = int((bounds_max[2] - bounds_min[2]) / resolution)
        
        # Initialize occupancy grid
        # -1: unknown, 0: free, 1: occupied
        self.occupancy_grid = np.full((self.size_x, self.size_y), -1, dtype=np.int8)
        
        # 3D voxel grid for height information
        self.voxel_grid = np.full((self.size_x, self.size_y, self.size_z), -1, dtype=np.int8)
        
    def world_to_grid(self, world_pos: np.ndarray) -> np.ndarray:
        """Convert world coordinates to grid coordinates"""
        grid_pos = ((world_pos - self.bounds_min) / self.resolution).astype(int)
        return np.clip(grid_pos, [0, 0, 0], [self.size_x-1, self.size_y-1, self.size_z-1])
    
    def grid_to_world(self, grid_pos: np.ndarray) -> np.ndarray:
        """Convert grid coordinates to world coordinates"""
        return grid_pos * self.resolution + self.bounds_min + self.resolution/2
    
    def is_occupied(self, world_pos: np.ndarray) -> bool:
        """Check if a world position is occupied"""
        grid_pos = self.world_to_grid(world_pos)
        if (0 <= grid_pos[0] < self.size_x and 
            0 <= grid_pos[1] < self.size_y):
            return self.occupancy_grid[grid_pos[0], grid_pos[1]] == 1
        return True  # Out of bounds considered occupied
    
    def is_free(self, world_pos: np.ndarray) -> bool:
        """Check if a world position is free space"""
        grid_pos = self.world_to_grid(world_pos)
        if (0 <= grid_pos[0] < self.size_x and 
            0 <= grid_pos[1] < self.size_y):
            return self.occupancy_grid[grid_pos[0], grid_pos[1]] == 0
        return False
    
    def is_unknown(self, world_pos: np.ndarray) -> bool:
        """Check if a world position is unknown"""
        grid_pos = self.world_to_grid(world_pos)
        if (0 <= grid_pos[0] < self.size_x and 
            0 <= grid_pos[1] < self.size_y):
            return self.occupancy_grid[grid_pos[0], grid_pos[1]] == -1
        return False
    
    def update_with_lidar(self, robot_pos: np.ndarray, lidar_points: np.ndarray):
        """Update occupancy grid with LiDAR point cloud"""
        try:
            if len(lidar_points) == 0:
                return
            
            robot_grid = self.world_to_grid(robot_pos)
            
            # Mark free space between robot and obstacles
            for point in lidar_points:
                obstacle_world = robot_pos + point
                obstacle_grid = self.world_to_grid(obstacle_world)
                
                # Raycast from robot to obstacle
                self._raycast_update(robot_grid[:2], obstacle_grid[:2])
                
                # Mark obstacle
                if (0 <= obstacle_grid[0] < self.size_x and 
                    0 <= obstacle_grid[1] < self.size_y):
                    self.occupancy_grid[obstacle_grid[0], obstacle_grid[1]] = 1
                    
        except Exception as e:
            print(f"Error updating with LiDAR: {e}")
    
    def update_with_depth_camera(self, robot_pos: np.ndarray, depth_image: np.ndarray, 
                               camera_fov: float = 90.0, max_range: float = 10.0):
        """Update occupancy grid with depth camera data"""
        try:
            if depth_image is None or depth_image.size == 0:
                return
            
            height, width = depth_image.shape
            robot_grid = self.world_to_grid(robot_pos)
            
            # Camera parameters
            fx = width / (2 * np.tan(np.radians(camera_fov / 2)))
            fy = fx  # Assume square pixels
            cx = width / 2
            cy = height / 2
            
            # Sample depth image
            step = 4  # Sample every 4th pixel for efficiency
            for v in range(0, height, step):
                for u in range(0, width, step):
                    depth = depth_image[v, u]
                    
                    if depth <= 0 or depth > max_range:
                        continue
                    
                    # Convert pixel to 3D point in camera frame
                    x_cam = (u - cx) * depth / fx
                    y_cam = (v - cy) * depth / fy
                    z_cam = depth
                    
                    # Transform to world frame (assuming camera faces forward)
                    point_world = robot_pos + np.array([z_cam, -x_cam, -y_cam])
                    point_grid = self.world_to_grid(point_world)
                    
                    # Raycast and update
                    self._raycast_update(robot_grid[:2], point_grid[:2])
                    
                    # Mark obstacle if within reasonable range
                    if depth < max_range * 0.95:  # Not at max range
                        if (0 <= point_grid[0] < self.size_x and 
                            0 <= point_grid[1] < self.size_y):
                            self.occupancy_grid[point_grid[0], point_grid[1]] = 1
                            
        except Exception as e:
            print(f"Error updating with depth camera: {e}")
    
    def _raycast_update(self, start_grid: np.ndarray, end_grid: np.ndarray):
        """Update free space along a ray using Bresenham's line algorithm"""
        try:
            x0, y0 = start_grid[0], start_grid[1]
            x1, y1 = end_grid[0], end_grid[1]
            
            # Bresenham's line algorithm
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            
            x_step = 1 if x0 < x1 else -1
            y_step = 1 if y0 < y1 else -1
            
            error = dx - dy
            
            x, y = x0, y0
            
            while True:
                # Mark as free space if within bounds
                if (0 <= x < self.size_x and 0 <= y < self.size_y):
                    if self.occupancy_grid[x, y] == -1:  # Only update unknown cells
                        self.occupancy_grid[x, y] = 0  # Free space
                
                if x == x1 and y == y1:
                    break
                    
                error2 = 2 * error
                
                if error2 > -dy:
                    error -= dy
                    x += x_step
                    
                if error2 < dx:
                    error += dx
                    y += y_step
                    
        except Exception as e:
            print(f"Error in raycast update: {e}")
    
    def get_explored_volume(self) -> float:
        """Get the volume of explored space"""
        free_cells = np.sum(self.occupancy_grid == 0)
        occupied_cells = np.sum(self.occupancy_grid == 1)
        explored_cells = free_cells + occupied_cells
        
        cell_volume = self.resolution ** 3
        return explored_cells * cell_volume
    
    def get_total_volume(self) -> float:
        """Get the total volume of the exploration space"""
        total_cells = self.size_x * self.size_y * self.size_z
        cell_volume = self.resolution ** 3
        return total_cells * cell_volume
    
    def get_occupancy_probability(self, world_pos: np.ndarray) -> float:
        """Get occupancy probability at world position"""
        grid_pos = self.world_to_grid(world_pos)
        
        if (0 <= grid_pos[0] < self.size_x and 
            0 <= grid_pos[1] < self.size_y):
            value = self.occupancy_grid[grid_pos[0], grid_pos[1]]
            if value == 1:
                return 1.0  # Occupied
            elif value == 0:
                return 0.0  # Free
            else:
                return 0.5  # Unknown
        
        return 1.0  # Out of bounds considered occupied
    
    def save_map(self, filename: str):
        """Save occupancy grid to file"""
        try:
            np.savez(filename, 
                    occupancy_grid=self.occupancy_grid,
                    bounds_min=self.bounds_min,
                    bounds_max=self.bounds_max,
                    resolution=self.resolution)
        except Exception as e:
            print(f"Error saving map: {e}")
    
    def load_map(self, filename: str) -> bool:
        """Load occupancy grid from file"""
        try:
            data = np.load(filename)
            self.occupancy_grid = data['occupancy_grid']
            self.bounds_min = data['bounds_min']
            self.bounds_max = data['bounds_max']
            self.resolution = data['resolution']
            
            # Recalculate grid dimensions
            self.size_x, self.size_y = self.occupancy_grid.shape
            self.size_z = int((self.bounds_max[2] - self.bounds_min[2]) / self.resolution)
            
            return True
        except Exception as e:
            print(f"Error loading map: {e}")
            return False