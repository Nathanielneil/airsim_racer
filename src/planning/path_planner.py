import numpy as np
from typing import List, Optional, Tuple
import heapq
from dataclasses import dataclass
import math

from ..exploration.grid_map import GridMap
from ..utils.math_utils import distance_3d


@dataclass
class Node:
    position: np.ndarray
    g_cost: float = float('inf')
    h_cost: float = 0.0
    f_cost: float = float('inf')
    parent: Optional['Node'] = None
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class PathPlanner:
    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map
        self.inflation_radius = 0.5  # meters
        
    def plan_path(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """Plan a path from start to goal using A* algorithm"""
        try:
            # Check if start and goal are valid
            if self.grid_map.is_occupied(start) or self.grid_map.is_occupied(goal):
                return None
            
            # Convert to 2D planning (assume constant height)
            start_2d = start[:2]
            goal_2d = goal[:2]
            height = start[2]  # Use start height
            
            # A* search
            path_2d = self._astar_search(start_2d, goal_2d)
            
            if path_2d is None:
                return None
            
            # Convert back to 3D path
            path_3d = []
            for waypoint_2d in path_2d:
                waypoint_3d = np.array([waypoint_2d[0], waypoint_2d[1], height])
                path_3d.append(waypoint_3d)
            
            # Smooth path
            smoothed_path = self._smooth_path(path_3d)
            
            return smoothed_path
            
        except Exception as e:
            print(f"Error in path planning: {e}")
            return None
    
    def _astar_search(self, start: np.ndarray, goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """A* search algorithm for 2D grid"""
        start_grid = self.grid_map.world_to_grid(np.append(start, 0))[:2]
        goal_grid = self.grid_map.world_to_grid(np.append(goal, 0))[:2]
        
        # Initialize open and closed sets
        open_set = []
        closed_set = set()
        
        # Create start node
        start_node = Node(
            position=start_grid.astype(float),
            g_cost=0.0,
            h_cost=self._heuristic(start_grid, goal_grid)
        )
        start_node.f_cost = start_node.g_cost + start_node.h_cost
        
        heapq.heappush(open_set, start_node)
        
        # Node lookup
        all_nodes = {tuple(start_grid): start_node}
        
        # 8-connected neighbors
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        while open_set:
            current_node = heapq.heappop(open_set)
            current_pos = tuple(current_node.position.astype(int))
            
            if current_pos in closed_set:
                continue
                
            closed_set.add(current_pos)
            
            # Check if goal reached
            if self._is_goal_reached(current_node.position.astype(int), goal_grid):
                return self._reconstruct_path(current_node)
            
            # Explore neighbors
            for dx, dy in neighbors:
                neighbor_grid = current_node.position.astype(int) + np.array([dx, dy])
                neighbor_pos = tuple(neighbor_grid)
                
                # Check bounds
                if (neighbor_grid[0] < 0 or neighbor_grid[0] >= self.grid_map.size_x or
                    neighbor_grid[1] < 0 or neighbor_grid[1] >= self.grid_map.size_y):
                    continue
                
                # Check if already in closed set
                if neighbor_pos in closed_set:
                    continue
                
                # Check occupancy with inflation
                if self._is_inflated_occupied(neighbor_grid):
                    continue
                
                # Calculate costs
                move_cost = math.sqrt(dx*dx + dy*dy) * self.grid_map.resolution
                tentative_g = current_node.g_cost + move_cost
                
                # Get or create neighbor node
                if neighbor_pos not in all_nodes:
                    neighbor_node = Node(
                        position=neighbor_grid.astype(float),
                        h_cost=self._heuristic(neighbor_grid, goal_grid)
                    )
                    all_nodes[neighbor_pos] = neighbor_node
                else:
                    neighbor_node = all_nodes[neighbor_pos]
                
                # Update if better path found
                if tentative_g < neighbor_node.g_cost:
                    neighbor_node.g_cost = tentative_g
                    neighbor_node.f_cost = neighbor_node.g_cost + neighbor_node.h_cost
                    neighbor_node.parent = current_node
                    heapq.heappush(open_set, neighbor_node)
        
        return None  # No path found
    
    def _heuristic(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """Heuristic function for A* (Euclidean distance)"""
        diff = pos1 - pos2
        return math.sqrt(diff[0]*diff[0] + diff[1]*diff[1]) * self.grid_map.resolution
    
    def _is_goal_reached(self, current: np.ndarray, goal: np.ndarray) -> bool:
        """Check if goal is reached"""
        return np.linalg.norm(current - goal) <= 1.0  # Within 1 grid cell
    
    def _is_inflated_occupied(self, grid_pos: np.ndarray) -> bool:
        """Check if position is occupied considering inflation radius"""
        inflation_cells = int(self.inflation_radius / self.grid_map.resolution)
        
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                check_pos = grid_pos + np.array([dx, dy])
                
                if (0 <= check_pos[0] < self.grid_map.size_x and 
                    0 <= check_pos[1] < self.grid_map.size_y):
                    if self.grid_map.occupancy_grid[check_pos[0], check_pos[1]] == 1:
                        # Check if within inflation radius
                        dist = math.sqrt(dx*dx + dy*dy) * self.grid_map.resolution
                        if dist <= self.inflation_radius:
                            return True
        return False
    
    def _reconstruct_path(self, goal_node: Node) -> List[np.ndarray]:
        """Reconstruct path from goal node"""
        path = []
        current = goal_node
        
        while current is not None:
            # Convert grid to world coordinates
            world_pos = self.grid_map.grid_to_world(
                np.append(current.position.astype(int), 0)
            )[:2]
            path.append(world_pos)
            current = current.parent
        
        path.reverse()
        return path
    
    def _smooth_path(self, path: List[np.ndarray]) -> List[np.ndarray]:
        """Smooth the path by removing unnecessary waypoints"""
        if len(path) <= 2:
            return path
        
        smoothed = [path[0]]
        
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            
            # Find the farthest reachable point
            while j > i + 1:
                if self._is_line_free(path[i], path[j]):
                    break
                j -= 1
            
            smoothed.append(path[j])
            i = j
        
        return smoothed
    
    def _is_line_free(self, start: np.ndarray, end: np.ndarray) -> bool:
        """Check if line segment between two points is free of obstacles"""
        try:
            # Sample points along the line
            dist = np.linalg.norm(end - start)
            num_samples = max(10, int(dist / (self.grid_map.resolution / 2)))
            
            for i in range(num_samples + 1):
                t = i / num_samples
                point = start + t * (end - start)
                
                if self.grid_map.is_occupied(np.append(point, start[2] if len(start) > 2 else 0)):
                    return False
            
            return True
            
        except Exception as e:
            print(f"Error checking line clearance: {e}")
            return False
    
    def plan_exploration_path(self, current_pos: np.ndarray, 
                            frontiers: List[np.ndarray]) -> Optional[List[np.ndarray]]:
        """Plan optimal path visiting multiple frontiers"""
        if not frontiers:
            return None
        
        if len(frontiers) == 1:
            return self.plan_path(current_pos, frontiers[0])
        
        # For multiple frontiers, use a greedy approach
        # In the original RACER, this would use TSP solver
        remaining_frontiers = frontiers.copy()
        path = [current_pos]
        current = current_pos
        
        while remaining_frontiers:
            # Find nearest frontier
            min_dist = float('inf')
            nearest_frontier = None
            nearest_idx = -1
            
            for i, frontier in enumerate(remaining_frontiers):
                dist = distance_3d(current, frontier)
                if dist < min_dist:
                    min_dist = dist
                    nearest_frontier = frontier
                    nearest_idx = i
            
            if nearest_frontier is not None:
                # Plan path to nearest frontier
                segment_path = self.plan_path(current, nearest_frontier)
                if segment_path:
                    path.extend(segment_path[1:])  # Skip duplicate start point
                    current = nearest_frontier
                
                remaining_frontiers.pop(nearest_idx)
            else:
                break
        
        return path if len(path) > 1 else None