import numpy as np
import time
import threading
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from queue import Queue, PriorityQueue
import copy

from ..utils.math_utils import distance_3d, check_point_in_bounds
from ..airsim_interface.airsim_client import AirSimClient
from .frontier_finder import FrontierFinder
from .grid_map import GridMap
from ..planning.path_planner import PathPlanner

if TYPE_CHECKING:
    from ..coordination.swarm_coordinator import SwarmCoordinator


@dataclass
class ExplorationConfig:
    drone_num: int = 1
    drone_id: int = 0
    max_exploration_time: float = 300.0  # seconds
    exploration_bounds_min: np.ndarray = np.array([-10.0, -10.0, 0.0])
    exploration_bounds_max: np.ndarray = np.array([10.0, 10.0, 3.0])
    safety_distance: float = 2.0
    frontier_cluster_tolerance: float = 1.0
    min_frontier_size: int = 5
    communication_range: float = 50.0
    update_frequency: float = 10.0  # Hz


# Import DroneState from coordination module to avoid circular imports
from ..coordination.types import DroneState


class ExplorationManager:
    def __init__(self, config: ExplorationConfig, airsim_client: AirSimClient, swarm_coordinator: Optional['SwarmCoordinator'] = None):
        self.config = config
        self.airsim_client = airsim_client
        self.swarm_coordinator = swarm_coordinator
        
        # Initialize components
        self.grid_map = GridMap(
            bounds_min=config.exploration_bounds_min,
            bounds_max=config.exploration_bounds_max,
            resolution=0.1
        )
        
        self.frontier_finder = FrontierFinder(self.grid_map)
        self.path_planner = PathPlanner(self.grid_map)
        
        # State management
        self.swarm_states: Dict[int, DroneState] = {}
        self.local_frontiers: List[np.ndarray] = []
        self.exploration_complete = False
        self.start_time = None
        
        # Threading
        self.state_lock = threading.Lock()
        self.exploration_thread = None
        self.running = False
        
        # Communication simulation
        self.message_queue = Queue()
        
    def initialize_exploration(self) -> bool:
        """Initialize the exploration system"""
        try:
            # Take off
            if not self.airsim_client.takeoff():
                return False
                
            # Initialize own state
            pos = self.airsim_client.get_position()
            vel = self.airsim_client.get_velocity()
            
            self.swarm_states[self.config.drone_id] = DroneState(
                drone_id=self.config.drone_id,
                position=np.array(pos),
                velocity=np.array(vel),
                battery_level=100.0,
                timestamp=time.time()
            )
            
            self.start_time = time.time()
            
            # Register with swarm coordinator if available
            if self.swarm_coordinator:
                self.swarm_coordinator.register_drone(self.config.drone_id, self.swarm_states[self.config.drone_id])
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize exploration: {e}")
            return False
    
    def start_exploration(self):
        """Start the exploration process"""
        if not self.initialize_exploration():
            print("Failed to initialize exploration")
            return
            
        self.running = True
        self.exploration_thread = threading.Thread(target=self._exploration_loop)
        self.exploration_thread.start()
    
    def stop_exploration(self):
        """Stop the exploration process"""
        self.running = False
        if self.exploration_thread:
            self.exploration_thread.join()
        self.airsim_client.hover()
    
    def _exploration_loop(self):
        """Main exploration loop"""
        while self.running and not self.exploration_complete:
            try:
                # Update map with sensor data
                self._update_occupancy_map()
                
                # Update own state
                self._update_own_state()
                
                # Process communication messages
                self._process_communication()
                
                # Find frontiers
                self._find_frontiers()
                
                # Plan next exploration target
                target = self._plan_next_target()
                
                if target is not None:
                    # Execute movement to target
                    success = self._move_to_target(target)
                    
                    # Notify swarm coordinator of task completion if applicable
                    if self.swarm_coordinator and hasattr(self, '_current_task_id'):
                        self.swarm_coordinator.complete_task(self._current_task_id, success)
                        delattr(self, '_current_task_id')
                        
                else:
                    # No more frontiers, exploration complete
                    self.exploration_complete = True
                    print("Exploration completed!")
                
                # Check exploration timeout
                if time.time() - self.start_time > self.config.max_exploration_time:
                    print("Exploration timeout reached")
                    self.exploration_complete = True
                
                # Sleep for update frequency
                time.sleep(1.0 / self.config.update_frequency)
                
            except Exception as e:
                print(f"Error in exploration loop: {e}")
                break
    
    def _update_occupancy_map(self):
        """Update occupancy map with current sensor data"""
        try:
            # Get camera images
            images = self.airsim_client.get_camera_images(["front_center"])
            
            # Get LiDAR data if available
            lidar_points = self.airsim_client.get_lidar_data()
            
            # Get current position and orientation
            pos = np.array(self.airsim_client.get_position())
            
            # Debug output
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 1
            
            if self._debug_counter % 10 == 1:  # Print every 10th update
                print(f"Map update {self._debug_counter}: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            
            # Update grid map with sensor data - prioritize depth camera
            camera_updated = False
            if images and "front_center" in images:
                depth_img = images["front_center"]["depth"]
                if depth_img is not None:
                    # Enhanced depth camera processing for exploration
                    self.grid_map.update_with_depth_camera(pos, depth_img, camera_fov=90.0, max_range=12.0)
                    camera_updated = True
                    if self._debug_counter % 10 == 1:
                        valid_pixels = np.sum(depth_img > 0.1)
                        depth_range = f"{np.min(depth_img):.2f} - {np.max(depth_img):.2f}"
                        print(f"  Camera: {depth_img.shape}, {valid_pixels} valid pixels, range: {depth_range}m")
            
            # Try LiDAR as supplementary (if available)
            if lidar_points is not None and len(lidar_points) > 0:
                self.grid_map.update_with_lidar(pos, lidar_points)
                if self._debug_counter % 10 == 1:
                    print(f"  LiDAR: {len(lidar_points)} points")
            elif camera_updated and self._debug_counter % 10 == 1:
                print(f"  LiDAR: Not available, using depth camera only")
                
        except Exception as e:
            print(f"Error updating occupancy map: {e}")
    
    def _update_own_state(self):
        """Update own drone state"""
        with self.state_lock:
            pos = self.airsim_client.get_position()
            vel = self.airsim_client.get_velocity()
            
            self.swarm_states[self.config.drone_id].position = np.array(pos)
            self.swarm_states[self.config.drone_id].velocity = np.array(vel)
            self.swarm_states[self.config.drone_id].timestamp = time.time()
    
    def _process_communication(self):
        """Process communication messages from other drones"""
        # In a real system, this would handle actual network communication
        # For simulation, we can skip this or implement simplified message passing
        pass
    
    def _find_frontiers(self):
        """Find exploration frontiers and submit to coordinator"""
        try:
            current_pos = self.swarm_states[self.config.drone_id].position
            self.local_frontiers = self.frontier_finder.find_frontiers(current_pos, search_radius=8.0)
            
            # Submit frontiers as tasks to swarm coordinator
            if self.swarm_coordinator and len(self.local_frontiers) > 0:
                for frontier in self.local_frontiers:
                    # Calculate priority based on distance and other factors
                    distance = distance_3d(current_pos, frontier)
                    priority = 1.0 / (1.0 + distance / 10.0)  # Closer frontiers have higher priority
                    
                    # Submit task to coordinator
                    self.swarm_coordinator.submit_exploration_task(
                        target_position=frontier,
                        priority=priority,
                        estimated_time=max(10.0, distance * 2.0)  # Rough time estimate
                    )
            
            # Debug output every 10 updates
            if hasattr(self, '_debug_counter') and self._debug_counter % 10 == 1:
                print(f"  Frontiers: {len(self.local_frontiers)} found")
                if len(self.local_frontiers) > 0:
                    for i, frontier in enumerate(self.local_frontiers[:3]):
                        print(f"    {i+1}: [{frontier[0]:.2f}, {frontier[1]:.2f}, {frontier[2]:.2f}]")
                        
        except Exception as e:
            print(f"Error finding frontiers: {e}")
            self.local_frontiers = []
    
    def _plan_next_target(self) -> Optional[np.ndarray]:
        """Plan the next exploration target using swarm coordination"""
        current_pos = self.swarm_states[self.config.drone_id].position
        
        # Update drone state in coordinator
        if self.swarm_coordinator:
            self.swarm_coordinator.update_drone_state(self.config.drone_id, self.swarm_states[self.config.drone_id])
            
            # Get coordinated target from swarm coordinator
            coordinated_target = self.swarm_coordinator.get_optimal_target(self.config.drone_id)
            if coordinated_target is not None:
                print(f"Drone {self.config.drone_id} received coordinated target: [{coordinated_target[0]:.2f}, {coordinated_target[1]:.2f}, {coordinated_target[2]:.2f}]")
                
                # Store task ID for completion tracking
                # Find the task ID for this target (simplified approach)
                for task_id, task in self.swarm_coordinator.active_tasks.items():
                    if (task.assigned_drone == self.config.drone_id and 
                        np.allclose(task.target_position, coordinated_target, atol=0.1)):
                        self._current_task_id = task_id
                        break
                
                return coordinated_target
        
        # Fallback: If no coordinated target and no frontiers, create exploration waypoints
        if not self.local_frontiers:
            # Force exploration with predefined waypoints
            if not hasattr(self, '_exploration_waypoints'):
                self._exploration_waypoints = [
                    current_pos + np.array([3.0, 0.0, 0.0]),
                    current_pos + np.array([3.0, 3.0, 0.0]),
                    current_pos + np.array([0.0, 3.0, 0.0]),
                    current_pos + np.array([-3.0, 3.0, 0.0]),
                    current_pos + np.array([-3.0, 0.0, 0.0]),
                    current_pos + np.array([-3.0, -3.0, 0.0]),
                    current_pos + np.array([0.0, -3.0, 0.0]),
                    current_pos + np.array([3.0, -3.0, 0.0])
                ]
                self._waypoint_index = 0
            
            if self._waypoint_index < len(self._exploration_waypoints):
                target = self._exploration_waypoints[self._waypoint_index]
                self._waypoint_index += 1
                print(f"No frontiers found, using exploration waypoint: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
                return target
            else:
                return None
        
        # Simple greedy selection - choose closest accessible frontier
        best_frontier = None
        min_cost = float('inf')
        
        for frontier in self.local_frontiers:
            # Check if frontier is within bounds
            if not check_point_in_bounds(frontier, 
                                       self.config.exploration_bounds_min,
                                       self.config.exploration_bounds_max):
                continue
                
            # Calculate cost (distance + other factors)
            dist = distance_3d(current_pos, frontier)
            
            # Add penalty for areas already explored by other drones
            penalty = self._calculate_swarm_penalty(frontier)
            
            total_cost = dist + penalty
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_frontier = frontier
        
        return best_frontier
    
    def _calculate_swarm_penalty(self, target: np.ndarray) -> float:
        """Calculate penalty for targets near other drones"""
        penalty = 0.0
        for drone_id, state in self.swarm_states.items():
            if drone_id != self.config.drone_id:
                dist = distance_3d(target, state.position)
                if dist < self.config.safety_distance:
                    penalty += (self.config.safety_distance - dist) * 10.0
        return penalty
    
    def _move_to_target(self, target: np.ndarray) -> bool:
        """Move drone to target position and return success status"""
        try:
            # Plan path to target
            current_pos = self.swarm_states[self.config.drone_id].position
            path = self.path_planner.plan_path(current_pos, target)
            
            if path:
                # Execute path
                for waypoint in path[1:]:  # Skip current position
                    if not self.running:
                        return False
                    success = self.airsim_client.move_to_position(
                        waypoint[0], waypoint[1], waypoint[2], velocity=3.0
                    )
                    if not success:
                        print(f"Failed to reach waypoint: {waypoint}")
                        return False
                    time.sleep(0.1)  # Small delay between waypoints
                return True
            else:
                # Direct movement if no path found
                success = self.airsim_client.move_to_position(
                    target[0], target[1], target[2], velocity=2.0
                )
                return success
                
        except Exception as e:
            print(f"Error moving to target: {e}")
            return False
    
    def get_exploration_progress(self) -> Dict:
        """Get current exploration progress"""
        explored_volume = self.grid_map.get_explored_volume()
        total_volume = self.grid_map.get_total_volume()
        
        return {
            'explored_percentage': (explored_volume / total_volume) * 100.0 if total_volume > 0 else 0.0,
            'frontiers_count': len(self.local_frontiers),
            'elapsed_time': time.time() - self.start_time if self.start_time else 0.0,
            'drone_position': self.swarm_states[self.config.drone_id].position.tolist(),
            'exploration_complete': self.exploration_complete
        }