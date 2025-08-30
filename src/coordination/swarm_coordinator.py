#!/usr/bin/env python3
"""
Swarm Coordinator - Advanced multi-drone coordination system
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import queue

from ..utils.math_utils import distance_3d
from .adaptive_task_allocation import AdaptiveLearningTaskAllocator, TaskType
from .types import DroneState, TaskStatus, DroneRole




@dataclass
class ExplorationTask:
    task_id: str
    target_position: np.ndarray
    priority: float
    estimated_time: float
    assigned_drone: Optional[int] = None
    status: TaskStatus = TaskStatus.PENDING
    created_time: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    required_capabilities: List[str] = field(default_factory=list)


@dataclass
class DroneCapability:
    drone_id: int
    role: DroneRole
    max_speed: float = 5.0
    battery_level: float = 100.0
    sensor_quality: float = 1.0
    exploration_efficiency: float = 1.0
    current_load: int = 0
    max_load: int = 3


class SwarmCoordinator:
    """Advanced multi-drone swarm coordination system with ALTA integration"""
    
    def __init__(self, num_drones: int, communication_range: float = 50.0):
        self.num_drones = num_drones
        self.communication_range = communication_range
        
        # Core data structures
        self.drone_states: Dict[int, DroneState] = {}
        self.drone_capabilities: Dict[int, DroneCapability] = {}
        self.active_tasks: Dict[str, ExplorationTask] = {}
        self.completed_tasks: List[ExplorationTask] = []
        
        # Coordination mechanisms
        self.task_queue = queue.PriorityQueue()
        self.communication_network: Dict[int, List[int]] = {}
        self.shared_map_data: Dict[str, any] = {}
        
        # ALTA Integration - Adaptive Learning Task Allocator
        self.alta_allocator = AdaptiveLearningTaskAllocator(num_drones)
        self.task_performance_history: Dict[str, Dict] = {}
        
        # Threading and control
        self.coordinator_lock = threading.Lock()
        self.running = False
        self.coordinator_thread = None
        
        # Performance metrics
        self.performance_metrics = {
            'total_area_explored': 0.0,
            'exploration_efficiency': 0.0,
            'coordination_overhead': 0.0,
            'collision_avoidance_count': 0,
            'learning_accuracy': 0.0,
            'adaptation_rate': 0.0
        }
        
        # Initialize drone capabilities
        self._initialize_drone_capabilities()
        
    def _initialize_drone_capabilities(self):
        """Initialize capabilities for each drone"""
        roles = [DroneRole.EXPLORER, DroneRole.MAPPER, DroneRole.COORDINATOR]
        
        for drone_id in range(self.num_drones):
            # Assign roles cyclically with some variation
            base_role = roles[drone_id % len(roles)]
            
            # Create capability profile
            capability = DroneCapability(
                drone_id=drone_id,
                role=base_role,
                max_speed=4.0 + np.random.uniform(-0.5, 1.0),  # 3.5-5.0 m/s
                sensor_quality=0.8 + np.random.uniform(0, 0.4),  # 0.8-1.2
                exploration_efficiency=0.9 + np.random.uniform(-0.1, 0.2)  # 0.8-1.1
            )
            
            self.drone_capabilities[drone_id] = capability
            print(f"Drone {drone_id}: Role={base_role.value}, Speed={capability.max_speed:.1f}m/s")
    
    def start_coordination(self):
        """Start the swarm coordination system"""
        self.running = True
        self.coordinator_thread = threading.Thread(target=self._coordination_loop)
        self.coordinator_thread.start()
        print("Swarm coordination system started")
    
    def stop_coordination(self):
        """Stop the swarm coordination system"""
        self.running = False
        if self.coordinator_thread:
            self.coordinator_thread.join()
        print("Swarm coordination system stopped")
    
    def _coordination_loop(self):
        """Main coordination loop"""
        while self.running:
            try:
                with self.coordinator_lock:
                    # Update communication network
                    self._update_communication_network()
                    
                    # Task allocation and management
                    self._allocate_tasks()
                    
                    # Conflict resolution
                    self._resolve_conflicts()
                    
                    # Performance monitoring
                    self._update_performance_metrics()
                
                time.sleep(1.0)  # Coordination frequency: 1Hz
                
            except Exception as e:
                print(f"Error in coordination loop: {e}")
    
    def register_drone(self, drone_id: int, drone_state: DroneState):
        """Register a drone with the coordinator"""
        with self.coordinator_lock:
            self.drone_states[drone_id] = drone_state
            # Initialize communication links
            self.communication_network[drone_id] = []
    
    def update_drone_state(self, drone_id: int, drone_state: DroneState):
        """Update drone state information"""
        with self.coordinator_lock:
            self.drone_states[drone_id] = drone_state
    
    def submit_exploration_task(self, target_position: np.ndarray, 
                              priority: float = 1.0, 
                              estimated_time: float = 30.0) -> str:
        """Submit a new exploration task"""
        task_id = f"task_{time.time():.3f}_{np.random.randint(1000, 9999)}"
        
        task = ExplorationTask(
            task_id=task_id,
            target_position=target_position,
            priority=priority,
            estimated_time=estimated_time
        )
        
        with self.coordinator_lock:
            self.active_tasks[task_id] = task
            # Add to priority queue (negative priority for max-heap behavior)
            self.task_queue.put((-priority, time.time(), task_id))
        
        return task_id
    
    def get_optimal_target(self, drone_id: int) -> Optional[np.ndarray]:
        """Get optimal exploration target for a specific drone"""
        with self.coordinator_lock:
            # Check if drone has assigned tasks
            assigned_task = self._get_assigned_task(drone_id)
            if assigned_task and assigned_task.status == TaskStatus.ASSIGNED:
                assigned_task.status = TaskStatus.IN_PROGRESS
                return assigned_task.target_position
            
            return None
    
    def _get_assigned_task(self, drone_id: int) -> Optional[ExplorationTask]:
        """Get task assigned to specific drone"""
        for task in self.active_tasks.values():
            if task.assigned_drone == drone_id:
                return task
        return None
    
    def _allocate_tasks(self):
        """Intelligent task allocation using ALTA (Adaptive Learning Task Allocation)"""
        if self.task_queue.empty():
            return
        
        available_drones = self._get_available_drones()
        if not available_drones:
            return
        
        # Process tasks from priority queue
        tasks_to_process = []
        temp_queue = queue.PriorityQueue()
        
        while not self.task_queue.empty() and len(tasks_to_process) < len(available_drones):
            priority, timestamp, task_id = self.task_queue.get()
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                if task.status == TaskStatus.PENDING:
                    tasks_to_process.append((priority, task))
                else:
                    temp_queue.put((priority, timestamp, task_id))
        
        # Restore unprocessed tasks to queue
        while not temp_queue.empty():
            self.task_queue.put(temp_queue.get())
        
        # Use ALTA for intelligent task allocation
        if tasks_to_process:
            # Prepare task data for ALTA
            task_contexts = []
            for priority, task in tasks_to_process:
                # Convert exploration task to ALTA format
                task_context = {
                    'task_id': task.task_id,
                    'task_type': TaskType.EXPLORATION,
                    'location': task.target_position,
                    'priority': task.priority,
                    'estimated_duration': task.estimated_time,
                    'complexity': self._estimate_task_complexity(task),
                    'required_sensors': ['depth_camera', 'lidar'],
                    'environmental_conditions': self._get_environmental_context(task.target_position)
                }
                task_contexts.append(task_context)
            
            # Get drone capabilities for ALTA
            current_drone_states = {}
            for drone_id in available_drones:
                if drone_id in self.drone_states:
                    drone_state = self.drone_states[drone_id]
                    capability = self.drone_capabilities[drone_id]
                    current_drone_states[drone_id] = {
                        'position': drone_state.position,
                        'battery_level': capability.battery_level,
                        'current_load': capability.current_load,
                        'max_load': capability.max_load,
                        'exploration_efficiency': capability.exploration_efficiency,
                        'sensor_quality': capability.sensor_quality,
                        'role': capability.role.value
                    }
            
            # Use ALTA to allocate tasks
            try:
                allocations = self.alta_allocator.allocate_tasks(task_contexts, current_drone_states)
                
                # Apply allocations
                for task_id, drone_id in allocations.items():
                    # Find the corresponding task
                    for priority, task in tasks_to_process:
                        if task.task_id == task_id:
                            task.assigned_drone = drone_id
                            task.status = TaskStatus.ASSIGNED
                            self.drone_capabilities[drone_id].current_load += 1
                            if drone_id in available_drones:
                                available_drones.remove(drone_id)
                            
                            # Track task start for learning
                            self.task_performance_history[task_id] = {
                                'assigned_drone': drone_id,
                                'start_time': time.time(),
                                'task_type': TaskType.EXPLORATION,
                                'expected_duration': task.estimated_time,
                                'complexity': self._estimate_task_complexity(task)
                            }
                            
                            print(f"ALTA: Task {task.task_id} assigned to Drone {drone_id} (confidence: {self.alta_allocator._calculate_confidence(drone_id, task_contexts[0]):.2f})")
                            break
                            
            except Exception as e:
                print(f"ALTA allocation failed, falling back to basic allocation: {e}")
                # Fallback to original allocation method
                for priority, task in tasks_to_process:
                    best_drone = self._select_best_drone_for_task(task, available_drones)
                    if best_drone is not None:
                        task.assigned_drone = best_drone
                        task.status = TaskStatus.ASSIGNED
                        self.drone_capabilities[best_drone].current_load += 1
                        available_drones.remove(best_drone)
                        print(f"Fallback: Task {task.task_id} assigned to Drone {best_drone}")
    
    def _get_available_drones(self) -> List[int]:
        """Get list of available drones for task assignment"""
        available = []
        for drone_id, capability in self.drone_capabilities.items():
            if (capability.current_load < capability.max_load and
                capability.battery_level > 20.0 and  # Minimum battery threshold
                drone_id in self.drone_states):
                available.append(drone_id)
        return available
    
    def _select_best_drone_for_task(self, task: ExplorationTask, 
                                  available_drones: List[int]) -> Optional[int]:
        """Select the best drone for a specific task"""
        if not available_drones:
            return None
        
        best_drone = None
        best_score = float('-inf')
        
        for drone_id in available_drones:
            if drone_id not in self.drone_states:
                continue
                
            drone_state = self.drone_states[drone_id]
            capability = self.drone_capabilities[drone_id]
            
            # Calculate selection score
            score = self._calculate_drone_task_score(drone_state, capability, task)
            
            if score > best_score:
                best_score = score
                best_drone = drone_id
        
        return best_drone
    
    def _calculate_drone_task_score(self, drone_state: DroneState, 
                                  capability: DroneCapability, 
                                  task: ExplorationTask) -> float:
        """Calculate fitness score for drone-task pairing"""
        # Distance factor (closer is better)
        distance = distance_3d(drone_state.position, task.target_position)
        distance_score = 1.0 / (1.0 + distance / 10.0)  # Normalize to 0-1
        
        # Capability factors
        capability_score = (
            capability.exploration_efficiency * 0.3 +
            capability.sensor_quality * 0.2 +
            (capability.battery_level / 100.0) * 0.2 +
            (1.0 - capability.current_load / capability.max_load) * 0.3
        )
        
        # Role matching
        role_score = 1.0
        if capability.role == DroneRole.EXPLORER:
            role_score = 1.2  # Prefer explorers for exploration tasks
        elif capability.role == DroneRole.MAPPER:
            role_score = 1.1  # Mappers are also good for exploration
        
        # Combined score
        total_score = distance_score * 0.4 + capability_score * 0.5 + role_score * 0.1
        
        return total_score
    
    def _update_communication_network(self):
        """Update inter-drone communication network"""
        # Clear existing connections
        for drone_id in self.communication_network:
            self.communication_network[drone_id].clear()
        
        # Rebuild network based on proximity
        drone_positions = {
            drone_id: state.position 
            for drone_id, state in self.drone_states.items()
        }
        
        for drone_a in drone_positions:
            for drone_b in drone_positions:
                if drone_a != drone_b:
                    distance = distance_3d(drone_positions[drone_a], drone_positions[drone_b])
                    if distance <= self.communication_range:
                        self.communication_network[drone_a].append(drone_b)
    
    def _resolve_conflicts(self):
        """Detect and resolve conflicts between drones"""
        conflicts_detected = 0
        
        # Check for spatial conflicts
        positions = [(drone_id, state.position) for drone_id, state in self.drone_states.items()]
        
        for i, (drone_a, pos_a) in enumerate(positions):
            for j, (drone_b, pos_b) in enumerate(positions[i+1:], i+1):
                distance = distance_3d(pos_a, pos_b)
                
                if distance < 3.0:  # Minimum safe distance
                    conflicts_detected += 1
                    self._handle_spatial_conflict(drone_a, drone_b, distance)
        
        self.performance_metrics['collision_avoidance_count'] += conflicts_detected
    
    def _handle_spatial_conflict(self, drone_a: int, drone_b: int, distance: float):
        """Handle spatial conflict between two drones"""
        # Simple priority-based resolution
        # Higher capability drone gets priority
        cap_a = self.drone_capabilities.get(drone_a)
        cap_b = self.drone_capabilities.get(drone_b)
        
        if cap_a and cap_b:
            if cap_a.exploration_efficiency > cap_b.exploration_efficiency:
                priority_drone, yield_drone = drone_a, drone_b
            else:
                priority_drone, yield_drone = drone_b, drone_a
                
            print(f"Conflict detected: Drone {yield_drone} yielding to Drone {priority_drone}")
            
            # Mark yielding drone's current task for reassignment
            yielding_task = self._get_assigned_task(yield_drone)
            if yielding_task:
                yielding_task.status = TaskStatus.PENDING
                yielding_task.assigned_drone = None
                self.drone_capabilities[yield_drone].current_load -= 1
                # Add back to queue with higher priority
                self.task_queue.put((-yielding_task.priority - 0.1, time.time(), yielding_task.task_id))
    
    def _update_performance_metrics(self):
        """Update swarm performance metrics including ALTA statistics"""
        if not self.drone_states:
            return
        
        # Calculate exploration efficiency
        active_drones = len([d for d in self.drone_states.values() if d.timestamp > time.time() - 10])
        total_drones = len(self.drone_capabilities)
        
        self.performance_metrics['exploration_efficiency'] = active_drones / total_drones if total_drones > 0 else 0
        
        # Calculate coordination overhead
        active_tasks = len([t for t in self.active_tasks.values() if t.status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]])
        self.performance_metrics['coordination_overhead'] = active_tasks / max(1, active_drones)
        
        # Update ALTA-specific metrics
        try:
            alta_stats = self.alta_allocator.get_performance_statistics()
            self.performance_metrics['learning_accuracy'] = alta_stats.get('average_prediction_accuracy', 0.0)
            self.performance_metrics['adaptation_rate'] = alta_stats.get('adaptation_rate', 0.0)
        except Exception as e:
            # Keep previous values if ALTA stats unavailable
            pass
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report with ALTA insights"""
        with self.coordinator_lock:
            report = {
                'timestamp': time.time(),
                'active_drones': len(self.drone_states),
                'active_tasks': len([t for t in self.active_tasks.values() if t.status != TaskStatus.COMPLETED]),
                'completed_tasks': len(self.completed_tasks),
                'communication_links': sum(len(links) for links in self.communication_network.values()),
                'performance_metrics': self.performance_metrics.copy(),
                'drone_status': {
                    drone_id: {
                        'role': cap.role.value,
                        'battery': cap.battery_level,
                        'load': f"{cap.current_load}/{cap.max_load}",
                        'efficiency': cap.exploration_efficiency
                    }
                    for drone_id, cap in self.drone_capabilities.items()
                },
                'alta_statistics': self.get_alta_statistics()
            }
        
        return report
    
    def complete_task(self, task_id: str, success: bool = True):
        """Mark a task as completed and update ALTA learning"""
        with self.coordinator_lock:
            if task_id in self.active_tasks:
                task = self.active_tasks[task_id]
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                
                # Update ALTA learning with task completion
                if task_id in self.task_performance_history:
                    history = self.task_performance_history[task_id]
                    actual_duration = time.time() - history['start_time']
                    
                    # Calculate performance metrics
                    duration_accuracy = 1.0 - abs(actual_duration - history['expected_duration']) / max(history['expected_duration'], 1.0)
                    duration_accuracy = max(0.0, min(1.0, duration_accuracy))
                    
                    # Update ALTA with learning feedback
                    learning_context = {
                        'task_type': history['task_type'],
                        'complexity': history['complexity'],
                        'actual_duration': actual_duration,
                        'success': success,
                        'performance_score': duration_accuracy if success else 0.0
                    }
                    
                    try:
                        self.alta_allocator.update_learning(
                            history['assigned_drone'], 
                            learning_context
                        )
                        print(f"ALTA updated: Drone {history['assigned_drone']} performance {duration_accuracy:.2f}")
                    except Exception as e:
                        print(f"ALTA learning update failed: {e}")
                    
                    # Clean up history
                    del self.task_performance_history[task_id]
                
                if task.assigned_drone is not None:
                    self.drone_capabilities[task.assigned_drone].current_load -= 1
                
                self.completed_tasks.append(task)
                del self.active_tasks[task_id]
                
                print(f"Task {task_id} marked as {'completed' if success else 'failed'}")
    
    def _estimate_task_complexity(self, task: ExplorationTask) -> float:
        """Estimate task complexity based on various factors"""
        # Base complexity
        complexity = 0.5
        
        # Distance factor - farther targets are more complex
        if hasattr(self, 'drone_states') and self.drone_states:
            min_distance = float('inf')
            for drone_state in self.drone_states.values():
                dist = distance_3d(drone_state.position, task.target_position)
                min_distance = min(min_distance, dist)
            
            if min_distance != float('inf'):
                # Normalize distance to complexity (0-1 scale)
                complexity += min(0.4, min_distance / 50.0)  # 50m = max complexity contribution
        
        # Priority factor - higher priority = higher complexity
        complexity += (1.0 - task.priority) * 0.1
        
        return min(1.0, complexity)
    
    def _get_environmental_context(self, position: np.ndarray) -> Dict:
        """Get environmental context for task location"""
        # Simplified environmental context
        # In real implementation, this could include terrain, weather, obstacles, etc.
        return {
            'terrain_difficulty': 0.5,  # Placeholder
            'obstacle_density': 0.3,    # Placeholder  
            'weather_conditions': 'clear',
            'lighting_conditions': 'good'
        }
    
    def get_alta_statistics(self) -> Dict:
        """Get ALTA performance statistics"""
        try:
            return self.alta_allocator.get_performance_statistics()
        except Exception as e:
            print(f"Failed to get ALTA statistics: {e}")
            return {}