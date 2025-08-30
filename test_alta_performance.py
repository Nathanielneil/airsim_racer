#!/usr/bin/env python3
"""
ALTA Performance Test - Validate Adaptive Learning Task Allocation
Test scenario to demonstrate learning improvement over time
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.coordination.adaptive_task_allocation import AdaptiveLearningTaskAllocator, TaskType
from src.coordination.swarm_coordinator import SwarmCoordinator


def generate_test_tasks(num_tasks: int = 50) -> List[Dict]:
    """Generate diverse test tasks for evaluation"""
    tasks = []
    
    for i in range(num_tasks):
        # Create varied task scenarios
        task_type = TaskType.EXPLORATION
        complexity = np.random.uniform(0.1, 1.0)
        
        # Different locations to test spatial reasoning
        location = np.array([
            np.random.uniform(-20, 20),  # X
            np.random.uniform(-20, 20),  # Y
            np.random.uniform(1, 5)      # Z
        ])
        
        priority = np.random.uniform(0.1, 1.0)
        estimated_duration = 10.0 + complexity * 30.0  # 10-40 seconds
        
        task = {
            'task_id': f'test_task_{i:03d}',
            'task_type': task_type,
            'location': location,
            'priority': priority,
            'estimated_duration': estimated_duration,
            'complexity': complexity,
            'required_sensors': ['depth_camera'] + (['lidar'] if complexity > 0.5 else []),
            'environmental_conditions': {
                'terrain_difficulty': np.random.uniform(0.2, 0.8),
                'obstacle_density': np.random.uniform(0.1, 0.6),
                'weather_conditions': 'clear',
                'lighting_conditions': 'good'
            }
        }
        
        tasks.append(task)
    
    return tasks


def generate_drone_states(num_drones: int = 3) -> Dict:
    """Generate simulated drone states"""
    drone_states = {}
    
    for drone_id in range(num_drones):
        # Spread drones across area
        position = np.array([
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10), 
            np.random.uniform(2, 4)
        ])
        
        drone_states[drone_id] = {
            'position': position,
            'battery_level': np.random.uniform(60, 100),
            'current_load': np.random.randint(0, 2),
            'max_load': 3,
            'exploration_efficiency': 0.8 + np.random.uniform(-0.2, 0.2),
            'sensor_quality': 0.9 + np.random.uniform(-0.1, 0.1),
            'role': ['explorer', 'mapper', 'coordinator'][drone_id % 3]
        }
    
    return drone_states


def simulate_task_execution(allocator: AdaptiveLearningTaskAllocator, 
                          allocations: Dict[str, int],
                          tasks: List[Dict],
                          drone_states: Dict) -> List[Dict]:
    """Simulate task execution and generate learning feedback"""
    results = []
    
    for task in tasks:
        task_id = task['task_id']
        if task_id not in allocations:
            continue
            
        assigned_drone = allocations[task_id]
        
        # Simulate actual execution time with some realism
        expected_duration = task['estimated_duration']
        complexity = task['complexity']
        drone_efficiency = drone_states[assigned_drone]['exploration_efficiency']
        
        # Add realistic variance based on drone capability and task complexity
        performance_factor = drone_efficiency * (2.0 - complexity)  # Better drones handle complex tasks better
        actual_duration = expected_duration / performance_factor + np.random.normal(0, 2.0)
        actual_duration = max(5.0, actual_duration)  # Minimum task time
        
        # Determine success based on realistic factors
        success_probability = 0.8 * drone_efficiency * (1.1 - complexity * 0.5)
        success = np.random.random() < success_probability
        
        # Calculate performance score
        if success:
            duration_accuracy = 1.0 - abs(actual_duration - expected_duration) / max(expected_duration, 1.0)
            performance_score = max(0.0, min(1.0, duration_accuracy)) * drone_efficiency
        else:
            performance_score = 0.0
        
        # Create learning context
        learning_context = {
            'task_type': task['task_type'],
            'complexity': complexity,
            'actual_duration': actual_duration,
            'success': success,
            'performance_score': performance_score
        }
        
        # Update ALTA learning
        allocator.update_learning(assigned_drone, learning_context)
        
        results.append({
            'task_id': task_id,
            'assigned_drone': assigned_drone,
            'expected_duration': expected_duration,
            'actual_duration': actual_duration,
            'success': success,
            'performance_score': performance_score,
            'complexity': complexity
        })
    
    return results


def run_alta_performance_test():
    """Run comprehensive ALTA performance test"""
    print("=== ALTA Performance Test ===")
    
    # Test configuration
    num_drones = 3
    num_test_rounds = 10
    tasks_per_round = 20
    
    # Initialize ALTA
    allocator = AdaptiveLearningTaskAllocator(num_drones)
    
    # Performance tracking
    round_performance = []
    allocation_accuracy = []
    learning_progress = []
    
    for round_num in range(num_test_rounds):
        print(f"\n--- Test Round {round_num + 1}/{num_test_rounds} ---")
        
        # Generate test scenario
        test_tasks = generate_test_tasks(tasks_per_round)
        drone_states = generate_drone_states(num_drones)
        
        # Allocate tasks using ALTA
        start_time = time.time()
        allocations = allocator.allocate_tasks(test_tasks, drone_states)
        allocation_time = time.time() - start_time
        
        print(f"Allocated {len(allocations)} tasks in {allocation_time:.3f}s")
        
        # Simulate task execution and learning
        execution_results = simulate_task_execution(allocator, allocations, test_tasks, drone_states)
        
        # Calculate round performance metrics
        success_rate = sum(1 for r in execution_results if r['success']) / len(execution_results)
        avg_performance = np.mean([r['performance_score'] for r in execution_results])
        duration_accuracy = np.mean([
            1.0 - abs(r['actual_duration'] - r['expected_duration']) / max(r['expected_duration'], 1.0)
            for r in execution_results if r['success']
        ]) if any(r['success'] for r in execution_results) else 0.0
        
        round_performance.append({
            'round': round_num + 1,
            'success_rate': success_rate,
            'avg_performance': avg_performance,
            'duration_accuracy': duration_accuracy,
            'allocation_time': allocation_time
        })
        
        # Get ALTA statistics
        alta_stats = allocator.get_performance_statistics()
        learning_progress.append(alta_stats)
        
        print(f"Success Rate: {success_rate:.2f}")
        print(f"Avg Performance: {avg_performance:.2f}")
        print(f"Duration Accuracy: {duration_accuracy:.2f}")
        print(f"Learning Episodes: {alta_stats.get('learning_episodes', 0)}")
        
        # Check for learning improvement
        if round_num >= 2:
            recent_performance = [rp['avg_performance'] for rp in round_performance[-3:]]
            if len(recent_performance) >= 2:
                improvement = recent_performance[-1] - recent_performance[0]
                print(f"Learning Trend: {improvement:+.3f}")
    
    # Analysis and visualization
    print("\n=== Performance Analysis ===")
    
    # Calculate learning improvement
    initial_performance = np.mean([rp['avg_performance'] for rp in round_performance[:3]])
    final_performance = np.mean([rp['avg_performance'] for rp in round_performance[-3:]])
    improvement = final_performance - initial_performance
    
    print(f"Initial Performance: {initial_performance:.3f}")
    print(f"Final Performance: {final_performance:.3f}")
    print(f"Learning Improvement: {improvement:+.3f} ({improvement/initial_performance*100:+.1f}%)")
    
    # Plot learning curves
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        rounds = [rp['round'] for rp in round_performance]
        
        # Performance over time
        ax1.plot(rounds, [rp['avg_performance'] for rp in round_performance], 'b-o', label='Avg Performance')
        ax1.plot(rounds, [rp['success_rate'] for rp in round_performance], 'g-s', label='Success Rate')
        ax1.set_xlabel('Test Round')
        ax1.set_ylabel('Performance Score')
        ax1.set_title('ALTA Learning Performance Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Duration accuracy
        ax2.plot(rounds, [rp['duration_accuracy'] for rp in round_performance], 'r-^', label='Duration Accuracy')
        ax2.set_xlabel('Test Round')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Task Duration Prediction Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Allocation time efficiency
        ax3.plot(rounds, [rp['allocation_time'] for rp in round_performance], 'm-d', label='Allocation Time')
        ax3.set_xlabel('Test Round')
        ax3.set_ylabel('Time (seconds)')
        ax3.set_title('Task Allocation Efficiency')
        ax3.legend()
        ax3.grid(True)
        
        # Learning episodes
        learning_episodes = [lp.get('learning_episodes', 0) for lp in learning_progress]
        ax4.plot(rounds, learning_episodes, 'c-o', label='Learning Episodes')
        ax4.set_xlabel('Test Round')
        ax4.set_ylabel('Episodes')
        ax4.set_title('Cumulative Learning Episodes')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('alta_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("Performance plots saved as 'alta_performance_analysis.png'")
        
    except ImportError:
        print("Matplotlib not available, skipping plots")
    
    # Final ALTA statistics
    final_stats = allocator.get_performance_statistics()
    print(f"\n=== Final ALTA Statistics ===")
    print(f"Total Allocations: {final_stats.get('total_allocations', 0)}")
    print(f"Learning Episodes: {final_stats.get('learning_episodes', 0)}")
    print(f"Average Confidence: {final_stats.get('average_confidence', 0):.3f}")
    print(f"Prediction Accuracy: {final_stats.get('average_prediction_accuracy', 0):.3f}")
    
    # Drone-specific learning progress
    if 'drone_capabilities' in final_stats:
        print("\nDrone Learning Progress:")
        for drone_id, capabilities in final_stats['drone_capabilities'].items():
            exploration_skill = capabilities.get('exploration', {}).get('skill_level', 0)
            print(f"  Drone {drone_id}: Exploration Skill {exploration_skill:.3f}")
    
    return improvement > 0.05  # Test passes if improvement > 5%


if __name__ == "__main__":
    # Run the test
    success = run_alta_performance_test()
    
    if success:
        print("\n✅ ALTA Performance Test PASSED - Learning improvement detected!")
        sys.exit(0)
    else:
        print("\n❌ ALTA Performance Test FAILED - No significant learning improvement")
        sys.exit(1)