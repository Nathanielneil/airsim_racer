#!/usr/bin/env python3
"""
AirSim RACER - Multi-drone collaborative exploration system
Adapted for Windows deployment with AirSim-UE simulation platform
"""

import argparse
import sys
import time
import yaml
import numpy as np
from pathlib import Path
import threading

from src.airsim_interface.airsim_client import AirSimClient
from src.exploration.exploration_manager import ExplorationManager, ExplorationConfig
from src.utils.math_utils import distance_3d


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}


def create_exploration_config(yaml_config: dict, drone_id: int) -> ExplorationConfig:
    """Create ExplorationConfig from YAML configuration"""
    exp_config = yaml_config.get('exploration', {})
    bounds = exp_config.get('bounds', {})
    
    config = ExplorationConfig(
        drone_num=exp_config.get('drone_num', 1),
        drone_id=drone_id,
        max_exploration_time=exp_config.get('max_exploration_time', 120.0),  # Shorter timeout
        exploration_bounds_min=np.array([
            bounds.get('min_x', -10.0),  # Smaller bounds for depth camera range
            bounds.get('min_y', -10.0), 
            bounds.get('min_z', 0.0)
        ]),
        exploration_bounds_max=np.array([
            bounds.get('max_x', 10.0),
            bounds.get('max_y', 10.0),
            bounds.get('max_z', 5.0)
        ]),
        safety_distance=exp_config.get('safety_distance', 2.0),
        frontier_cluster_tolerance=exp_config.get('frontier', {}).get('cluster_tolerance', 1.0),
        min_frontier_size=exp_config.get('frontier', {}).get('min_frontier_size', 3),  # Smaller minimum
        communication_range=exp_config.get('communication_range', 50.0),
        update_frequency=exp_config.get('update_frequency', 5.0)  # Lower frequency for debugging
    )
    
    return config


def run_single_drone_exploration(drone_id: int, config_path: str):
    """Run exploration for a single drone"""
    print(f"Starting exploration for Drone {drone_id}")
    
    # Load configuration
    yaml_config = load_config(config_path)
    exploration_config = create_exploration_config(yaml_config, drone_id)
    
    # Connect to AirSim
    vehicle_name = f"Drone{drone_id + 1}"  # AirSim uses 1-based naming
    airsim_config = yaml_config.get('airsim', {})
    
    try:
        airsim_client = AirSimClient(
            vehicle_name=vehicle_name,
            ip_address=airsim_config.get('ip_address', '127.0.0.1'),
            port=airsim_config.get('port', 41451)
        )
        
        print(f"Connected to AirSim for {vehicle_name}")
        
        # Create exploration manager
        exploration_manager = ExplorationManager(exploration_config, airsim_client)
        
        # Start exploration
        exploration_manager.start_exploration()
        
        # Monitor exploration progress
        last_progress_time = time.time()
        while exploration_manager.running:
            time.sleep(1.0)
            
            # Print progress every 10 seconds
            if time.time() - last_progress_time > 10.0:
                progress = exploration_manager.get_exploration_progress()
                print(f"Drone {drone_id} - Progress: {progress['explored_percentage']:.1f}%, "
                      f"Frontiers: {progress['frontiers_count']}, "
                      f"Time: {progress['elapsed_time']:.1f}s")
                last_progress_time = time.time()
        
        print(f"Exploration completed for Drone {drone_id}")
        
        # Land the drone
        airsim_client.land()
        airsim_client.disconnect()
        
    except Exception as e:
        print(f"Error in drone {drone_id} exploration: {e}")


def run_multi_drone_exploration(num_drones: int, config_path: str):
    """Run multi-drone collaborative exploration"""
    print(f"Starting multi-drone exploration with {num_drones} drones")
    
    # Create threads for each drone
    threads = []
    
    for drone_id in range(num_drones):
        thread = threading.Thread(
            target=run_single_drone_exploration,
            args=(drone_id, config_path)
        )
        threads.append(thread)
        thread.start()
        
        # Small delay between drone starts
        time.sleep(2.0)
    
    # Wait for all drones to complete
    for thread in threads:
        thread.join()
    
    print("Multi-drone exploration completed!")


def main():
    parser = argparse.ArgumentParser(description='AirSim RACER - Multi-drone collaborative exploration')
    parser.add_argument('--config', '-c', 
                       default='config/exploration_config.yaml',
                       help='Configuration file path')
    parser.add_argument('--drone-id', '-d', 
                       type=int, default=None,
                       help='Single drone ID to run (0-based). If not specified, runs all drones.')
    parser.add_argument('--num-drones', '-n',
                       type=int, default=None, 
                       help='Number of drones for multi-drone exploration')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not Path(args.config).exists():
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        if args.drone_id is not None:
            # Run single drone
            run_single_drone_exploration(args.drone_id, args.config)
        else:
            # Load config to get number of drones
            yaml_config = load_config(args.config)
            num_drones = args.num_drones or yaml_config.get('exploration', {}).get('drone_num', 1)
            
            if num_drones == 1:
                run_single_drone_exploration(0, args.config)
            else:
                run_multi_drone_exploration(num_drones, args.config)
                
    except KeyboardInterrupt:
        print("\nExploration interrupted by user")
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()