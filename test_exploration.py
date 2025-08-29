#!/usr/bin/env python3
"""
Simple exploration test with debugging output
"""

import time
import numpy as np
from src.airsim_interface.airsim_client import AirSimClient
from src.exploration.grid_map import GridMap
from src.exploration.frontier_finder import FrontierFinder

def test_basic_exploration():
    """Test basic exploration functionality"""
    print("Starting basic exploration test...")
    
    try:
        # Connect to AirSim
        client = AirSimClient("Drone1")
        print("Connected to AirSim")
        
        # Create grid map
        bounds_min = np.array([-10.0, -10.0, 0.0])
        bounds_max = np.array([10.0, 10.0, 5.0])
        grid_map = GridMap(bounds_min, bounds_max, resolution=0.5)
        print(f"Grid map created: {grid_map.size_x}x{grid_map.size_y}x{grid_map.size_z}")
        
        # Create frontier finder
        frontier_finder = FrontierFinder(grid_map, cluster_tolerance=2.0, min_frontier_size=5)
        print("Frontier finder created")
        
        # Takeoff
        print("Taking off...")
        if not client.takeoff():
            print("Takeoff failed")
            return
        print("Takeoff successful")
        
        time.sleep(3)  # Wait for stable hover
        
        # Test sensor data acquisition
        for i in range(5):
            print(f"\n--- Test iteration {i+1} ---")
            
            # Get position
            pos = np.array(client.get_position())
            print(f"Position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
            
            # Check if position is within bounds
            if not (bounds_min[0] <= pos[0] <= bounds_max[0] and
                    bounds_min[1] <= pos[1] <= bounds_max[1] and
                    bounds_min[2] <= pos[2] <= bounds_max[2]):
                print(f"WARNING: Position outside bounds!")
                print(f"Bounds: {bounds_min} to {bounds_max}")
            
            # Get camera data
            try:
                images = client.get_camera_images(["front_center"])
                if images and 'front_center' in images:
                    depth_img = images['front_center']['depth']
                    if depth_img is not None:
                        valid_pixels = np.sum(depth_img > 0.1)  # Valid depth > 10cm
                        print(f"Depth image: {depth_img.shape}, valid pixels: {valid_pixels}")
                        print(f"Depth range: {np.min(depth_img):.2f} - {np.max(depth_img):.2f}")
                        
                        # Update grid map with depth camera
                        print("Updating grid map with camera data...")
                        grid_map.update_with_depth_camera(pos, depth_img, camera_fov=90.0, max_range=15.0)
                    else:
                        print("No depth image data")
                else:
                    print("No camera images received")
            except Exception as e:
                print(f"Camera error: {e}")
            
            # Get LiDAR data
            try:
                lidar_points = client.get_lidar_data()
                if lidar_points is not None and len(lidar_points) > 0:
                    print(f"LiDAR points: {len(lidar_points)}")
                    print(f"Sample points: {lidar_points[:3]}")
                    
                    # Update grid map with LiDAR
                    print("Updating grid map with LiDAR data...")
                    grid_map.update_with_lidar(pos, lidar_points)
                else:
                    print("No LiDAR data")
            except Exception as e:
                print(f"LiDAR error: {e}")
            
            # Check grid map status
            explored_vol = grid_map.get_explored_volume()
            total_vol = grid_map.get_total_volume()
            explored_pct = (explored_vol / total_vol) * 100 if total_vol > 0 else 0
            print(f"Grid map: {explored_pct:.2f}% explored")
            
            # Find frontiers
            try:
                print("Finding frontiers...")
                frontiers = frontier_finder.find_frontiers(pos, search_radius=10.0)
                print(f"Found {len(frontiers)} frontiers")
                
                if len(frontiers) > 0:
                    print("Frontier positions:")
                    for j, frontier in enumerate(frontiers[:3]):  # Show first 3
                        print(f"  {j+1}: [{frontier[0]:.2f}, {frontier[1]:.2f}, {frontier[2]:.2f}]")
                    
                    # Try moving to first frontier
                    if len(frontiers) > 0:
                        target = frontiers[0]
                        print(f"Moving to frontier: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")
                        
                        success = client.move_to_position(target[0], target[1], target[2], velocity=2.0)
                        if success:
                            print("Movement command sent")
                        else:
                            print("Movement failed")
                        
                        time.sleep(3)  # Wait for movement
                else:
                    print("No frontiers found - this might be normal if area is fully explored")
                    
                    # Add some artificial exploration by moving in a pattern
                    print("Performing pattern movement for exploration...")
                    current_pos = pos
                    moves = [
                        [current_pos[0] + 2, current_pos[1], current_pos[2]],
                        [current_pos[0] + 2, current_pos[1] + 2, current_pos[2]],
                        [current_pos[0], current_pos[1] + 2, current_pos[2]],
                        [current_pos[0], current_pos[1], current_pos[2]]
                    ]
                    
                    for move in moves:
                        if (bounds_min[0] <= move[0] <= bounds_max[0] and
                            bounds_min[1] <= move[1] <= bounds_max[1]):
                            client.move_to_position(move[0], move[1], move[2], velocity=1.0)
                            time.sleep(2)
                            break
                    
            except Exception as e:
                print(f"Frontier finding error: {e}")
                import traceback
                traceback.print_exc()
            
            print(f"Waiting before next iteration...")
            time.sleep(2)
        
        # Final statistics
        print(f"\n--- Final Results ---")
        explored_vol = grid_map.get_explored_volume()
        total_vol = grid_map.get_total_volume()
        explored_pct = (explored_vol / total_vol) * 100 if total_vol > 0 else 0
        print(f"Final exploration: {explored_pct:.2f}%")
        
        # Check occupancy grid status
        free_cells = np.sum(grid_map.occupancy_grid == 0)
        occupied_cells = np.sum(grid_map.occupancy_grid == 1)
        unknown_cells = np.sum(grid_map.occupancy_grid == -1)
        total_cells = grid_map.size_x * grid_map.size_y
        
        print(f"Grid status:")
        print(f"  Free: {free_cells}/{total_cells} ({100*free_cells/total_cells:.1f}%)")
        print(f"  Occupied: {occupied_cells}/{total_cells} ({100*occupied_cells/total_cells:.1f}%)")
        print(f"  Unknown: {unknown_cells}/{total_cells} ({100*unknown_cells/total_cells:.1f}%)")
        
        # Land
        print("Landing...")
        client.land()
        client.disconnect()
        print("Test completed")
        
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_exploration()