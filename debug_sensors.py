#!/usr/bin/env python3
"""
Debug script to test AirSim sensor data acquisition
"""

import sys
import time
import numpy as np
from src.airsim_interface.airsim_client import AirSimClient

def test_sensors():
    """Test AirSim sensor data acquisition"""
    print("Testing AirSim sensor data...")
    
    try:
        # Connect to AirSim
        client = AirSimClient("Drone1")
        print("Connected to AirSim")
        
        # Test position
        pos = client.get_position()
        print(f"Position: {pos}")
        
        # Test camera images
        print("Testing camera images...")
        images = client.get_camera_images(["front_center"])
        if images:
            for camera_name, data in images.items():
                print(f"Camera {camera_name}:")
                if 'rgb' in data:
                    print(f"  RGB: {data['rgb'].shape if data['rgb'] is not None else 'None'}")
                if 'depth' in data:
                    print(f"  Depth: {data['depth'].shape if data['depth'] is not None else 'None'}")
        else:
            print("No camera data received")
        
        # Test LiDAR
        print("Testing LiDAR...")
        lidar_data = client.get_lidar_data()
        print(f"LiDAR points: {len(lidar_data) if lidar_data is not None else 'None'}")
        
        # Test takeoff
        print("Testing takeoff...")
        if client.takeoff():
            print("Takeoff successful")
            time.sleep(2)
            
            # Test again after takeoff
            print("Re-testing sensors after takeoff...")
            
            pos = client.get_position()
            print(f"New position: {pos}")
            
            images = client.get_camera_images(["front_center"])
            if images and 'front_center' in images:
                data = images['front_center']
                if data['depth'] is not None:
                    depth = data['depth']
                    print(f"Depth image shape: {depth.shape}")
                    print(f"Depth range: {np.min(depth):.2f} - {np.max(depth):.2f}")
                    
                    # Check for valid depth data
                    valid_pixels = np.sum(depth > 0)
                    total_pixels = depth.size
                    print(f"Valid depth pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
                else:
                    print("No depth data")
            
            lidar_data = client.get_lidar_data()
            if lidar_data is not None and len(lidar_data) > 0:
                print(f"LiDAR points after takeoff: {len(lidar_data)}")
                print(f"LiDAR point sample: {lidar_data[:3]}")
            else:
                print("No LiDAR data after takeoff")
            
            client.hover()
        else:
            print("Takeoff failed")
        
        client.disconnect()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sensors()