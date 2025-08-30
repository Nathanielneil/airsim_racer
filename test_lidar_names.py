#!/usr/bin/env python3
"""
Test different LiDAR sensor names
"""

from src.airsim_interface.airsim_client import AirSimClient

def test_lidar_names():
    client = AirSimClient("Drone1")
    client.takeoff()
    
    # Test different possible LiDAR names
    names_to_test = [
        'Lidar', 'LiDAR', 'lidar', 'LIDAR',
        'Lidar1', 'LiDAR1', 'lidar1',
        'VelodyneLidar', 'Velodyne', 'RpLidar'
    ]
    
    print("Testing different LiDAR sensor names...")
    for name in names_to_test:
        try:
            data = client.get_lidar_data(name)
            if data is not None and len(data) > 0:
                print(f"✅ SUCCESS: '{name}' returned {len(data)} points")
                # Show sample points
                print(f"Sample points: {data[:3]}")
                break
            else:
                print(f"❌ '{name}': No data")
        except Exception as e:
            print(f"❌ '{name}': Error - {str(e)[:50]}")
    
    client.land()

if __name__ == "__main__":
    test_lidar_names()