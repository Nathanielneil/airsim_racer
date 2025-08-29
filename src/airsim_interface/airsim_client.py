import airsim
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
import threading
import queue


class AirSimClient:
    def __init__(self, vehicle_name: str = "Drone1", ip_address: str = "127.0.0.1", port: int = 41451):
        self.vehicle_name = vehicle_name
        self.client = airsim.MultirotorClient(ip=ip_address, port=port)
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name)
        self.client.armDisarm(True, vehicle_name)
        
        self.position_lock = threading.Lock()
        self.current_position = airsim.Vector3r(0, 0, 0)
        self.current_orientation = airsim.Quaternionr()
        
    def takeoff(self, timeout_sec: float = 20.0) -> bool:
        try:
            self.client.takeoffAsync(timeout_sec, self.vehicle_name).join()
            return True
        except Exception as e:
            print(f"Takeoff failed: {e}")
            return False
    
    def move_to_position(self, x: float, y: float, z: float, velocity: float = 5.0) -> bool:
        try:
            result = self.client.moveToPositionAsync(x, y, z, velocity, vehicle_name=self.vehicle_name)
            result.join()
            return True
        except Exception as e:
            print(f"Move to position failed: {e}")
            return False
    
    def get_position(self) -> Tuple[float, float, float]:
        with self.position_lock:
            state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
            pos = state.kinematics_estimated.position
            return pos.x_val, pos.y_val, pos.z_val
    
    def get_orientation(self) -> Tuple[float, float, float, float]:
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        orientation = state.kinematics_estimated.orientation
        return orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val
    
    def get_velocity(self) -> Tuple[float, float, float]:
        state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
        vel = state.kinematics_estimated.linear_velocity
        return vel.x_val, vel.y_val, vel.z_val
    
    def get_camera_images(self, camera_names: List[str] = None) -> Dict:
        if camera_names is None:
            camera_names = ["front_center"]
        
        requests = []
        for camera_name in camera_names:
            requests.append(airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False))
            requests.append(airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True))
        
        responses = self.client.simGetImages(requests, vehicle_name=self.vehicle_name)
        
        images = {}
        for i, camera_name in enumerate(camera_names):
            rgb_response = responses[i*2]
            depth_response = responses[i*2 + 1]
            
            rgb_img = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
            rgb_img = rgb_img.reshape(rgb_response.height, rgb_response.width, 3)
            
            depth_img = airsim.list_to_2d_float_array(depth_response.image_data_float, 
                                                    depth_response.width, 
                                                    depth_response.height)
            depth_img = np.array(depth_img)
            
            # Scale depth values from normalized (0-1) to actual meters
            # AirSim normalizes depth based on camera settings
            # Convert to actual distance in meters (up to 50m based on camera config)
            depth_img = depth_img * 50.0  # Scale to actual meters
            
            images[camera_name] = {
                'rgb': rgb_img,
                'depth': depth_img,
                'camera_info': {
                    'width': rgb_response.width,
                    'height': rgb_response.height,
                    'fov': 90.0,  # Default FOV
                    'max_depth': 50.0  # Maximum depth range
                }
            }
        
        return images
    
    def get_lidar_data(self, lidar_name: str = "Lidar") -> np.ndarray:
        try:
            lidar_data = self.client.getLidarData(lidar_name, vehicle_name=self.vehicle_name)
            points = np.array(lidar_data.point_cloud_xyz, dtype=np.float32)
            return points.reshape(-1, 3)
        except:
            return np.array([])
    
    def set_velocity(self, vx: float, vy: float, vz: float, duration: float = 1.0):
        self.client.moveByVelocityAsync(vx, vy, vz, duration, vehicle_name=self.vehicle_name)
    
    def hover(self):
        self.client.hoverAsync(vehicle_name=self.vehicle_name).join()
    
    def land(self):
        self.client.landAsync(vehicle_name=self.vehicle_name).join()
    
    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True, self.vehicle_name)
        self.client.armDisarm(True, self.vehicle_name)
    
    def get_collision_info(self) -> bool:
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.vehicle_name)
        return collision_info.has_collided
    
    def disconnect(self):
        self.client.armDisarm(False, self.vehicle_name)
        self.client.enableApiControl(False, self.vehicle_name)