import numpy as np
import math
from typing import Tuple, List
import transforms3d


def quaternion_to_euler(q_w: float, q_x: float, q_y: float, q_z: float) -> Tuple[float, float, float]:
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    return transforms3d.euler.quat2euler([q_w, q_x, q_y, q_z])


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    """Convert Euler angles to quaternion (w, x, y, z)"""
    return transforms3d.euler.euler2quat(roll, pitch, yaw)


def distance_3d(p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance between two points"""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate angle between two vectors in radians"""
    v1_normalized = normalize_vector(v1)
    v2_normalized = normalize_vector(v2)
    return np.arccos(np.clip(np.dot(v1_normalized, v2_normalized), -1.0, 1.0))


def point_to_line_distance(point: np.ndarray, line_start: np.ndarray, line_end: np.ndarray) -> float:
    """Calculate distance from a point to a line segment"""
    line_vec = line_end - line_start
    point_vec = point - line_start
    
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)
    
    line_unit = line_vec / line_len
    proj_length = np.dot(point_vec, line_unit)
    
    if proj_length < 0:
        return np.linalg.norm(point_vec)
    elif proj_length > line_len:
        return np.linalg.norm(point - line_end)
    else:
        proj_point = line_start + proj_length * line_unit
        return np.linalg.norm(point - proj_point)


def wrap_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def interpolate_waypoints(start: np.ndarray, end: np.ndarray, num_points: int) -> List[np.ndarray]:
    """Interpolate waypoints between start and end positions"""
    waypoints = []
    for i in range(num_points):
        t = i / (num_points - 1)
        waypoint = start + t * (end - start)
        waypoints.append(waypoint)
    return waypoints


def check_point_in_bounds(point: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> bool:
    """Check if a point is within specified bounds"""
    return np.all(point >= bounds_min) and np.all(point <= bounds_max)