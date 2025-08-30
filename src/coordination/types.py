#!/usr/bin/env python3
"""
Common types for coordination module
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DroneRole(Enum):
    EXPLORER = "explorer"      # Primary exploration
    MAPPER = "mapper"          # Focus on mapping
    COORDINATOR = "coordinator"  # Coordination tasks
    SUPPORT = "support"        # Support other drones


@dataclass 
class DroneState:
    drone_id: int
    position: np.ndarray
    velocity: np.ndarray
    battery_level: float
    timestamp: float
    current_goal: Optional[np.ndarray] = None