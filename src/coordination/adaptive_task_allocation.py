#!/usr/bin/env python3
"""
Adaptive Learning Task Allocation (ALTA) Algorithm
自适应学习任务分配算法 - 详细实现

核心创新：
1. Multi-Armed Bandit based drone capability learning
2. Dynamic Q-Learning for task-drone matching  
3. Experience Replay for continuous improvement
4. Meta-learning for cross-task knowledge transfer
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json
import threading
from enum import Enum
import pickle

from ..utils.math_utils import distance_3d


class TaskType(Enum):
    EXPLORATION = "exploration"
    MAPPING = "mapping"
    SURVEILLANCE = "surveillance"
    TRANSPORTATION = "transportation"
    EMERGENCY_RESPONSE = "emergency_response"


class LearningStrategy(Enum):
    EPSILON_GREEDY = "epsilon_greedy"
    UCB = "upper_confidence_bound"
    THOMPSON_SAMPLING = "thompson_sampling"
    META_LEARNING = "meta_learning"


@dataclass
class Task:
    task_id: str
    task_type: TaskType
    priority: float
    complexity: float
    required_capabilities: List[str]
    estimated_duration: float
    deadline: Optional[float] = None
    location: Optional[np.ndarray] = None
    dependencies: List[str] = field(default_factory=list)
    reward_function: Optional[callable] = None


@dataclass
class DroneCapability:
    drone_id: int
    base_capabilities: Dict[str, float]  # 基础能力值
    learned_capabilities: Dict[str, float]  # 学习到的能力值
    experience_count: Dict[TaskType, int] = field(default_factory=lambda: defaultdict(int))
    success_rate: Dict[TaskType, float] = field(default_factory=lambda: defaultdict(float))
    avg_completion_time: Dict[TaskType, float] = field(default_factory=lambda: defaultdict(float))
    learning_rate: float = 0.1
    confidence_intervals: Dict[TaskType, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class AllocationExperience:
    """经验回放数据结构"""
    state: Dict  # 当时的系统状态
    action: int  # 选择的无人机
    reward: float  # 获得的奖励
    next_state: Dict  # 执行后的状态
    task_info: Task  # 任务信息
    timestamp: float
    actual_duration: float
    success: bool


class AdaptiveLearningTaskAllocator:
    """自适应学习任务分配器"""
    
    def __init__(self, num_drones: int, learning_strategy: LearningStrategy = LearningStrategy.META_LEARNING):
        self.num_drones = num_drones
        self.learning_strategy = learning_strategy
        
        # 核心数据结构
        self.drone_capabilities: Dict[int, DroneCapability] = {}
        self.q_table: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.experience_replay_buffer = deque(maxlen=10000)
        
        # 学习参数
        self.epsilon = 0.3  # 探索率
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        
        # Multi-Armed Bandit 参数
        self.bandit_arms: Dict[int, Dict[str, Any]] = {}  # 每个无人机的bandit arm
        self.ucb_exploration_param = 2.0
        
        # Meta-learning 参数
        self.meta_knowledge: Dict[str, Any] = {}
        self.cross_task_transfer_matrix = np.ones((len(TaskType), len(TaskType))) * 0.1
        
        # 性能跟踪
        self.allocation_history: List[Dict] = []
        self.performance_metrics = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'avg_allocation_time': 0.0,
            'learning_convergence': 0.0
        }
        
        # 线程安全
        self.allocation_lock = threading.Lock()
        
        # 初始化无人机能力
        self._initialize_drone_capabilities()
        self._initialize_bandit_arms()
    
    def _initialize_drone_capabilities(self):
        """初始化无人机能力模型"""
        capability_types = [
            'speed', 'endurance', 'sensor_quality', 'navigation_accuracy',
            'communication_range', 'payload_capacity', 'obstacle_avoidance',
            'cooperative_ability', 'learning_speed', 'adaptability'
        ]
        
        for drone_id in range(self.num_drones):
            # 为每个无人机生成独特的能力配置
            base_caps = {}
            learned_caps = {}
            
            # 生成基础能力 (0.3-1.0范围，每个无人机有不同特长)
            specialization = np.random.choice(capability_types)  # 每个无人机的专长
            
            for cap_type in capability_types:
                if cap_type == specialization:
                    base_value = 0.7 + np.random.uniform(0, 0.3)  # 专长能力较高
                else:
                    base_value = 0.3 + np.random.uniform(0, 0.4)  # 其他能力适中
                
                base_caps[cap_type] = base_value
                learned_caps[cap_type] = base_value  # 初始学习值等于基础值
            
            self.drone_capabilities[drone_id] = DroneCapability(
                drone_id=drone_id,
                base_capabilities=base_caps,
                learned_capabilities=learned_caps
            )
            
            print(f"Drone {drone_id} specialized in: {specialization}")
    
    def _initialize_bandit_arms(self):
        """初始化多臂老虎机模型"""
        for drone_id in range(self.num_drones):
            self.bandit_arms[drone_id] = {
                'rewards': [],
                'selections': 0,
                'estimated_reward': 0.0,
                'confidence_bound': float('inf'),
                'beta_alpha': 1.0,  # Beta分布参数 (用于Thompson采样)
                'beta_beta': 1.0
            }
    
    def allocate_tasks(self, tasks: List[Dict], drone_states: Dict) -> Dict[str, int]:
        """
        批量任务分配函数 - 用于测试和外部接口
        支持负载均衡的智能分配
        参数:
            tasks: 任务列表，每个任务包含 task_id, task_type, location 等信息
            drone_states: 无人机状态字典
        返回:
            Dict[task_id, assigned_drone_id]
        """
        allocations = {}
        available_drones = list(drone_states.keys())
        
        # 追踪每个无人机的当前负载
        current_loads = {drone_id: 0 for drone_id in available_drones}
        
        for task_data in tasks:
            if not available_drones:
                break
                
            # 转换为内部任务格式
            task = Task(
                task_id=task_data['task_id'],
                task_type=task_data.get('task_type', TaskType.EXPLORATION),
                priority=task_data.get('priority', 0.5),
                complexity=task_data.get('complexity', 0.5),
                required_capabilities=task_data.get('required_sensors', []),
                estimated_duration=task_data.get('estimated_duration', 30.0),
                location=task_data['location']
            )
            
            # 转换系统状态格式，包含当前负载信息
            current_system_state = {
                'drone_states': drone_states,
                'environmental_conditions': task_data.get('environmental_conditions', {}),
                'system_load': len(tasks) / len(drone_states) if drone_states else 1.0,
                'current_loads': current_loads.copy()
            }
            
            try:
                selected_drone, confidence = self.allocate_task(task, available_drones, current_system_state)
                
                # 应用负载均衡检查
                selected_drone = self._apply_load_balancing(
                    task, selected_drone, current_loads, available_drones, current_system_state
                )
                
                allocations[task_data['task_id']] = selected_drone
                current_loads[selected_drone] += 1
                
                # 更新分配统计
                self._update_allocation_statistics(selected_drone)
                
                print(f"Task {task_data['task_id']} assigned to Drone {selected_drone} (Load: {current_loads[selected_drone]})")
                
            except Exception as e:
                print(f"Failed to allocate task {task_data['task_id']}: {e}")
                continue
        
        # 强制分配多样化检查
        allocations = self._enforce_allocation_diversity(allocations, drone_states)
        
        return allocations
    
    def allocate_task(self, task: Task, available_drones: List[int], 
                     current_system_state: Dict) -> Tuple[int, float]:
        """
        核心任务分配函数
        返回: (选择的无人机ID, 预期性能得分)
        """
        with self.allocation_lock:
            # 1. 状态特征提取
            state_features = self._extract_state_features(task, current_system_state)
            
            # 2. 根据学习策略选择无人机
            if self.learning_strategy == LearningStrategy.EPSILON_GREEDY:
                selected_drone, confidence = self._epsilon_greedy_selection(
                    task, available_drones, state_features
                )
            elif self.learning_strategy == LearningStrategy.UCB:
                selected_drone, confidence = self._ucb_selection(
                    task, available_drones, state_features
                )
            elif self.learning_strategy == LearningStrategy.THOMPSON_SAMPLING:
                selected_drone, confidence = self._thompson_sampling_selection(
                    task, available_drones, state_features
                )
            elif self.learning_strategy == LearningStrategy.META_LEARNING:
                selected_drone, confidence = self._meta_learning_selection(
                    task, available_drones, state_features
                )
            else:
                selected_drone, confidence = self._epsilon_greedy_selection(
                    task, available_drones, state_features
                )
            
            # 3. 记录分配决策
            allocation_record = {
                'timestamp': time.time(),
                'task_id': task.task_id,
                'task_type': task.task_type.value,
                'selected_drone': selected_drone,
                'confidence': confidence,
                'available_drones': available_drones.copy(),
                'state_features': state_features.copy(),
                'learning_strategy': self.learning_strategy.value
            }
            
            self.allocation_history.append(allocation_record)
            self.performance_metrics['total_allocations'] += 1
            
            return selected_drone, confidence
    
    def _extract_state_features(self, task: Task, system_state: Dict) -> Dict:
        """提取系统状态特征"""
        features = {
            'task_priority': task.priority,
            'task_complexity': task.complexity,
            'task_type_encoding': list(TaskType).index(task.task_type),
            'system_load': system_state.get('system_load', 0.5),
            'active_drones': system_state.get('active_drones', self.num_drones),
            'current_time': time.time() % (24 * 3600),  # 时间特征
            'task_urgency': self._calculate_task_urgency(task),
        }
        
        # 添加任务位置特征
        if task.location is not None:
            features['task_x'] = task.location[0]
            features['task_y'] = task.location[1]
            features['task_z'] = task.location[2]
        
        return features
    
    def _calculate_task_urgency(self, task: Task) -> float:
        """计算任务紧急程度"""
        if task.deadline is None:
            return 0.5  # 中等紧急程度
        
        current_time = time.time()
        time_to_deadline = task.deadline - current_time
        
        if time_to_deadline <= 0:
            return 1.0  # 极其紧急
        elif time_to_deadline <= task.estimated_duration:
            return 0.9  # 非常紧急
        elif time_to_deadline <= task.estimated_duration * 2:
            return 0.7  # 紧急
        else:
            return 0.3  # 不紧急
    
    def _epsilon_greedy_selection(self, task: Task, available_drones: List[int], 
                                 state_features: Dict) -> Tuple[int, float]:
        """ε-贪心策略选择"""
        # 探索 vs 利用
        if np.random.random() < self.epsilon:
            # 探索：随机选择
            selected_drone = np.random.choice(available_drones)
            confidence = 0.5  # 探索的置信度较低
        else:
            # 利用：选择Q值最高的无人机
            best_drone = None
            best_q_value = float('-inf')
            
            state_key = self._encode_state(state_features)
            
            for drone_id in available_drones:
                q_value = self.q_table[state_key][drone_id]
                
                # 添加能力匹配奖励
                capability_match = self._calculate_capability_match(drone_id, task)
                adjusted_q_value = q_value + capability_match * 0.3
                
                if adjusted_q_value > best_q_value:
                    best_q_value = adjusted_q_value
                    best_drone = drone_id
            
            selected_drone = best_drone if best_drone is not None else np.random.choice(available_drones)
            confidence = min(1.0, max(0.0, best_q_value))
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return selected_drone, confidence
    
    def _ucb_selection(self, task: Task, available_drones: List[int], 
                      state_features: Dict) -> Tuple[int, float]:
        """Upper Confidence Bound 选择策略"""
        total_selections = sum(self.bandit_arms[drone_id]['selections'] 
                             for drone_id in available_drones)
        
        best_drone = None
        best_ucb_value = float('-inf')
        
        for drone_id in available_drones:
            arm_data = self.bandit_arms[drone_id]
            
            if arm_data['selections'] == 0:
                # 未选择过的无人机优先级最高
                ucb_value = float('inf')
            else:
                # 计算UCB值
                mean_reward = arm_data['estimated_reward']
                exploration_bonus = self.ucb_exploration_param * np.sqrt(
                    np.log(total_selections) / arm_data['selections']
                )
                
                # 添加任务匹配度
                task_match = self._calculate_capability_match(drone_id, task)
                
                ucb_value = mean_reward + exploration_bonus + task_match * 0.2
            
            if ucb_value > best_ucb_value:
                best_ucb_value = ucb_value
                best_drone = drone_id
        
        # 更新选择计数
        if best_drone is not None:
            self.bandit_arms[best_drone]['selections'] += 1
        
        confidence = min(1.0, max(0.0, best_ucb_value / 2.0))  # 规范化置信度
        
        return best_drone or np.random.choice(available_drones), confidence
    
    def _thompson_sampling_selection(self, task: Task, available_drones: List[int], 
                                   state_features: Dict) -> Tuple[int, float]:
        """Thompson采样选择策略 - 改进版本，解决性能崩塌问题"""
        best_drone = None
        best_sample = float('-inf')
        samples = {}
        current_loads = state_features.get('current_loads', {})
        
        # 多样本平均策略：减少单次采样的随机性
        num_samples = 5  # 每个arm采样5次取平均
        
        for drone_id in available_drones:
            arm_data = self.bandit_arms[drone_id]
            
            # 改进的Beta参数处理
            alpha = max(1.0, arm_data['beta_alpha'])
            beta = max(1.0, arm_data['beta_beta'])
            
            # 多次采样取平均，减少方差
            samples_list = []
            for _ in range(num_samples):
                try:
                    sample = np.random.beta(alpha, beta)
                    samples_list.append(sample)
                except (ValueError, FloatingPointError):
                    # 处理极端参数值
                    sample = alpha / (alpha + beta)  # 使用期望值作为fallback
                    samples_list.append(sample)
            
            # 使用样本均值，提高稳定性
            avg_sample = np.mean(samples_list)
            
            # 添加任务匹配奖励（权重降低，避免过度影响）
            task_match = self._calculate_capability_match(drone_id, task)
            
            # 负载均衡惩罚：高负载的无人机降低选择概率
            load_penalty = 0.0
            if current_loads and drone_id in current_loads:
                avg_load = sum(current_loads.values()) / len(current_loads) if current_loads else 0
                if current_loads[drone_id] > avg_load:
                    load_penalty = (current_loads[drone_id] - avg_load) * 0.15
            
            # 综合评分：平衡采样值、任务匹配和负载均衡
            adjusted_sample = (avg_sample * 0.7 + task_match * 0.2) - load_penalty
            
            # 确保结果在合理范围内
            adjusted_sample = max(0.0, min(1.0, adjusted_sample))
            
            samples[drone_id] = adjusted_sample
            
            if adjusted_sample > best_sample:
                best_sample = adjusted_sample
                best_drone = drone_id
        
        # 改进的置信度计算：基于Beta分布的方差和样本质量
        if len(samples) > 1:
            sample_values = list(samples.values())
            sample_mean = np.mean(sample_values)
            sample_std = np.std(sample_values)
            
            # 基于相对优势和分布稳定性计算置信度
            if sample_std > 0:
                relative_advantage = (best_sample - sample_mean) / sample_std
                confidence = min(0.95, max(0.1, 0.5 + relative_advantage * 0.2))
            else:
                confidence = 0.6  # 所有选项相同时的中等置信度
            
            # 基于Beta分布的置信度调整
            if best_drone and best_drone in self.bandit_arms:
                arm = self.bandit_arms[best_drone]
                # Beta分布的方差越小，置信度越高
                alpha = arm['beta_alpha']
                beta = arm['beta_beta']
                beta_variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
                variance_confidence = max(0.0, 1.0 - beta_variance * 5)  # 方差越小置信度越高
                confidence = (confidence + variance_confidence) / 2
        else:
            confidence = 0.4
        
        # 最终置信度限制在合理范围
        confidence = max(0.1, min(0.95, confidence))
        
        return best_drone or np.random.choice(available_drones), confidence
    
    def _meta_learning_selection(self, task: Task, available_drones: List[int], 
                                state_features: Dict) -> Tuple[int, float]:
        """元学习选择策略 - 核心创新"""
        # 1. 基于历史经验的快速适应
        similar_tasks = self._find_similar_historical_tasks(task)
        
        # 2. 跨任务知识迁移
        transferred_knowledge = self._transfer_knowledge_across_tasks(task)
        
        # 3. 计算每个无人机的元学习得分
        meta_scores = {}
        
        for drone_id in available_drones:
            # 基础Q值
            state_key = self._encode_state(state_features)
            base_q = self.q_table[state_key][drone_id]
            
            # 相似任务经验加权
            similar_task_bonus = self._calculate_similar_task_bonus(
                drone_id, similar_tasks
            )
            
            # 知识迁移奖励
            transfer_bonus = transferred_knowledge.get(drone_id, 0.0)
            
            # 快速适应能力
            adaptation_score = self._calculate_adaptation_score(drone_id, task)
            
            # 元学习综合得分
            meta_score = (
                base_q * 0.4 +
                similar_task_bonus * 0.3 +
                transfer_bonus * 0.2 +
                adaptation_score * 0.1
            )
            
            meta_scores[drone_id] = meta_score
        
        # 选择得分最高的无人机
        best_drone = max(meta_scores, key=meta_scores.get)
        best_score = meta_scores[best_drone]
        
        # 计算置信度
        if len(meta_scores) > 1:
            scores = list(meta_scores.values())
            confidence = (best_score - np.mean(scores)) / (np.std(scores) + 1e-6)
            confidence = min(1.0, max(0.0, confidence))
        else:
            confidence = 0.8
        
        return best_drone, confidence
    
    def _find_similar_historical_tasks(self, target_task: Task) -> List[Dict]:
        """查找相似的历史任务"""
        similar_tasks = []
        
        for record in self.allocation_history[-100:]:  # 只考虑最近100个任务
            similarity = self._calculate_task_similarity(target_task, record)
            if similarity > 0.7:  # 相似度阈值
                similar_tasks.append({
                    'record': record,
                    'similarity': similarity
                })
        
        # 按相似度排序
        similar_tasks.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_tasks[:10]  # 返回最相似的10个
    
    def _calculate_task_similarity(self, task1: Task, historical_record: Dict) -> float:
        """计算任务相似度"""
        # 任务类型匹配
        type_match = 1.0 if task1.task_type.value == historical_record['task_type'] else 0.3
        
        # 优先级相似度
        priority_diff = abs(task1.priority - historical_record.get('task_priority', 0.5))
        priority_similarity = 1.0 - priority_diff
        
        # 复杂度相似度
        complexity_diff = abs(task1.complexity - historical_record.get('task_complexity', 0.5))
        complexity_similarity = 1.0 - complexity_diff
        
        # 综合相似度
        similarity = (
            type_match * 0.5 +
            priority_similarity * 0.3 +
            complexity_similarity * 0.2
        )
        
        return max(0.0, min(1.0, similarity))
    
    def _transfer_knowledge_across_tasks(self, target_task: Task) -> Dict[int, float]:
        """跨任务知识迁移"""
        transfer_scores = {}
        
        target_type_idx = list(TaskType).index(target_task.task_type)
        
        for drone_id in range(self.num_drones):
            transfer_score = 0.0
            
            # 基于任务类型间的迁移矩阵
            for source_type_idx, source_type in enumerate(TaskType):
                if drone_id in self.drone_capabilities:
                    source_experience = self.drone_capabilities[drone_id].experience_count.get(source_type, 0)
                    if source_experience > 0:
                        transfer_weight = self.cross_task_transfer_matrix[source_type_idx][target_type_idx]
                        transfer_score += source_experience * transfer_weight
            
            transfer_scores[drone_id] = min(1.0, transfer_score / 10.0)  # 规范化
        
        return transfer_scores
    
    def _calculate_similar_task_bonus(self, drone_id: int, similar_tasks: List[Dict]) -> float:
        """计算相似任务经验奖励"""
        if not similar_tasks:
            return 0.0
        
        bonus = 0.0
        total_weight = 0.0
        
        for task_info in similar_tasks:
            record = task_info['record']
            similarity = task_info['similarity']
            
            if record['selected_drone'] == drone_id:
                # 该无人机在相似任务中的表现
                performance = record.get('performance', 0.5)  # 默认性能
                weighted_performance = performance * similarity
                
                bonus += weighted_performance
                total_weight += similarity
        
        return bonus / total_weight if total_weight > 0 else 0.0
    
    def _calculate_adaptation_score(self, drone_id: int, task: Task) -> float:
        """计算快速适应能力得分"""
        if drone_id not in self.drone_capabilities:
            return 0.5
        
        capability = self.drone_capabilities[drone_id]
        
        # 基于学习速度和适应性
        learning_speed = capability.base_capabilities.get('learning_speed', 0.5)
        adaptability = capability.base_capabilities.get('adaptability', 0.5)
        
        # 经验多样性奖励
        experience_diversity = len([t for t, count in capability.experience_count.items() if count > 0])
        diversity_bonus = min(0.3, experience_diversity * 0.05)
        
        adaptation_score = (learning_speed + adaptability) / 2 + diversity_bonus
        
        return min(1.0, adaptation_score)
    
    def _calculate_capability_match(self, drone_id: int, task: Task) -> float:
        """计算无人机能力与任务需求的匹配度"""
        if drone_id not in self.drone_capabilities:
            return 0.5
        
        capability = self.drone_capabilities[drone_id]
        match_score = 0.0
        
        # 基于任务类型的能力需求
        required_capabilities = self._get_task_capability_requirements(task)
        
        for cap_name, importance in required_capabilities.items():
            drone_capability = capability.learned_capabilities.get(cap_name, 0.5)
            match_score += drone_capability * importance
        
        # 基于历史成功率
        historical_success = capability.success_rate.get(task.task_type, 0.5)
        match_score += historical_success * 0.3
        
        return min(1.0, match_score)
    
    def _get_task_capability_requirements(self, task: Task) -> Dict[str, float]:
        """获取任务类型对各种能力的需求权重"""
        requirements = {
            TaskType.EXPLORATION: {
                'speed': 0.8, 'endurance': 0.9, 'sensor_quality': 0.7,
                'navigation_accuracy': 0.8, 'obstacle_avoidance': 0.6
            },
            TaskType.MAPPING: {
                'sensor_quality': 0.9, 'navigation_accuracy': 0.9,
                'endurance': 0.7, 'speed': 0.5
            },
            TaskType.SURVEILLANCE: {
                'sensor_quality': 0.9, 'endurance': 0.8, 'communication_range': 0.7,
                'obstacle_avoidance': 0.6
            },
            TaskType.TRANSPORTATION: {
                'payload_capacity': 0.9, 'navigation_accuracy': 0.8,
                'speed': 0.7, 'endurance': 0.6
            },
            TaskType.EMERGENCY_RESPONSE: {
                'speed': 0.9, 'adaptability': 0.8, 'communication_range': 0.7,
                'cooperative_ability': 0.6
            }
        }
        
        return requirements.get(task.task_type, {})
    
    def _encode_state(self, state_features: Dict) -> str:
        """将状态特征编码为字符串键"""
        # 量化连续特征以减少状态空间
        quantized_features = {}
        
        for key, value in state_features.items():
            if isinstance(value, float):
                # 将浮点数量化为离散值
                quantized_features[key] = round(value, 2)
            elif isinstance(value, np.ndarray):
                # 将numpy数组转换为列表
                quantized_features[key] = [round(float(x), 2) for x in value.flatten()]
            elif isinstance(value, (int, str, bool)):
                quantized_features[key] = value
            elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                # 处理其他可迭代对象
                try:
                    quantized_features[key] = list(value)
                except:
                    quantized_features[key] = str(value)
            else:
                # 其他类型转换为字符串
                quantized_features[key] = str(value)
        
        return json.dumps(quantized_features, sort_keys=True)
    
    def update_learning(self, task_id: str, selected_drone: int, 
                       actual_performance: float, task_duration: float,
                       success: bool, final_state: Dict):
        """更新学习模型 - 核心学习函数"""
        with self.allocation_lock:
            # 应用学习平滑化
            if selected_drone in self.bandit_arms and self.bandit_arms[selected_drone]['rewards']:
                smoothed_performance = self._apply_learning_smoothing(
                    actual_performance, self.bandit_arms[selected_drone]['rewards'][-10:]
                )
            else:
                smoothed_performance = actual_performance
            
            # 1. 更新Q-learning
            self._update_q_learning(task_id, selected_drone, smoothed_performance, final_state)
            
            # 2. 更新Multi-Armed Bandit
            self._update_bandit_arms(selected_drone, smoothed_performance)
            
            # 3. 更新无人机能力模型
            self._update_drone_capabilities(selected_drone, task_id, smoothed_performance, 
                                          task_duration, success)
            
            # 4. 更新经验回放缓存
            self._update_experience_replay(task_id, selected_drone, actual_performance, 
                                         task_duration, success, final_state)
            
            # 5. 更新跨任务迁移矩阵
            self._update_transfer_matrix(task_id, selected_drone, actual_performance)
            
            # 6. 性能指标更新
            self._update_performance_metrics(success, actual_performance)
            
            # 动态调整学习策略
            self._update_learning_strategy()
            
            print(f"Learning updated for Task {task_id}, Drone {selected_drone}, "
                  f"Performance: {actual_performance:.3f}, Success: {success}, "
                  f"Strategy: {self.learning_strategy.value}, ε: {self.epsilon:.3f}")
    
    def _update_q_learning(self, task_id: str, selected_drone: int, 
                          reward: float, final_state: Dict):
        """更新Q-learning表"""
        # 找到对应的分配记录
        allocation_record = None
        for record in reversed(self.allocation_history):
            if record['task_id'] == task_id and record['selected_drone'] == selected_drone:
                allocation_record = record
                break
        
        if allocation_record is None:
            return
        
        # 获取状态键
        state_key = self._encode_state(allocation_record['state_features'])
        next_state_key = self._encode_state(final_state)
        
        # Q-learning更新公式
        current_q = self.q_table[state_key][selected_drone]
        
        # 计算下一状态的最大Q值
        next_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
        
        # Q值更新
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        
        self.q_table[state_key][selected_drone] = new_q
    
    def _update_bandit_arms(self, selected_drone: int, reward: float):
        """更新Multi-Armed Bandit模型"""
        arm_data = self.bandit_arms[selected_drone]
        
        # 更新奖励历史
        arm_data['rewards'].append(reward)
        if len(arm_data['rewards']) > 100:  # 保持最近100个奖励
            arm_data['rewards'] = arm_data['rewards'][-100:]
        
        # 更新估计奖励
        arm_data['estimated_reward'] = np.mean(arm_data['rewards'])
        
        # 更新Thompson采样的Beta分布参数 - 改进版本
        # 使用奖励值作为成功概率更新，而不是简单的阈值判断
        success_weight = max(0.0, min(1.0, reward * 2.0))  # 将[0,1] reward映射到更敏感的范围
        failure_weight = 1.0 - success_weight
        
        # 渐进式更新Beta参数
        learning_rate_beta = 0.1
        arm_data['beta_alpha'] += success_weight * learning_rate_beta
        arm_data['beta_beta'] += failure_weight * learning_rate_beta
        
        # 防止参数过小导致极端采样
        arm_data['beta_alpha'] = max(0.1, arm_data['beta_alpha'])
        arm_data['beta_beta'] = max(0.1, arm_data['beta_beta'])
    
    def _update_drone_capabilities(self, drone_id: int, task_id: str, 
                                 performance: float, duration: float, success: bool):
        """更新无人机能力模型 - 改进版本支持实际技能提升"""
        if drone_id not in self.drone_capabilities:
            return
        
        # 找到任务信息
        task_info = None
        for record in reversed(self.allocation_history):
            if record['task_id'] == task_id:
                task_type = TaskType(record['task_type'])
                task_info = task_type
                break
        
        if task_info is None:
            return
        
        capability = self.drone_capabilities[drone_id]
        
        # 更新经验计数
        capability.experience_count[task_info] += 1
        
        # 更新成功率 (指数移动平均)
        current_success_rate = capability.success_rate.get(task_info, 0.5)
        success_value = 1.0 if success else 0.0
        new_success_rate = (
            current_success_rate * 0.9 + success_value * 0.1
        )
        capability.success_rate[task_info] = new_success_rate
        
        # 更新平均完成时间
        current_avg_time = capability.avg_completion_time.get(task_info, duration)
        new_avg_time = current_avg_time * 0.8 + duration * 0.2
        capability.avg_completion_time[task_info] = new_avg_time
        
        # 改进的能力学习更新机制
        # 性能因子：将[0,1]映射到[-0.2, +0.2]的学习调整范围
        performance_factor = (performance - 0.5) * 0.4  # 最大±20%调整
        
        # 成功奖励机制：成功的任务获得额外奖励
        success_bonus = 0.05 if success else -0.02  # 成功+5%，失败-2%
        
        # 经验加成：更多经验的任务类型学习效率更高
        experience_multiplier = min(2.0, 1.0 + capability.experience_count[task_info] * 0.1)
        
        required_caps = self._get_task_capability_requirements_by_type(task_info)
        for cap_name, importance in required_caps.items():
            if cap_name in capability.learned_capabilities:
                current_value = capability.learned_capabilities[cap_name]
                
                # 综合调整：性能 + 成功奖励 + 经验加成
                total_adjustment = (performance_factor + success_bonus) * importance * experience_multiplier
                
                # 防止过度调整：单次最大±15%
                total_adjustment = max(-0.15, min(0.15, total_adjustment))
                
                new_value = max(0.1, min(1.0, current_value + total_adjustment))
                capability.learned_capabilities[cap_name] = new_value
                
                # 调试输出：记录关键学习更新
                if abs(total_adjustment) > 0.02:  # 只记录显著变化
                    print(f"Skill Update: Drone {drone_id} {cap_name}: {current_value:.3f} -> {new_value:.3f} ({total_adjustment:+.3f})")
    
    def _get_task_capability_requirements_by_type(self, task_type: TaskType) -> Dict[str, float]:
        """根据任务类型获取能力需求"""
        requirements = {
            TaskType.EXPLORATION: {
                'speed': 0.8, 'endurance': 0.9, 'sensor_quality': 0.7,
                'navigation_accuracy': 0.8, 'obstacle_avoidance': 0.6
            },
            TaskType.MAPPING: {
                'sensor_quality': 0.9, 'navigation_accuracy': 0.9,
                'endurance': 0.7, 'speed': 0.5
            },
            TaskType.SURVEILLANCE: {
                'sensor_quality': 0.9, 'endurance': 0.8, 'communication_range': 0.7,
                'obstacle_avoidance': 0.6
            },
            TaskType.TRANSPORTATION: {
                'payload_capacity': 0.9, 'navigation_accuracy': 0.8,
                'speed': 0.7, 'endurance': 0.6
            },
            TaskType.EMERGENCY_RESPONSE: {
                'speed': 0.9, 'adaptability': 0.8, 'communication_range': 0.7,
                'cooperative_ability': 0.6
            }
        }
        
        return requirements.get(task_type, {})
    
    def _update_experience_replay(self, task_id: str, selected_drone: int,
                                performance: float, duration: float, success: bool,
                                final_state: Dict):
        """更新经验回放缓存"""
        # 找到对应的分配记录
        allocation_record = None
        for record in reversed(self.allocation_history):
            if record['task_id'] == task_id:
                allocation_record = record
                break
        
        if allocation_record is None:
            return
        
        # 创建经验条目
        experience = AllocationExperience(
            state=allocation_record['state_features'],
            action=selected_drone,
            reward=performance,
            next_state=final_state,
            task_info=None,  # 简化，实际应该包含完整任务信息
            timestamp=time.time(),
            actual_duration=duration,
            success=success
        )
        
        self.experience_replay_buffer.append(experience)
        
        # 定期进行经验回放学习
        if len(self.experience_replay_buffer) >= 50 and len(self.experience_replay_buffer) % 10 == 0:
            self._experience_replay_learning()
    
    def _experience_replay_learning(self):
        """基于经验回放的批量学习"""
        if len(self.experience_replay_buffer) < 10:
            return
        
        # 随机采样经验进行批量学习
        sample_size = min(32, len(self.experience_replay_buffer))
        experiences = np.random.choice(
            list(self.experience_replay_buffer), 
            size=sample_size, 
            replace=False
        )
        
        # 批量Q-learning更新
        for exp in experiences:
            state_key = self._encode_state(exp.state)
            next_state_key = self._encode_state(exp.next_state)
            
            current_q = self.q_table[state_key][exp.action]
            next_max_q = max(self.q_table[next_state_key].values()) if self.q_table[next_state_key] else 0.0
            
            # 使用更小的学习率进行经验回放学习
            replay_learning_rate = self.learning_rate * 0.5
            new_q = current_q + replay_learning_rate * (
                exp.reward + self.discount_factor * next_max_q - current_q
            )
            
            self.q_table[state_key][exp.action] = new_q
    
    def _update_transfer_matrix(self, task_id: str, selected_drone: int, performance: float):
        """更新跨任务迁移矩阵"""
        # 找到任务类型
        task_type = None
        for record in reversed(self.allocation_history):
            if record['task_id'] == task_id:
                task_type = TaskType(record['task_type'])
                break
        
        if task_type is None:
            return
        
        target_idx = list(TaskType).index(task_type)
        
        # 基于表现更新迁移权重
        if selected_drone in self.drone_capabilities:
            capability = self.drone_capabilities[selected_drone]
            
            for source_type, experience_count in capability.experience_count.items():
                if experience_count > 0 and source_type != task_type:
                    source_idx = list(TaskType).index(source_type)
                    
                    # 如果表现好，增强迁移权重
                    if performance > 0.7:
                        self.cross_task_transfer_matrix[source_idx][target_idx] *= 1.01
                    elif performance < 0.3:
                        self.cross_task_transfer_matrix[source_idx][target_idx] *= 0.99
                    
                    # 确保权重在合理范围内
                    self.cross_task_transfer_matrix[source_idx][target_idx] = np.clip(
                        self.cross_task_transfer_matrix[source_idx][target_idx], 0.01, 0.5
                    )
    
    def _update_performance_metrics(self, success: bool, performance: float):
        """更新性能指标"""
        if success:
            self.performance_metrics['successful_allocations'] += 1
        
        # 更新平均分配时间（这里简化处理）
        current_avg = self.performance_metrics['avg_allocation_time']
        new_avg = current_avg * 0.9 + 0.1 * 1.0  # 假设分配时间为1秒
        self.performance_metrics['avg_allocation_time'] = new_avg
        
        # 计算学习收敛性
        recent_performances = [exp.reward for exp in list(self.experience_replay_buffer)[-20:]]
        if len(recent_performances) >= 10:
            convergence = 1.0 - np.std(recent_performances)
            self.performance_metrics['learning_convergence'] = max(0.0, convergence)
    
    def get_learning_report(self) -> Dict:
        """获取详细的学习报告"""
        report = {
            'timestamp': time.time(),
            'learning_strategy': self.learning_strategy.value,
            'performance_metrics': self.performance_metrics.copy(),
            'q_table_size': len(self.q_table),
            'experience_buffer_size': len(self.experience_replay_buffer),
            'epsilon': self.epsilon,
            'drone_capabilities': {},
            'transfer_matrix': self.cross_task_transfer_matrix.tolist(),
            'bandit_arms_status': {}
        }
        
        # 无人机能力报告
        for drone_id, capability in self.drone_capabilities.items():
            report['drone_capabilities'][drone_id] = {
                'experience_count': {t.value: count for t, count in capability.experience_count.items()},
                'success_rates': {t.value: rate for t, rate in capability.success_rate.items()},
                'learned_capabilities': capability.learned_capabilities.copy(),
                'total_experience': sum(capability.experience_count.values())
            }
        
        # Bandit Arms状态报告
        for drone_id, arm_data in self.bandit_arms.items():
            report['bandit_arms_status'][drone_id] = {
                'selections': arm_data['selections'],
                'estimated_reward': arm_data['estimated_reward'],
                'recent_rewards': arm_data['rewards'][-10:] if arm_data['rewards'] else [],
                'beta_parameters': (arm_data['beta_alpha'], arm_data['beta_beta'])
            }
        
        return report
    
    def save_learning_model(self, filepath: str):
        """保存学习模型"""
        model_data = {
            'q_table': dict(self.q_table),
            'drone_capabilities': self.drone_capabilities,
            'bandit_arms': self.bandit_arms,
            'cross_task_transfer_matrix': self.cross_task_transfer_matrix,
            'experience_replay_buffer': list(self.experience_replay_buffer),
            'performance_metrics': self.performance_metrics,
            'epsilon': self.epsilon
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Learning model saved to {filepath}")
    
    def load_learning_model(self, filepath: str):
        """加载学习模型"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_table = defaultdict(lambda: defaultdict(float), model_data['q_table'])
            self.drone_capabilities = model_data['drone_capabilities']
            self.bandit_arms = model_data['bandit_arms']
            self.cross_task_transfer_matrix = model_data['cross_task_transfer_matrix']
            self.experience_replay_buffer = deque(model_data['experience_replay_buffer'], maxlen=10000)
            self.performance_metrics = model_data['performance_metrics']
            self.epsilon = model_data['epsilon']
            
            print(f"Learning model loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to load learning model: {e}")
            return False
    
    def get_performance_statistics(self) -> Dict:
        """获取ALTA性能统计信息"""
        with self.allocation_lock:
            # 计算总体统计
            total_allocations = sum(self.performance_metrics.get('allocations', {}).values())
            learning_episodes = len(self.experience_replay_buffer)
            
            # 计算平均预测准确率 - 基于实际的学习经验
            prediction_accuracies = []
            for exp in self.experience_replay_buffer[-50:]:  # 最近50次经验
                if hasattr(exp, 'reward') and hasattr(exp, 'success'):
                    # 预测准确性 = 实际成功率与预期的匹配度
                    expected_success = 1.0 if exp.reward > 0.5 else 0.0
                    actual_success = 1.0 if exp.success else 0.0
                    accuracy = 1.0 - abs(expected_success - actual_success)
                    prediction_accuracies.append(accuracy)
            
            avg_prediction_accuracy = np.mean(prediction_accuracies) if prediction_accuracies else 0.0
            
            # 计算平均置信度
            avg_confidence = 0.0
            confidence_count = 0
            for drone_id in range(self.num_drones):
                if drone_id in self.bandit_arms:
                    if self.bandit_arms[drone_id]['selections'] > 0:
                        avg_confidence += self.bandit_arms[drone_id]['estimated_reward']
                        confidence_count += 1
            
            if confidence_count > 0:
                avg_confidence /= confidence_count
            
            # 计算适应率（基于最近的学习活动）
            recent_episodes = min(100, len(self.experience_replay_buffer))
            if recent_episodes > 0:
                recent_performances = [exp.reward for exp in list(self.experience_replay_buffer)[-recent_episodes:]]
                if len(recent_performances) > 10:
                    early_avg = np.mean(recent_performances[:10])
                    late_avg = np.mean(recent_performances[-10:])
                    adaptation_rate = max(0.0, (late_avg - early_avg) / max(early_avg, 0.1))
                else:
                    adaptation_rate = 0.0
            else:
                adaptation_rate = 0.0
            
            # 无人机能力统计
            drone_capabilities_stats = {}
            for drone_id, capabilities in self.drone_capabilities.items():
                # 计算每个无人机的经验数量
                drone_experience_count = len([exp for exp in self.experience_replay_buffer if exp.action == drone_id])
                
                # 获取探索效率技能等级 - 使用正确的键名
                # 从学习能力和基础能力中获取，优先使用学习结果
                exploration_learned = capabilities.learned_capabilities.get('exploration_efficiency', 
                                    capabilities.learned_capabilities.get('endurance', 0.0))
                exploration_base = capabilities.base_capabilities.get('exploration_efficiency',
                                 capabilities.base_capabilities.get('endurance', 0.5))
                
                # 学习后的技能 = 基础技能 + 学习增益，但不超过1.0
                exploration_skill = min(1.0, exploration_base + (exploration_learned - exploration_base) * 0.5)
                
                # 获取专长信息 - 从基础能力中找到最高的能力作为专长
                specialization = 'general'
                max_capability_value = 0.0
                for cap_name, cap_value in capabilities.base_capabilities.items():
                    if cap_value > max_capability_value:
                        max_capability_value = cap_value
                        specialization = cap_name
                
                drone_capabilities_stats[str(drone_id)] = {
                    'exploration': {
                        'skill_level': exploration_skill,
                        'experience_count': drone_experience_count
                    },
                    'specialization': specialization,
                    'total_tasks': self.performance_metrics.get('allocations', {}).get(drone_id, 0),
                    'success_rates': {task_type.value: rate for task_type, rate in capabilities.success_rate.items()},
                    'avg_completion_times': {task_type.value: time for task_type, time in capabilities.avg_completion_time.items()}
                }
            
            return {
                'total_allocations': total_allocations,
                'learning_episodes': learning_episodes,
                'average_prediction_accuracy': avg_prediction_accuracy,
                'average_confidence': avg_confidence,
                'adaptation_rate': adaptation_rate,
                'drone_capabilities': drone_capabilities_stats,
                'learning_strategy': self.learning_strategy.value,
                'epsilon': self.epsilon,
                'q_table_size': len(self.q_table),
                'cross_task_transfer_entries': len(self.cross_task_transfer_matrix)
            }
    
    def _apply_load_balancing(self, task: Task, selected_drone: int, 
                            current_loads: Dict[int, int], available_drones: List[int],
                            system_state: Dict) -> int:
        """应用负载均衡策略"""
        if len(available_drones) <= 1:
            return selected_drone
        
        # 计算负载均衡惩罚
        avg_load = sum(current_loads.values()) / len(current_loads)
        current_drone_load = current_loads[selected_drone]
        
        # 如果选定的无人机负载过高，尝试重新分配
        if current_drone_load > avg_load + 2:  # 负载阈值
            # 找到负载最轻的无人机
            min_load_drone = min(current_loads.items(), key=lambda x: x[1])[0]
            
            # 如果负载差异显著，重新分配
            if current_loads[min_load_drone] < current_drone_load - 1:
                print(f"Load balancing: Reassigning from Drone {selected_drone} (load:{current_drone_load}) to Drone {min_load_drone} (load:{current_loads[min_load_drone]})")
                return min_load_drone
        
        return selected_drone
    
    def _enforce_allocation_diversity(self, allocations: Dict[str, int], 
                                    drone_states: Dict) -> Dict[str, int]:
        """强制分配多样化 - 确保所有无人机都参与"""
        if len(allocations) == 0 or len(drone_states) <= 1:
            return allocations
        
        # 统计每个无人机的任务数量
        drone_task_counts = {drone_id: 0 for drone_id in drone_states.keys()}
        for task_id, drone_id in allocations.items():
            drone_task_counts[drone_id] += 1
        
        # 检查是否有无人机完全没有任务
        zero_task_drones = [drone_id for drone_id, count in drone_task_counts.items() if count == 0]
        
        if zero_task_drones:
            print(f"Enforcing diversity: {len(zero_task_drones)} drones with zero tasks")
            
            # 找到任务最多的无人机
            max_task_drone = max(drone_task_counts.items(), key=lambda x: x[1])[0]
            max_tasks = drone_task_counts[max_task_drone]
            
            # 重新分配部分任务
            tasks_to_redistribute = list(allocations.items())
            redistribute_count = min(len(zero_task_drones), max_tasks // 2)
            
            for i, zero_drone in enumerate(zero_task_drones[:redistribute_count]):
                # 从负载最重的无人机那里转移任务
                for task_id, current_drone in tasks_to_redistribute:
                    if current_drone == max_task_drone:
                        allocations[task_id] = zero_drone
                        drone_task_counts[max_task_drone] -= 1
                        drone_task_counts[zero_drone] += 1
                        print(f"Diversity: Reassigned {task_id} from Drone {max_task_drone} to Drone {zero_drone}")
                        break
        
        return allocations
    
    def _update_allocation_statistics(self, drone_id: int):
        """更新分配统计"""
        if 'allocations' not in self.performance_metrics:
            self.performance_metrics['allocations'] = {}
        
        if drone_id not in self.performance_metrics['allocations']:
            self.performance_metrics['allocations'][drone_id] = 0
        
        self.performance_metrics['allocations'][drone_id] += 1
    
    def _update_learning_strategy(self):
        """动态更新学习策略"""
        total_episodes = len(self.experience_replay_buffer)
        
        # 基于学习进度动态选择策略
        if total_episodes < 30:
            # 初期：更多探索
            self.learning_strategy = LearningStrategy.EPSILON_GREEDY
            self.epsilon = max(0.2, self.epsilon)
        elif total_episodes < 80:
            # 中期：平衡探索和利用
            self.learning_strategy = LearningStrategy.UCB
            self.epsilon = max(0.15, self.epsilon * 0.998)
        else:
            # 后期：精细调优
            self.learning_strategy = LearningStrategy.THOMPSON_SAMPLING
            self.epsilon = max(0.1, self.epsilon * 0.998)
    
    def _apply_learning_smoothing(self, new_performance: float, historical_performances: List[float]) -> float:
        """平滑学习更新，减少波动"""
        if not historical_performances:
            return new_performance
        
        # 指数移动平均
        smoothing_factor = 0.7
        recent_avg = np.mean(historical_performances[-5:])  # 最近5次的平均
        smoothed_performance = smoothing_factor * recent_avg + (1 - smoothing_factor) * new_performance
        
        return smoothed_performance
    
    def _calculate_confidence(self, drone_id: int, task_context: Dict) -> float:
        """计算分配置信度"""
        if drone_id not in self.bandit_arms:
            return 0.3  # 低置信度
        
        arm = self.bandit_arms[drone_id]
        
        # 基于选择次数和奖励估计计算置信度
        if arm['selections'] == 0:
            return 0.3
        elif arm['selections'] < 5:
            return 0.5  # 中等置信度
        else:
            # 基于奖励方差计算置信度
            estimated_reward = arm['estimated_reward']
            confidence = min(0.9, 0.3 + estimated_reward * 0.6)
            return confidence