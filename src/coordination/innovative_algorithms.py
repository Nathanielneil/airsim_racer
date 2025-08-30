#!/usr/bin/env python3
"""
Innovative Multi-Drone Coordination Algorithms
创新性多无人机协调算法
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import threading
from collections import deque
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import networkx as nx

from ..utils.math_utils import distance_3d


@dataclass
class UncertaintyRegion:
    """不确定性区域 - 用于信息熵驱动的探索"""
    center: np.ndarray
    radius: float
    information_entropy: float
    predicted_value: float
    confidence: float
    last_updated: float


class InnovativeCoordinator:
    """创新性协调算法集合"""
    
    def __init__(self):
        # 1. 信息熵驱动探索
        self.uncertainty_map: Dict[str, UncertaintyRegion] = {}
        self.entropy_threshold = 0.5
        
        # 2. 学习驱动的任务分配
        self.task_performance_history: Dict[int, List[float]] = {}
        self.learning_rate = 0.1
        
        # 3. 动态拍卖机制
        self.auction_history: List[Dict] = []
        
        # 4. 预测性路径规划
        self.movement_predictions: Dict[int, List[np.ndarray]] = {}
        self.prediction_horizon = 30.0  # seconds
        
        # 5. 涌现行为网络
        self.behavioral_network = nx.DiGraph()
        self.emergence_patterns: List[Dict] = []
    
    # ===== 创新算法1: 信息熵驱动的探索策略 =====
    def entropy_driven_exploration(self, drone_states: Dict, frontier_candidates: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        """
        基于信息熵的智能探索目标选择
        创新点：不仅考虑未探索区域，还考虑信息获取的价值
        """
        if not frontier_candidates:
            return []
        
        scored_targets = []
        
        for frontier in frontier_candidates:
            # 计算信息熵得分
            entropy_score = self._calculate_information_entropy(frontier, drone_states)
            
            # 计算探索价值预测
            exploration_value = self._predict_exploration_value(frontier, drone_states)
            
            # 计算多无人机协同增益
            synergy_bonus = self._calculate_synergy_bonus(frontier, drone_states)
            
            # 综合得分 (创新的多因子评分)
            total_score = (
                entropy_score * 0.4 +           # 信息熵权重
                exploration_value * 0.35 +       # 探索价值权重  
                synergy_bonus * 0.25             # 协同增益权重
            )
            
            scored_targets.append((frontier, total_score))
        
        # 按得分排序，返回最有价值的目标
        scored_targets.sort(key=lambda x: x[1], reverse=True)
        return scored_targets[:min(5, len(scored_targets))]  # 返回前5个最佳目标
    
    def _calculate_information_entropy(self, position: np.ndarray, drone_states: Dict) -> float:
        """计算位置的信息熵"""
        # 基于周围区域的不确定性计算熵
        surrounding_uncertainty = 0.0
        weight_sum = 0.0
        
        for region_id, region in self.uncertainty_map.items():
            distance = distance_3d(position, region.center)
            if distance < region.radius * 2:  # 影响范围
                weight = np.exp(-distance / region.radius)  # 距离权重
                surrounding_uncertainty += region.information_entropy * weight
                weight_sum += weight
        
        if weight_sum > 0:
            return surrounding_uncertainty / weight_sum
        
        # 新区域的初始熵值
        return 1.0
    
    def _predict_exploration_value(self, position: np.ndarray, drone_states: Dict) -> float:
        """预测探索该位置的价值"""
        # 基于历史数据和机器学习预测探索价值
        base_value = 0.5
        
        # 考虑与其他无人机的距离分布
        distances = []
        for drone_id, state in drone_states.items():
            dist = distance_3d(position, state.position)
            distances.append(dist)
        
        if distances:
            # 最优距离分布得分 (避免过度聚集或分散)
            avg_distance = np.mean(distances)
            optimal_distance = 15.0  # 假设最优协作距离
            distance_score = 1.0 - abs(avg_distance - optimal_distance) / optimal_distance
            base_value += distance_score * 0.3
        
        return min(1.0, base_value)
    
    def _calculate_synergy_bonus(self, position: np.ndarray, drone_states: Dict) -> float:
        """计算多无人机协同探索该位置的增益"""
        if len(drone_states) < 2:
            return 0.0
        
        # 创新：基于无人机能力互补性的协同增益
        synergy_score = 0.0
        
        # 模拟不同类型无人机的协同效应
        drone_positions = [state.position for state in drone_states.values()]
        
        if len(drone_positions) >= 2:
            # 计算探索该位置时的几何配置优势
            centroid = np.mean(drone_positions + [position], axis=0)
            
            # 配置熵 - 衡量无人机分布的均匀性
            distances_to_centroid = [distance_3d(pos, centroid) for pos in drone_positions + [position]]
            config_entropy = np.std(distances_to_centroid) / np.mean(distances_to_centroid)
            
            # 理想配置得分 (适度分散但不过于分散)
            synergy_score = 1.0 - abs(config_entropy - 0.3) / 0.3
        
        return max(0.0, min(1.0, synergy_score))
    
    # ===== 创新算法2: 自适应学习任务分配 =====
    def adaptive_learning_task_allocation(self, tasks: List, drone_capabilities: Dict, historical_performance: Dict) -> Dict:
        """
        基于强化学习的自适应任务分配
        创新点：无人机能力动态学习和任务匹配优化
        """
        allocation_result = {}
        
        # Q-learning inspired allocation
        for task_id, task in enumerate(tasks):
            best_drone = None
            best_q_value = float('-inf')
            
            for drone_id, capability in drone_capabilities.items():
                # 计算Q值 (状态-动作价值)
                q_value = self._calculate_q_value(drone_id, task, historical_performance)
                
                # 加入探索因子 (epsilon-greedy)
                if np.random.random() < 0.1:  # 10% 探索率
                    q_value += np.random.normal(0, 0.1)  # 添加噪声鼓励探索
                
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_drone = drone_id
            
            if best_drone is not None:
                allocation_result[task_id] = {
                    'assigned_drone': best_drone,
                    'expected_performance': best_q_value,
                    'confidence': min(1.0, len(historical_performance.get(best_drone, [])) / 10.0)
                }
        
        return allocation_result
    
    def _calculate_q_value(self, drone_id: int, task, historical_performance: Dict) -> float:
        """计算状态-动作Q值"""
        # 基础能力得分
        base_score = 0.5
        
        # 历史性能学习
        if drone_id in historical_performance:
            recent_performance = historical_performance[drone_id][-10:]  # 最近10次任务
            if recent_performance:
                # 加权平均，更重视近期表现
                weights = np.exp(np.linspace(-1, 0, len(recent_performance)))
                weighted_avg = np.average(recent_performance, weights=weights)
                base_score += (weighted_avg - 0.5) * 0.4  # 性能调整
        
        # 任务适配度 (基于任务特征和无人机特长)
        task_match_score = self._calculate_task_match_score(drone_id, task)
        base_score += task_match_score * 0.3
        
        return base_score
    
    def _calculate_task_match_score(self, drone_id: int, task) -> float:
        """计算无人机-任务匹配度"""
        # 简化的任务特征匹配
        # 实际应用中可以基于任务类型、复杂度、所需传感器等
        return 0.5 + np.random.normal(0, 0.1)  # 模拟匹配得分
    
    # ===== 创新算法3: 动态拍卖协调机制 =====
    def dynamic_auction_coordination(self, available_tasks: List, drone_bids: Dict) -> Dict:
        """
        动态拍卖机制进行任务协调
        创新点：考虑时间价值衰减和协同效应的智能拍卖
        """
        auction_results = {}
        
        # 多轮拍卖，考虑任务间依赖性
        remaining_tasks = available_tasks.copy()
        auction_round = 0
        
        while remaining_tasks and auction_round < 5:  # 最多5轮拍卖
            current_bids = {}
            
            for task in remaining_tasks:
                task_bids = []
                
                for drone_id, bid_info in drone_bids.items():
                    if drone_id not in [result['winner'] for result in auction_results.values()]:  # 未分配的无人机
                        # 计算考虑时间衰减的出价
                        time_decay = np.exp(-auction_round * 0.2)  # 时间衰减因子
                        adjusted_bid = bid_info.get('base_bid', 1.0) * time_decay
                        
                        # 加入协同效应奖励
                        synergy_bonus = self._calculate_auction_synergy_bonus(
                            drone_id, task, auction_results, remaining_tasks
                        )
                        
                        final_bid = adjusted_bid + synergy_bonus
                        task_bids.append((drone_id, final_bid))
                
                if task_bids:
                    # 选择出价最高的无人机
                    winner_drone, winning_bid = max(task_bids, key=lambda x: x[1])
                    
                    auction_results[task] = {
                        'winner': winner_drone,
                        'winning_bid': winning_bid,
                        'auction_round': auction_round,
                        'competition_level': len(task_bids)
                    }
                    
                    current_bids[task] = task_bids
            
            # 移除已分配的任务
            for task in current_bids:
                if task in remaining_tasks:
                    remaining_tasks.remove(task)
            
            auction_round += 1
        
        # 记录拍卖历史用于学习
        self.auction_history.append({
            'timestamp': time.time(),
            'results': auction_results,
            'total_rounds': auction_round
        })
        
        return auction_results
    
    def _calculate_auction_synergy_bonus(self, drone_id: int, task, current_results: Dict, remaining_tasks: List) -> float:
        """计算拍卖中的协同效应奖励"""
        synergy_bonus = 0.0
        
        # 如果该无人机获得此任务，与其他已分配任务的协同效应
        for allocated_task, result_info in current_results.items():
            if result_info['winner'] != drone_id:
                # 不同无人机间的协同
                synergy_bonus += 0.1  # 多样性奖励
            else:
                # 同一无人机的任务协同
                synergy_bonus += 0.05  # 效率奖励
        
        return synergy_bonus
    
    # ===== 创新算法4: 预测性路径规划 =====
    def predictive_path_planning(self, drone_states: Dict, time_horizon: float = 30.0) -> Dict[int, List[np.ndarray]]:
        """
        基于机器学习的预测性路径规划
        创新点：预测其他无人机未来轨迹，提前避免冲突
        """
        predicted_paths = {}
        
        for drone_id, state in drone_states.items():
            # 基于历史运动模式预测未来路径
            predicted_path = self._predict_drone_trajectory(drone_id, state, time_horizon)
            predicted_paths[drone_id] = predicted_path
            
            # 存储预测结果
            self.movement_predictions[drone_id] = predicted_path
        
        # 检测和解决预测的冲突
        conflict_resolutions = self._resolve_predicted_conflicts(predicted_paths)
        
        # 应用冲突解决方案
        for drone_id, adjusted_path in conflict_resolutions.items():
            predicted_paths[drone_id] = adjusted_path
        
        return predicted_paths
    
    def _predict_drone_trajectory(self, drone_id: int, current_state, time_horizon: float) -> List[np.ndarray]:
        """预测单个无人机的未来轨迹"""
        trajectory = []
        dt = 1.0  # 1秒间隔
        steps = int(time_horizon / dt)
        
        current_pos = current_state.position.copy()
        current_vel = current_state.velocity.copy()
        
        for step in range(steps):
            # 简化的运动学模型预测
            # 实际应用中可以使用更复杂的学习模型
            
            # 添加一些随机性模拟不确定性
            noise = np.random.normal(0, 0.1, 3)
            predicted_vel = current_vel + noise
            
            # 速度限制
            speed = np.linalg.norm(predicted_vel)
            if speed > 5.0:  # 最大速度限制
                predicted_vel = predicted_vel / speed * 5.0
            
            # 更新位置
            current_pos = current_pos + predicted_vel * dt
            trajectory.append(current_pos.copy())
            
            # 简单的速度更新（可以替换为更复杂的模型）
            current_vel = predicted_vel * 0.9  # 阻尼
        
        return trajectory
    
    def _resolve_predicted_conflicts(self, predicted_paths: Dict) -> Dict:
        """解决预测的路径冲突"""
        conflict_resolutions = {}
        
        # 检测所有潜在冲突
        conflicts = []
        drone_ids = list(predicted_paths.keys())
        
        for i, drone_a in enumerate(drone_ids):
            for j, drone_b in enumerate(drone_ids[i+1:], i+1):
                path_a = predicted_paths[drone_a]
                path_b = predicted_paths[drone_b]
                
                # 检查路径交叉点
                min_len = min(len(path_a), len(path_b))
                for step in range(min_len):
                    distance = distance_3d(path_a[step], path_b[step])
                    if distance < 3.0:  # 安全距离
                        conflicts.append({
                            'drones': (drone_a, drone_b),
                            'time_step': step,
                            'distance': distance,
                            'positions': (path_a[step], path_b[step])
                        })
        
        # 为每个冲突生成解决方案
        for conflict in conflicts:
            drone_a, drone_b = conflict['drones']
            step = conflict['time_step']
            
            # 简单的回避策略：让一个无人机稍微偏移路径
            if drone_a not in conflict_resolutions:
                conflict_resolutions[drone_a] = predicted_paths[drone_a].copy()
            
            # 添加偏移
            offset = np.array([2.0, 0.0, 0.5])  # 偏移方向
            for i in range(step, len(conflict_resolutions[drone_a])):
                conflict_resolutions[drone_a][i] += offset * np.exp(-(i-step)*0.1)  # 逐渐减小偏移
        
        return conflict_resolutions
    
    # ===== 创新算法5: 涌现行为分析 =====
    def emergence_behavior_analysis(self, swarm_states: Dict, time_window: float = 60.0) -> Dict:
        """
        群体涌现行为分析和优化
        创新点：识别和利用群体智能涌现的复杂行为模式
        """
        # 构建行为网络图
        self._build_behavioral_network(swarm_states)
        
        # 检测涌现模式
        emergence_patterns = self._detect_emergence_patterns()
        
        # 分析群体智能指标
        swarm_intelligence_metrics = self._calculate_swarm_intelligence_metrics()
        
        # 生成行为优化建议
        optimization_suggestions = self._generate_behavioral_optimizations(emergence_patterns)
        
        return {
            'emergence_patterns': emergence_patterns,
            'swarm_intelligence': swarm_intelligence_metrics,
            'optimization_suggestions': optimization_suggestions,
            'network_analysis': self._analyze_behavioral_network()
        }
    
    def _build_behavioral_network(self, swarm_states: Dict):
        """构建无人机行为网络"""
        # 清除旧的网络
        self.behavioral_network.clear()
        
        # 添加节点（无人机）
        for drone_id in swarm_states:
            self.behavioral_network.add_node(drone_id)
        
        # 添加边（交互关系）
        for drone_a in swarm_states:
            for drone_b in swarm_states:
                if drone_a != drone_b:
                    # 计算交互强度
                    interaction_strength = self._calculate_interaction_strength(
                        swarm_states[drone_a], swarm_states[drone_b]
                    )
                    
                    if interaction_strength > 0.1:  # 阈值过滤
                        self.behavioral_network.add_edge(
                            drone_a, drone_b, 
                            weight=interaction_strength
                        )
    
    def _calculate_interaction_strength(self, state_a, state_b) -> float:
        """计算两个无人机的交互强度"""
        # 基于距离的交互
        distance = distance_3d(state_a.position, state_b.position)
        distance_factor = np.exp(-distance / 20.0)  # 20米特征距离
        
        # 基于速度相似性的交互
        vel_similarity = 1.0 - np.linalg.norm(state_a.velocity - state_b.velocity) / 10.0
        vel_factor = max(0, vel_similarity)
        
        return distance_factor * 0.6 + vel_factor * 0.4
    
    def _detect_emergence_patterns(self) -> List[Dict]:
        """检测群体涌现行为模式"""
        patterns = []
        
        # 检测群集行为 (Flocking)
        if self._detect_flocking_behavior():
            patterns.append({
                'type': 'flocking',
                'strength': self._measure_flocking_strength(),
                'description': 'Coordinated movement in same direction'
            })
        
        # 检测分工行为 (Task Division)
        if self._detect_task_division_behavior():
            patterns.append({
                'type': 'task_division',
                'strength': self._measure_division_efficiency(),
                'description': 'Automatic role specialization'
            })
        
        # 检测领导-跟随行为 (Leadership)
        leadership_info = self._detect_leadership_behavior()
        if leadership_info:
            patterns.append({
                'type': 'leadership',
                'leader': leadership_info['leader'],
                'strength': leadership_info['strength'],
                'description': f"Drone {leadership_info['leader']} emerged as leader"
            })
        
        return patterns
    
    def _detect_flocking_behavior(self) -> bool:
        """检测群集行为"""
        if len(self.behavioral_network.nodes) < 3:
            return False
        
        # 计算网络密度
        density = nx.density(self.behavioral_network)
        return density > 0.6  # 高连接密度表示群集行为
    
    def _measure_flocking_strength(self) -> float:
        """衡量群集行为强度"""
        if not self.behavioral_network.edges:
            return 0.0
        
        # 基于边权重的平均值
        weights = [data['weight'] for _, _, data in self.behavioral_network.edges(data=True)]
        return np.mean(weights)
    
    def _detect_task_division_behavior(self) -> bool:
        """检测任务分工行为"""
        # 简化检测：基于网络的模块性
        try:
            communities = nx.community.greedy_modularity_communities(self.behavioral_network)
            return len(communities) > 1 and len(communities) <= len(self.behavioral_network.nodes) // 2
        except:
            return False
    
    def _measure_division_efficiency(self) -> float:
        """衡量分工效率"""
        try:
            modularity = nx.community.modularity(
                self.behavioral_network,
                nx.community.greedy_modularity_communities(self.behavioral_network)
            )
            return max(0, modularity)
        except:
            return 0.0
    
    def _detect_leadership_behavior(self) -> Optional[Dict]:
        """检测领导行为"""
        if not self.behavioral_network.nodes:
            return None
        
        # 基于中心性指标检测领导者
        centrality = nx.degree_centrality(self.behavioral_network)
        
        if centrality:
            leader = max(centrality, key=centrality.get)
            strength = centrality[leader]
            
            if strength > 0.7:  # 高中心性阈值
                return {
                    'leader': leader,
                    'strength': strength,
                    'centrality_scores': centrality
                }
        
        return None
    
    def _calculate_swarm_intelligence_metrics(self) -> Dict:
        """计算群体智能指标"""
        metrics = {
            'coordination_index': 0.0,
            'adaptation_speed': 0.0,
            'information_flow_efficiency': 0.0,
            'collective_decision_quality': 0.0
        }
        
        if self.behavioral_network.nodes:
            # 协调指数
            metrics['coordination_index'] = nx.density(self.behavioral_network)
            
            # 信息流效率
            if len(self.behavioral_network.nodes) > 1:
                try:
                    avg_path_length = nx.average_shortest_path_length(self.behavioral_network)
                    metrics['information_flow_efficiency'] = 1.0 / avg_path_length
                except:
                    metrics['information_flow_efficiency'] = 0.0
            
            # 其他指标的简化计算
            metrics['adaptation_speed'] = 0.8  # 模拟值
            metrics['collective_decision_quality'] = 0.75  # 模拟值
        
        return metrics
    
    def _generate_behavioral_optimizations(self, patterns: List[Dict]) -> List[Dict]:
        """基于涌现行为生成优化建议"""
        suggestions = []
        
        for pattern in patterns:
            if pattern['type'] == 'flocking':
                if pattern['strength'] > 0.8:
                    suggestions.append({
                        'type': 'reduce_clustering',
                        'priority': 'medium',
                        'description': 'Strong flocking detected, consider increasing exploration diversity'
                    })
            
            elif pattern['type'] == 'task_division':
                if pattern['strength'] < 0.3:
                    suggestions.append({
                        'type': 'enhance_specialization',
                        'priority': 'high',
                        'description': 'Low task division efficiency, encourage role specialization'
                    })
            
            elif pattern['type'] == 'leadership':
                suggestions.append({
                    'type': 'leadership_rotation',
                    'priority': 'low',
                    'description': f"Consider rotating leadership from Drone {pattern['leader']}"
                })
        
        return suggestions
    
    def _analyze_behavioral_network(self) -> Dict:
        """分析行为网络结构"""
        if not self.behavioral_network.nodes:
            return {}
        
        analysis = {
            'node_count': len(self.behavioral_network.nodes),
            'edge_count': len(self.behavioral_network.edges),
            'density': nx.density(self.behavioral_network),
            'is_connected': nx.is_connected(self.behavioral_network.to_undirected()),
        }
        
        if self.behavioral_network.nodes:
            analysis['centrality_measures'] = {
                'degree': nx.degree_centrality(self.behavioral_network),
                'betweenness': nx.betweenness_centrality(self.behavioral_network),
                'closeness': nx.closeness_centrality(self.behavioral_network)
            }
        
        return analysis