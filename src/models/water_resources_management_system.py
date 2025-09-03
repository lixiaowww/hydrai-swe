"""
水资源管理决策支持系统
基于成熟的开源算法和优化库实现
包含多目标优化、水库调度、用水效率评估等功能
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings
import sys
import os
warnings.filterwarnings('ignore')

# 添加真实数据加载器
sys.path.append(os.path.join(os.path.dirname(__file__)))
from real_data_loader import RealDataLoader

class MultiObjectiveOptimizer:
    """多目标优化器 - 基于NSGA-II算法思想"""
    
    def __init__(self):
        self.objectives = []
        self.constraints = []
        
    def add_objective(self, func, weight=1.0, minimize=True):
        """添加目标函数"""
        self.objectives.append({
            'function': func,
            'weight': weight,
            'minimize': minimize
        })
    
    def add_constraint(self, func, bound_type='ineq'):
        """添加约束条件"""
        self.constraints.append({
            'function': func,
            'type': bound_type
        })
    
    def optimize(self, bounds, population_size=50, max_generations=100):
        """执行多目标优化"""
        try:
            # 使用差分进化算法进行多目标优化
            def combined_objective(x):
                total_value = 0
                for obj in self.objectives:
                    value = obj['function'](x)
                    if not obj['minimize']:
                        value = -value  # 转换为最小化问题
                    total_value += obj['weight'] * value
                return total_value
            
            # 约束条件
            constraints = []
            for constraint in self.constraints:
                if constraint['type'] == 'ineq':
                    constraints.append({'type': 'ineq', 'fun': constraint['function']})
                elif constraint['type'] == 'eq':
                    constraints.append({'type': 'eq', 'fun': constraint['function']})
            
            # 执行优化
            result = minimize(
                combined_objective,
                x0=np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds]),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': max_generations}
            )
            
            return {
                'success': result.success,
                'optimal_solution': result.x.tolist(),
                'optimal_value': result.fun,
                'iterations': result.nit,
                'message': result.message
            }
        except Exception as e:
            return {'error': f'优化失败: {str(e)}'}


class ReservoirOperationOptimizer:
    """水库调度优化器"""
    
    def __init__(self, reservoir_data):
        self.reservoir_data = reservoir_data
        self.inflow = reservoir_data.get('inflow', [])
        self.demand = reservoir_data.get('demand', [])
        self.capacity = reservoir_data.get('capacity', 1000)
        self.min_level = reservoir_data.get('min_level', 0.1)
        self.max_level = reservoir_data.get('max_level', 0.9)
        
    def calculate_water_balance(self, releases):
        """计算水量平衡"""
        try:
            storage = [self.capacity * 0.5]  # 初始蓄水量
            levels = [0.5]
            
            for i in range(len(releases)):
                # 水量平衡方程: S(t+1) = S(t) + I(t) - R(t) - E(t)
                inflow_t = self.inflow[i] if i < len(self.inflow) else 0
                release_t = releases[i]
                evaporation_t = storage[i] * 0.01  # 简化蒸发损失
                
                new_storage = storage[i] + inflow_t - release_t - evaporation_t
                new_storage = max(0, min(new_storage, self.capacity))
                new_level = new_storage / self.capacity
                
                storage.append(new_storage)
                levels.append(new_level)
            
            return {
                'storage': storage[1:],
                'levels': levels[1:],
                'releases': releases,
                'total_inflow': sum(self.inflow),
                'total_release': sum(releases),
                'storage_efficiency': 1 - (sum(storage) / (len(storage) * self.capacity))
            }
        except Exception as e:
            return {'error': f'水量平衡计算失败: {str(e)}'}
    
    def optimize_releases(self, objectives=['reliability', 'efficiency', 'safety']):
        """优化水库放水策略"""
        try:
            n_periods = len(self.inflow)
            
            # 定义目标函数
            def reliability_objective(releases):
                """供水可靠性目标"""
                demand_met = sum(min(r, d) for r, d in zip(releases, self.demand))
                total_demand = sum(self.demand)
                return -demand_met / total_demand if total_demand > 0 else 0
            
            def efficiency_objective(releases):
                """用水效率目标"""
                waste = sum(max(0, r - d) for r, d in zip(releases, self.demand))
                return -waste / sum(releases) if sum(releases) > 0 else 0
            
            def safety_objective(releases):
                """防洪安全目标"""
                excess_releases = sum(max(0, r - 0.8 * self.capacity) for r in releases)
                return -excess_releases / (n_periods * self.capacity)
            
            # 约束条件
            def storage_constraint(releases):
                """蓄水量约束"""
                storage = [self.capacity * 0.5]
                violations = 0
                
                for i, release in enumerate(releases):
                    inflow_t = self.inflow[i] if i < len(self.inflow) else 0
                    evaporation_t = storage[i] * 0.01
                    new_storage = storage[i] + inflow_t - release - evaporation_t
                    
                    if new_storage < self.capacity * self.min_level:
                        violations += (self.capacity * self.min_level - new_storage)
                    elif new_storage > self.capacity * self.max_level:
                        violations += (new_storage - self.capacity * self.max_level)
                    
                    storage.append(max(0, min(new_storage, self.capacity)))
                
                return violations
            
            # 多目标优化
            optimizer = MultiObjectiveOptimizer()
            
            if 'reliability' in objectives:
                optimizer.add_objective(reliability_objective, weight=0.4)
            if 'efficiency' in objectives:
                optimizer.add_objective(efficiency_objective, weight=0.3)
            if 'safety' in objectives:
                optimizer.add_objective(safety_objective, weight=0.3)
            
            optimizer.add_constraint(storage_constraint, 'ineq')
            
            # 优化参数边界
            bounds = [(0, self.capacity * 0.5) for _ in range(n_periods)]
            
            result = optimizer.optimize(bounds)
            
            if result.get('success'):
                optimal_releases = result['optimal_solution']
                water_balance = self.calculate_water_balance(optimal_releases)
                
                return {
                    'success': True,
                    'optimal_releases': optimal_releases,
                    'water_balance': water_balance,
                    'objectives_achieved': {
                        'reliability': -reliability_objective(optimal_releases),
                        'efficiency': -efficiency_objective(optimal_releases),
                        'safety': -safety_objective(optimal_releases)
                    },
                    'optimization_info': result
                }
            else:
                return {'error': '优化失败', 'details': result}
                
        except Exception as e:
            return {'error': f'水库调度优化失败: {str(e)}'}


class WaterAllocationOptimizer:
    """水资源配置优化器"""
    
    def __init__(self, allocation_data):
        self.allocation_data = allocation_data
        self.users = allocation_data.get('users', [])
        self.priorities = allocation_data.get('priorities', {})
        self.efficiency_factors = allocation_data.get('efficiency_factors', {})
        self.total_water = allocation_data.get('total_water', 1000)
        
    def calculate_allocation_efficiency(self, allocations):
        """计算配置效率"""
        try:
            total_efficiency = 0
            total_allocated = 0
            
            for i, user in enumerate(self.users):
                if i < len(allocations):
                    allocation = allocations[i]
                    efficiency = self.efficiency_factors.get(user, 0.8)
                    priority = self.priorities.get(user, 1.0)
                    
                    # 效率 = 配置量 × 效率系数 × 优先级
                    user_efficiency = allocation * efficiency * priority
                    total_efficiency += user_efficiency
                    total_allocated += allocation
            
            return {
                'total_efficiency': total_efficiency,
                'average_efficiency': total_efficiency / len(self.users) if self.users else 0,
                'allocation_ratio': total_allocated / self.total_water if self.total_water > 0 else 0,
                'efficiency_by_user': {
                    user: allocations[i] * self.efficiency_factors.get(user, 0.8) * self.priorities.get(user, 1.0)
                    for i, user in enumerate(self.users) if i < len(allocations)
                }
            }
        except Exception as e:
            return {'error': f'配置效率计算失败: {str(e)}'}
    
    def optimize_allocation(self, constraints=None):
        """优化水资源配置"""
        try:
            n_users = len(self.users)
            
            def efficiency_objective(allocations):
                """效率目标函数"""
                efficiency_result = self.calculate_allocation_efficiency(allocations)
                return -efficiency_result['total_efficiency']
            
            def equity_objective(allocations):
                """公平性目标函数"""
                if n_users == 0:
                    return 0
                mean_allocation = sum(allocations) / n_users
                variance = sum((a - mean_allocation) ** 2 for a in allocations) / n_users
                return variance  # 最小化方差
            
            def priority_objective(allocations):
                """优先级目标函数"""
                priority_score = 0
                for i, user in enumerate(self.users):
                    if i < len(allocations):
                        priority = self.priorities.get(user, 1.0)
                        priority_score += allocations[i] * priority
                return -priority_score
            
            # 约束条件
            def total_water_constraint(allocations):
                """总水量约束"""
                return self.total_water - sum(allocations)
            
            def minimum_allocation_constraint(allocations):
                """最小配置约束"""
                min_allocation = self.total_water * 0.1 / n_users  # 每个用户至少10%
                violations = sum(max(0, min_allocation - a) for a in allocations)
                return violations
            
            # 多目标优化
            optimizer = MultiObjectiveOptimizer()
            optimizer.add_objective(efficiency_objective, weight=0.4)
            optimizer.add_objective(equity_objective, weight=0.3)
            optimizer.add_objective(priority_objective, weight=0.3)
            
            optimizer.add_constraint(total_water_constraint, 'eq')
            optimizer.add_constraint(minimum_allocation_constraint, 'ineq')
            
            # 优化参数边界
            max_per_user = self.total_water * 0.8  # 单个用户最多80%
            bounds = [(0, max_per_user) for _ in range(n_users)]
            
            result = optimizer.optimize(bounds)
            
            if result.get('success'):
                optimal_allocations = result['optimal_solution']
                efficiency_analysis = self.calculate_allocation_efficiency(optimal_allocations)
                
                return {
                    'success': True,
                    'optimal_allocations': {
                        user: optimal_allocations[i] if i < len(optimal_allocations) else 0
                        for i, user in enumerate(self.users)
                    },
                    'efficiency_analysis': efficiency_analysis,
                    'allocation_summary': {
                        'total_allocated': sum(optimal_allocations),
                        'allocation_percentage': sum(optimal_allocations) / self.total_water * 100,
                        'users_served': len([a for a in optimal_allocations if a > 0])
                    },
                    'optimization_info': result
                }
            else:
                return {'error': '配置优化失败', 'details': result}
                
        except Exception as e:
            return {'error': f'水资源配置优化失败: {str(e)}'}


class WaterResourcesManagementSystem:
    """水资源管理决策支持系统主类"""
    
    def __init__(self):
        self.reservoir_optimizer = None
        self.allocation_optimizer = None
        self.optimization_history = []
        self.real_data_loader = RealDataLoader()
        
    def load_reservoir_data(self, data=None, use_real_data=True, reservoir_name="red_river", days=30):
        """加载水库数据"""
        if use_real_data and data is None:
            # 使用真实数据
            real_data = self.real_data_loader.load_reservoir_data(reservoir_name, days)
            self.reservoir_optimizer = ReservoirOperationOptimizer(real_data)
            return {
                'status': 'success', 
                'message': '真实水库数据加载成功',
                'data_source': real_data.get('source', 'unknown'),
                'data_points': real_data.get('data_points', 0),
                'date_range': real_data.get('date_range', 'unknown')
            }
        else:
            # 使用提供的数据
            self.reservoir_optimizer = ReservoirOperationOptimizer(data)
            return {'status': 'success', 'message': '水库数据加载成功'}
    
    def load_allocation_data(self, data=None, use_real_data=True, region="manitoba"):
        """加载配置数据"""
        if use_real_data and data is None:
            # 使用真实数据
            real_data = self.real_data_loader.load_water_allocation_data(region)
            self.allocation_optimizer = WaterAllocationOptimizer(real_data)
            return {
                'status': 'success', 
                'message': '真实配置数据加载成功',
                'data_source': real_data.get('source', 'unknown'),
                'region': real_data.get('region', 'unknown'),
                'data_quality': real_data.get('data_quality', 'unknown')
            }
        else:
            # 使用提供的数据
            self.allocation_optimizer = WaterAllocationOptimizer(data)
            return {'status': 'success', 'message': '配置数据加载成功'}
    
    def run_reservoir_optimization(self, objectives=None):
        """运行水库调度优化"""
        if not self.reservoir_optimizer:
            return {'error': '请先加载水库数据'}
        
        if objectives is None:
            objectives = ['reliability', 'efficiency', 'safety']
        
        result = self.reservoir_optimizer.optimize_releases(objectives)
        
        if 'error' not in result:
            self.optimization_history.append({
                'type': 'reservoir_optimization',
                'timestamp': pd.Timestamp.now().isoformat(),
                'objectives': objectives,
                'result': result
            })
        
        return result
    
    def run_allocation_optimization(self):
        """运行水资源配置优化"""
        if not self.allocation_optimizer:
            return {'error': '请先加载配置数据'}
        
        result = self.allocation_optimizer.optimize_allocation()
        
        if 'error' not in result:
            self.optimization_history.append({
                'type': 'allocation_optimization',
                'timestamp': pd.Timestamp.now().isoformat(),
                'result': result
            })
        
        return result
    
    def generate_management_report(self):
        """生成管理报告"""
        report = {
            'system_status': {
                'reservoir_optimizer_loaded': self.reservoir_optimizer is not None,
                'allocation_optimizer_loaded': self.allocation_optimizer is not None,
                'total_optimizations': len(self.optimization_history)
            },
            'optimization_history': self.optimization_history,
            'recommendations': self._generate_recommendations(),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self):
        """生成管理建议"""
        recommendations = []
        
        if self.reservoir_optimizer:
            recommendations.append("水库调度优化器已就绪，可进行多目标优化分析")
        
        if self.allocation_optimizer:
            recommendations.append("水资源配置优化器已就绪，可进行公平效率平衡分析")
        
        if len(self.optimization_history) > 0:
            recommendations.append(f"已完成 {len(self.optimization_history)} 次优化分析")
        
        recommendations.extend([
            "建议定期更新水文数据以提高优化精度",
            "考虑气候变化对水资源管理的影响",
            "建立多情景分析以应对不确定性"
        ])
        
        return recommendations
