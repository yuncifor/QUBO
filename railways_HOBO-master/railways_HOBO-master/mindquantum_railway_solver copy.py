# mindquantum_railway_solver.py (修复循环导入)

"""
铁路调度问题的MindQuantum量子算法实现
修复循环导入问题
"""

import numpy as np
import mindquantum as mq
from mindquantum import Hamiltonian, QubitOperator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RX, RY, RZ, X, Y, Z
from mindquantum.core.operators import TimeEvolution
from mindquantum.simulator import Simulator
import matplotlib.pyplot as plt

class MindQuantumRailwaySolver:
    """MindQuantum铁路调度求解器"""
    
    def __init__(self, problem_data):
        self.problem_data = problem_data
        self.qubo_matrix = None
        self.hamiltonian = None
        self.hamiltonian_op = None  # 添加这个属性
        self.circuit = None
        self.simulator = None
        
    def build_qubo_from_problem(self):
        """从铁路调度问题构建QUBO矩阵"""
        trains_paths = getattr(self.problem_data, 'trains_paths', {})
        trains_timing = getattr(self.problem_data, 'trains_timing', {})
        d_max = getattr(self.problem_data, 'd_max', 5)
        
        indices, num_vars = self._get_indices(trains_paths, d_max)
        Q = np.zeros((num_vars, num_vars))
        
        Q += self._add_sum_constraints(indices, num_vars, trains_paths)
        Q += self._add_headway_constraints(indices, num_vars, trains_timing, trains_paths)
        Q += self._add_single_track_constraints(indices, num_vars, trains_timing, trains_paths)
        Q += self._add_minimal_stay_constraints(indices, num_vars, trains_timing, trains_paths)
        Q += self._add_switch_constraints(indices, num_vars, trains_timing, trains_paths)
        Q += self._add_penalty_terms(indices, num_vars, trains_timing)
        
        self.qubo_matrix = Q
        return Q
    
    def _get_indices(self, trains_paths, d_max):
        """创建变量索引映射"""
        indices = []
        S = trains_paths.get("Paths", {})
        J = trains_paths.get("J", [])
        
        for j in J:
            for s in S.get(j, []):
                if not self._skip_station(j, s, trains_paths):
                    for d in range(d_max + 1):
                        indices.append({"j": j, "s": s, "d": d})
        return indices, len(indices)
    
    def _skip_station(self, j, s, trains_paths):
        """检查是否跳过车站"""
        skip_info = trains_paths.get("skip_station", {})
        return j in skip_info and skip_info[j] == s
    
    def _add_sum_constraints(self, indices, num_vars, trains_paths):
        """添加求和约束"""
        Q = np.zeros((num_vars, num_vars))
        p_sum = getattr(self.problem_data, 'p_sum', 2.0)
        
        train_station_pairs = {}
        for idx, var in enumerate(indices):
            key = (var["j"], var["s"])
            if key not in train_station_pairs:
                train_station_pairs[key] = []
            train_station_pairs[key].append(idx)
        
        for key, var_indices in train_station_pairs.items():
            for i in var_indices:
                for j in var_indices:
                    if i == j:
                        Q[i, j] += p_sum * (-1)
                    else:
                        Q[i, j] += p_sum * 1
        
        return Q
    
    def _add_headway_constraints(self, indices, num_vars, trains_timing, trains_paths):
        """添加列车安全间隔约束"""
        Q = np.zeros((num_vars, num_vars))
        p_pair = getattr(self.problem_data, 'p_pair', 1.0)
        
        for i in range(num_vars):
            for j in range(i+1, num_vars):
                var_i = indices[i]
                var_j = indices[j]
                
                if self._should_have_headway(var_i, var_j, trains_timing, trains_paths):
                    Q[i, j] += p_pair
                    Q[j, i] += p_pair
        
        return Q
    
    def _should_have_headway(self, var_i, var_j, trains_timing, trains_paths):
        """检查安全间隔约束"""
        j1, s1 = var_i["j"], var_i["s"]
        j2, s2 = var_j["j"], var_j["s"]
        
        if s1 == s2 and j1 != j2:
            return True
        return False
    
    def _add_single_track_constraints(self, indices, num_vars, trains_timing, trains_paths):
        """添加单线轨道约束"""
        return np.zeros((num_vars, num_vars))  # 简化实现
    
    def _add_minimal_stay_constraints(self, indices, num_vars, trains_timing, trains_paths):
        """添加最小停留时间约束"""
        return np.zeros((num_vars, num_vars))
    
    def _add_switch_constraints(self, indices, num_vars, trains_timing, trains_paths):
        """添加道岔占用约束"""
        return np.zeros((num_vars, num_vars))
    
    def _add_penalty_terms(self, indices, num_vars, trains_timing):
        """添加惩罚项"""
        Q = np.zeros((num_vars, num_vars))
        
        for i in range(num_vars):
            delay = indices[i]["d"]
            penalty_weight = self._get_penalty_weight(indices[i], trains_timing)
            Q[i, i] += penalty_weight * delay
        
        return Q
    
    def _get_penalty_weight(self, var, trains_timing):
        """获取惩罚权重"""
        j, s = var["j"], var["s"]
        penalty_weights = trains_timing.get("penalty_weights", {})
        key = f"{j}_{s}"
        return penalty_weights.get(key, 1.0)
    
    def qubo_to_hamiltonian(self):
        """将QUBO矩阵转换为哈密顿量"""
        if self.qubo_matrix is None:
            self.build_qubo_from_problem()
        
        n_qubits = self.qubo_matrix.shape[0]
        hamiltonian_op = QubitOperator()
        
        for i in range(n_qubits):
            for j in range(i, n_qubits):
                if abs(self.qubo_matrix[i, j]) > 1e-10:
                    if i == j:
                        coeff = self.qubo_matrix[i, i] / 2.0
                        term = f"Z{i}"
                    else:
                        coeff = self.qubo_matrix[i, j] / 4.0
                        term = f"Z{i} Z{j}"
                    
                    hamiltonian_op += QubitOperator(term, float(coeff))
        
        constant_term = np.sum(np.diag(self.qubo_matrix)) / 2.0
        constant_term += np.sum(self.qubo_matrix) / 4.0
        hamiltonian_op += QubitOperator('', float(constant_term))
        
        self.hamiltonian_op = hamiltonian_op
        self.hamiltonian = Hamiltonian(hamiltonian_op)
        return self.hamiltonian
    
    def build_quantum_circuit(self, layers=3):
        """构建量子电路"""
        if self.hamiltonian is None:
            self.qubo_to_hamiltonian()
        
        try:
            from mindquantum.algorithm.nisq import QAOAAnsatz
            self.circuit = QAOAAnsatz(self.hamiltonian_op, layers)
            return self.circuit
        except (ImportError, AttributeError) as e:
            print(f"QAOAAnsatz失败: {e}，使用手动构建")
            return self._build_manual_qaoa(layers)
    
    def _build_manual_qaoa(self, layers=3):
        """手动构建QAOA电路"""
        n_qubits = self.hamiltonian_op.qubit_count
        circuit = Circuit()
        
        for i in range(n_qubits):
            circuit += H.on(i)
        
        for layer in range(layers):
            for i in range(n_qubits):
                circuit += RZ(f'gamma_{layer}_{i}').on(i)
            for i in range(n_qubits):
                circuit += RX(f'beta_{layer}_{i}').on(i)
        
        self.circuit = circuit
        return circuit
    
    def solve_with_vqe(self, max_iter=100):
        """使用VQE求解"""
        if self.circuit is None:
            self.build_quantum_circuit()
        
        try:
            self.simulator = Simulator('projectq', self.circuit.n_qubits)
            
            from mindquantum.framework import MQAnsatzOnlyLayer
            import mindspore as ms
            from mindspore import nn
            
            circuit = self.circuit.as_ansatz()
            quantum_net = MQAnsatzOnlyLayer(circuit, 'projectq', self.hamiltonian)
            
            optimizer = nn.Adam(quantum_net.trainable_params(), learning_rate=0.1)
            train_net = nn.TrainOneStepCell(quantum_net, optimizer)
            
            for i in range(max_iter):
                loss = train_net()
                if i % 10 == 0:
                    print(f'Step {i}, loss: {loss}')
            
            final_params = quantum_net.weight.asnumpy()
            self.simulator.set_qs(self.circuit.get_qs(pr=final_params))
            final_energy = quantum_net()
            
            return {
                'optimal_value': float(final_energy.asnumpy()),
                'optimal_vector': self.simulator.get_qs(),
                'optimal_params': final_params
            }
            
        except Exception as e:
            print(f"VQE求解失败: {e}")
            return self._fallback_solution()
    
    def solve_with_qaoa(self, steps=5):
        """使用QAOA算法求解"""
        if self.hamiltonian is None:
            self.qubo_to_hamiltonian()
        
        try:
            n_qubits = self.hamiltonian_op.qubit_count
            
            circuit = Circuit()
            for i in range(n_qubits):
                circuit += H.on(i)
            
            for i in range(n_qubits):
                circuit += RZ(f'gamma_{i}').on(i)
                circuit += RX(f'beta_{i}').on(i)
            
            self.circuit = circuit
            self.simulator = Simulator('projectq', n_qubits)
            
            import random
            params = {f'gamma_{i}': random.uniform(0, 1) for i in range(n_qubits)}
            params.update({f'beta_{i}': random.uniform(0, 1) for i in range(n_qubits)})
            
            self.simulator.apply_circuit(circuit, pr=params)
            result = self.simulator.get_qs()
            
            return result
            
        except Exception as e:
            print(f"QAOA求解失败: {e}")
            return self._fallback_solution()
    
    def _fallback_solution(self):
        """回退解决方案"""
        return {
            'optimal_value': 0.0,
            'optimal_vector': None,
            'status': 'fallback'
        }
    
    def interpret_solution(self, quantum_result):
        """解释量子计算结果"""
        if isinstance(quantum_result, dict):
            best_state = quantum_result.get('optimal_vector', None)
            best_energy = quantum_result.get('optimal_value', 0.0)
        else:
            best_state = quantum_result
            best_energy = 0.0
        
        n_vars = self.qubo_matrix.shape[0] if self.qubo_matrix is not None else 0
        solution = {}
        
        for i in range(min(n_vars, 10)):
            solution[f"var_{i}"] = {
                "assigned": True,
                "value": 0 if best_state is None else (abs(best_state[i]) if i < len(best_state) else 0)
            }
        
        return {
            'schedule': solution,
            'energy': best_energy,
            'quantum_state': best_state,
            'variables_count': n_vars
        }
    
    def visualize_circuit(self, max_gates=50):
        """可视化量子电路"""
        if self.circuit is not None:
            print("量子电路结构：")
            circuit_str = str(self.circuit)
            lines = circuit_str.split('\n')
            for i, line in enumerate(lines[:max_gates]):
                print(line)
            if len(lines) > max_gates:
                print(f"... 还有 {len(lines) - max_gates} 个门未显示")
            return self.circuit
        else:
            print("请先构建量子电路")
            return None
    
    def get_problem_stats(self):
        """获取问题统计信息"""
        if self.qubo_matrix is None:
            self.build_qubo_from_problem()
        
        stats = {
            'qubo_shape': self.qubo_matrix.shape,
            'qubo_nonzero': np.count_nonzero(self.qubo_matrix),
            'qubo_density': np.count_nonzero(self.qubo_matrix) / self.qubo_matrix.size,
            'qubo_symmetric': np.allclose(self.qubo_matrix, self.qubo_matrix.T)
        }
        
        if hasattr(self, 'hamiltonian_op'):
            stats['hamiltonian_terms'] = len(self.hamiltonian_op.terms)
            
            z_terms = sum(1 for term in self.hamiltonian_op.terms if 'Z' in str(term) and 'Z Z' not in str(term))
            zz_terms = sum(1 for term in self.hamiltonian_op.terms if 'Z Z' in str(term))
            stats['z_terms'] = z_terms
            stats['zz_terms'] = zz_terms
        
        return stats


# 将测试类移到文件末尾，避免循环导入
class SimpleTestProblem:
    """简化的测试问题数据"""
    def __init__(self):
        self.trains_paths = {
            "Paths": {"j1": ["S1", "S2"], "j2": ["S1", "S2"], "j3": ["S2", "S1"]},
            "J": ["j1", "j2", "j3"],
            "skip_station": {}
        }
        self.trains_timing = {
            "penalty_weights": {"j1_S1": 1.0, "j2_S1": 1.0, "j3_S2": 1.0}
        }
        self.d_max = 10
        self.p_sum = 2.5
        self.p_pair = 1.25
        self.p_qubic = 2.1


# 只有在直接运行此文件时才执行测试
if __name__ == "__main__":
    # 直接测试
    problem = SimpleTestProblem()
    solver = MindQuantumRailwaySolver(problem)
    
    print("=== MindQuantum铁路调度求解器测试 ===")
    
    # 1. 构建QUBO矩阵
    print("1. 构建QUBO矩阵...")
    qubo_matrix = solver.build_qubo_from_problem()
    print(f"QUBO矩阵形状: {qubo_matrix.shape}")
    
    # 2. 获取统计信息
    stats = solver.get_problem_stats()
    print(f"问题统计: {stats}")
    
    print("\n=== 测试完成 ===")
