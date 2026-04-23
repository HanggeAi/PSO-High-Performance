# PSO-Eigen: 高性能粒子群优化算法库

## 项目简介

PSO-Eigen 是一个基于 C++/Eigen 和 Python 的高性能粒子群优化（Particle Swarm Optimization, PSO）算法库。本项目基于高性能C++线性代数库Eigen, 通过向量化计算显著提升了大规模优化问题的求解效率。

## ✨ 核心特性

- **🚀 高性能计算**: Eigen 向量化实现，支持 SIMD 指令加速
- **🎯 易于使用**: 简洁的 Python API，开箱即用
- **📈 并行计算**: 支持多进程并行适应度计算
- **📊 可视化工具**: 内置收敛曲线和粒子分布可视化

## 📦 安装指南

### 前置依赖

- C++14 或更高版本编译器
- CMake 3.10+
- Eigen3
- pybind11
- Python 3.7+
- NumPy
- Matplotlib

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/pso-eigen.git
cd pso-eigen

# 2. 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 3. 安装 Python 依赖
pip install numpy matplotlib

# 4. 配置C++依赖
# setup.py中，修改pybind11和eigen3的头文件目录
include_dirs=[
        np.get_include(),
        './',         #
        'pybind11/include', # your pybind11 include
        'eigen-3.4.0'  # your eigen3 include
    ]

# 5. 编译安装
pip install -e .
```

## 🚀 快速开始

### 基本使用

```python
import numpy as np
import pso_eigen  # 或 import pso 使用原始版本

# 1. 创建 PSO 配置
config = pso_eigen.PSOConfig()
config.num_particles = 50
config.num_dimensions = 10
config.max_iterations = 1000
config.lower_bounds = [-5.0] * 10
config.upper_bounds = [5.0] * 10

# 2. 创建优化器
optimizer = pso_eigen.ParticleSwarmOptimizer(config)

# 3. 定义目标函数
def sphere_function(x):
    return np.sum(x**2)

# 4. 优化循环
for iteration in range(config.max_iterations):
    if optimizer.is_converged():
        break
    
    # 获取粒子位置
    positions = optimizer.get_particle_positions()
    positions_np = np.array(positions)
    
    # 计算适应度
    fitness_values = [sphere_function(pos) for pos in positions]
    
    # 更新优化器
    optimizer.set_particle_fitnesses(fitness_values)
    optimizer.update_particles()
    
    # 打印进度
    if iteration % 100 == 0:
        best_fitness = optimizer.get_best_fitness()
        print(f"Iteration {iteration}: Best fitness = {best_fitness:.6e}")

# 5. 获取结果
best_position = optimizer.get_best_position()
best_fitness = optimizer.get_best_fitness()
print(f"\n优化完成!")
print(f"最优适应度: {best_fitness:.6e}")
print(f"最优解: {best_position}")
```

### 使用 PSOWrapper（推荐）

```python
from pso_demo import PSOWrapper
import numpy as np

# 1. 创建优化器
pso_solver = PSOWrapper(
    num_particles=30,
    num_dimensions=2,
    max_iterations=500
)

# 2. 定义测试函数
def rastrigin_function(positions):
    """Rastrigin 函数"""
    A = 10
    n = positions.shape[1]
    return A * n + np.sum(positions**2 - A * np.cos(2 * np.pi * positions), axis=1)

# 3. 运行优化
result = pso_solver.optimize(rastrigin_function, verbose=True)

# 4. 可视化
pso_solver.plot_convergence()  # 收敛曲线
pso_solver.plot_particles(0)   # 初始粒子分布
pso_solver.plot_particles(-1)  # 最终粒子分布
```

## 🔧 高级用法

### 并行计算

对于计算密集的适应度函数，可以使用 `ParallelPSOWrapper` 进行并行计算：

```python
from pso_parallel import ParallelPSOWrapper

# 1. 创建并行优化器
parallel_solver = ParallelPSOWrapper(
    num_particles=100,
    num_dimensions=20,
    max_iterations=500,
    n_workers=4  # 使用 4 个工作进程
)

# 2. 定义复杂适应度函数
def complex_fitness_function(positions):
    """复杂的适应度函数，可能涉及数值计算库"""
    from scipy.special import jv  # 贝塞尔函数
    
    result = []
    for pos in positions:
        fitness = 0
        for i, x in enumerate(pos):
            fitness += (x - 1)**2 + 100 * (x**2 - pos[(i+1)%len(pos)])**2
            fitness += abs(jv(0, x * 10))
        
        result.append(fitness)
    
    return np.array(result)

# 3. 并行优化
result = parallel_solver.optimize_parallel(complex_fitness_function)
print(f"最优适应度: {result['best_fitness']:.6e}")
```

### 自定义配置

```python
config = pso_eigen.PSOConfig()

# 算法参数
config.num_particles = 100      # 粒子数量
config.num_dimensions = 30      # 问题维度
config.max_iterations = 1000    # 最大迭代次数
config.inertia_weight = 0.729   # 惯性权重
config.cognitive_coeff = 1.49445  # 认知系数
config.social_coeff = 1.49445     # 社会系数

# 搜索空间边界
config.lower_bounds = [-10.0] * config.num_dimensions
config.upper_bounds = [10.0] * config.num_dimensions
```

## 📊 测试函数

库内置了多个经典测试函数：

```python
import numpy as np

def sphere_function(positions):
    """Sphere 函数: f(x) = Σx_i²"""
    return np.sum(positions**2, axis=1)

def rastrigin_function(positions):
    """Rastrigin 函数"""
    A = 10
    n = positions.shape[1]
    return A * n + np.sum(positions**2 - A * np.cos(2 * np.pi * positions), axis=1)

def rosenbrock_function(positions):
    """Rosenbrock 函数"""
    x = positions[:, :-1]
    y = positions[:, 1:]
    return np.sum(100 * (y - x**2) ** 2 + (1 - x) ** 2, axis=1)

def ackley_function(positions):
    """Ackley 函数"""
    a, b, c = 20, 0.2, 2*np.pi
    n = positions.shape[1]
    sum1 = np.sum(positions**2, axis=1)
    sum2 = np.sum(np.cos(c * positions), axis=1)
    return -a * np.exp(-b * np.sqrt(sum1/n)) - np.exp(sum2/n) + a + np.e
```

## 🎯 API 参考

### PSOConfig 配置类

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| num_particles | int | 50 | 粒子数量 |
| num_dimensions | int | 10 | 问题维度 |
| max_iterations | int | 1000 | 最大迭代次数 |
| inertia_weight | float | 0.729 | 惯性权重 |
| cognitive_coeff | float | 1.49445 | 认知系数 |
| social_coeff | float | 1.49445 | 社会系数 |
| lower_bounds | list[float] | [] | 下界数组 |
| upper_bounds | list[float] | [] | 上界数组 |

### ParticleSwarmOptimizer 类

| 方法 | 返回类型 | 说明 |
|------|----------|------|
| `get_particle_positions()` | list[list[float]] | 获取所有粒子当前位置 |
| `set_particle_fitnesses(fitness_values)` | void | 设置所有粒子的适应度值 |
| `update_particles()` | void | 更新粒子位置和速度 |
| `get_best_position()` | list[float] | 获取历史最优位置 |
| `get_best_fitness()` | float | 获取历史最优适应度 |
| `get_current_iteration()` | int | 获取当前迭代次数 |
| `is_converged()` | bool | 检查是否收敛（达到最大迭代次数） |
| `reinitialize()` | void | 重新初始化粒子群 |

### PSOWrapper 类

| 方法 | 参数 | 说明 |
|------|------|------|
| `__init__(num_particles, num_dimensions, max_iterations)` | 粒子数, 维度, 最大迭代次数 | 构造函数 |
| `optimize(fitness_func, verbose=True)` | 适应度函数, 是否打印进度 | 执行优化 |
| `get_result()` | 无 | 获取优化结果 |
| `plot_convergence()` | 无 | 绘制收敛曲线 |
| `plot_particles(iteration=-1)` | 迭代次数（默认最后一代） | 绘制粒子分布 |

## 🔄 版本选择指南

| 场景 | 推荐版本 | 原因 |
|------|----------|------|
| 快速原型开发 | 原始版本 (pso) | 编译简单，调试方便 |
| 高维问题 (>50维) | Eigen 版本 (pso_eigen) | 向量化计算显著加速 |
| 大规模粒子群 (>1000粒子) | Eigen 版本 (pso_eigen) | 内存效率更高 |
| 计算密集型适应度函数 | ParallelPSOWrapper | 并行计算加速 |
| 需要自定义修改 | 原始版本 (pso) | 代码更简单易懂 |

## 📁 项目结构

```
pso-eigen/
├── README.md                 # 项目说明文档
├── CMakeLists.txt           # CMake 配置文件
├── pso_core.cpp            # 原始实现核心代码
├── pso_core_eigen.cpp      # Eigen 加速核心代码
├── pso_bindings.cpp        # 原始版本 Python 绑定
├── pso_bindings_eigen.cpp  # Eigen 版本 Python 绑定
├── pso_demo.py             # 基本使用示例
├── pso_parallel.py         # 并行计算示例
├── examples/               # 更多示例
│   ├── benchmark.py        # 性能对比测试
│   ├── high_dimension.py   # 高维问题示例
│   └── custom_function.py  # 自定义函数示例
└── tests/                  # 单元测试
    ├── test_basic.py       # 基本功能测试
    └── test_performance.py # 性能测试
```

## 📄 许可证

本项目采用 MIT 许可证。详情请见 LICENSE 文件。

## ⭐ 致谢

感谢以下开源项目的启发和帮助：

- https://eigen.tuxfamily.org/ - C++ 模板库，用于线性代数
- https://github.com/pybind/pybind11 - 无缝连接 C++ 和 Python
- https://numpy.org/ - Python 科学计算基础库
- https://matplotlib.org/ - Python 绘图库

---

**开始优化吧！** 🚀 使用 PSO-Eigen 解决您的优化问题，享受高性能计算带来的效率提升！
