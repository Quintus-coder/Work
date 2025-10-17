"""
麻雀搜索算法（SSA）求解 1 辆配送车 + 3 架无人机 的协同配送路径优化问题
（移动会合模型：Launch–Serve–Rendezvous）
输入：
- Excel 文件包含两张表：
  1) Sheet "points"：配送点坐标（含depot）
     列：id, x, y
  2) Sheet "params"：模型参数
     key, value
       n_drones, truck_speed, drone_speed,
       truck_cost_per_dist, drone_cost_per_dist
输出：
- 最优总成本、车辆行驶路线、无人机任务计划（起飞点-服务点-会合点）
- 输出图片 best_plan.png 展示路径结果
"""
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ 1）读取数据 ============

EXCEL_PATH = r"E:/大学/水课/人工智能基础/第一次大作业/无人机车辆协同配送数据.xlsx"
points_df = pd.read_excel(EXCEL_PATH, sheet_name="点数据")
params_df = pd.read_excel(EXCEL_PATH, sheet_name="参数")
points_df = points_df.reset_index().rename(columns={"index": "id", "X": "x", "Y": "y"})

# 参数字典化
param = {row[0]: float(str(row[1])) for row in params_df.values}
depot = tuple(float(x) for x in str(param.get("货站点坐标")).split("，"))     # 站点坐标
N_TRUCK = int(param.get("车数量", 1))                                          # 车数量
C_TRUCK  = float(param.get("快递车单位距离运输成本", 1.0))                         # 车单位距离成本
V_TRUCK  = float(param.get("快递车移动速度", 10.0))                                # 车速度
N_DRONES = int(param.get("无人机数量", 3))                                           # 无人机数量
V_DRONE  = float(param.get("无人机移动速度", 20.0))                                    # 无人机速度
C_DRONE  = float(param.get("无人机单位距离运输成本", 0.5))                                 # 无人机单位距离成本

# 读取配送点信息
points_df = points_df.reset_index().rename(columns={"index": "id"})
pts = points_df.sort_values("id").reset_index(drop=True)
ids = pts["id"].tolist()
points = pts[["x", "y"]].to_numpy(dtype=float)
assert ids[0] == 0, "注意：depot 的 id 必须为 0"

n_all = len(ids)               # 总点数（含depot）
customers = list(range(1, n_all))  # 客户点编号（不含depot）

# 计算距离矩阵（欧氏距离）
def haversine(coord1, coord2):
    """
    计算两点经纬度坐标之间的球面距离（单位：公里）
    coord1, coord2: (纬度, 经度)，十进制度
    """
    R = 6371.0  # 地球半径（km）

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # 转换为弧度
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # 哈弗辛公式
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

D = np.zeros((n_all, n_all), dtype=float)
for i in range(n_all):
    for j in range(n_all):
        if i != j:
            D[i, j] = haversine(points[i], points[j])

# ============ 2）个体编码与解码 ============
"""
连续编码结构：
  route_key：控制客户访问顺序
  truck_key：决定客户由车辆服务还是无人机服务（sigmoid > 0.5 → 车辆服务）
  L_key：无人机起飞节点索引
  R_key：无人机会合节点索引
"""
def sigmoid(x): return 1/(1+np.exp(-x))

def keys_to_plan(route_key, truck_key, L_key, R_key, rng):
    # 1）确定车辆访问的客户顺序
    order = np.argsort(route_key)
    truck_mask = sigmoid(truck_key) > 0.5
    # 确保至少有一个客户由车服务
    if truck_mask.sum() == 0:
        truck_mask[rng.integers(0, len(customers))] = True

    truck_nodes = [customers[i] for i in order if truck_mask[i]]
    route = [0] + truck_nodes + [0]  # 车路线（首尾都是depot）

    # 2）剩余客户交给无人机服务
    drone_nodes = [customers[i] for i in order if not truck_mask[i]]

    # 3）为每个无人机客户分配起飞点与会合点
    sorties = []
    m = len(route)
    if m < 2:
        route = [0,0]; m = 2
    for j in drone_nodes:
        li = int(np.clip(int((m-1) * abs(np.tanh(L_key[j-1]))), 0, m-2))
        rk_min = li + 1
        rk = rk_min + int(np.clip(int((m-1-rk_min) * abs(np.tanh(R_key[j-1]))), 0, max(0, (m-1-rk_min))))
        sorties.append((li, j, rk))

    return route, sorties

# ============ 3）可行性与成本计算 ============

def route_length(route):
    return sum(D[a,b] for a,b in zip(route[:-1], route[1:]))

def arrival_times(route):
    """车辆沿主路线的到达时间（不考虑装卸）"""
    t = [0.0]
    for a, b in zip(route[:-1], route[1:]):
        t.append(t[-1] + float(D[a,b]) / V_TRUCK)
    return np.array(t)

def check_and_cost(route, sorties):
    """计算总成本 + 不可行罚分"""
    BIG = 1e6
    penalty = 0.0

    # 车辆成本
    truck_cost = C_TRUCK * route_length(route)
    t_arr = arrival_times(route)

    # 无人机任务信息
    intervals = []
    drone_cost = 0.0
    truck_served = set(route[1:-1])
    drone_served = set()

    for (li, j, rk) in sorties:
        if not (0 <= li < len(route)-1 and li < rk < len(route)):
            penalty += BIG
            continue
        i_node, k_node = route[li], route[rk]
        flight = D[i_node, j] + D[j, k_node]
        t_need = flight / V_DRONE
        t_have = t_arr[rk] - t_arr[li]
        if t_have + 1e-9 < t_need:
            penalty += BIG * (t_need - t_have)
        if j in truck_served:
            penalty += BIG
        intervals.append((t_arr[li], t_arr[rk]))
        drone_cost += C_DRONE * flight
        drone_served.add(j)

    # 唯一服务约束
    all_served = truck_served.union(drone_served)
    missing = set(customers) - all_served
    if missing:
        penalty += BIG * len(missing)

    # 同时在空中的无人机 ≤ N_DRONES
    events = []
    for s, e in intervals:
        events.append((s, +1))
        events.append((e, -1))
    events.sort()
    active = 0
    for _, ev in events:
        active += ev
        if active > N_DRONES:
            penalty += BIG * (active - N_DRONES)

    total_cost = truck_cost + drone_cost + penalty
    return total_cost, truck_cost, drone_cost, penalty

# ============ 4）麻雀搜索算法（SSA） ============

class SSA:
    def __init__(self, dim, n_pop=40, max_iter=300, p_producer=0.2, p_awareness=0.1, seed=42):
        self.dim = dim
        self.n = n_pop
        self.T = max_iter
        self.nP = max(1, int(p_producer * n_pop))  # 发现者数量
        self.nA = max(1, int(p_awareness * n_pop)) # 警戒者数量
        self.rng = np.random.default_rng(seed)
        self.X = self.rng.normal(0, 1, size=(self.n, self.dim))
        self.F = np.full(self.n, np.inf)
        self.best_x = None
        self.best_f = np.inf

    def optimize(self, fitness_fn):
        for t in range(self.T):
            # 评估适应度
            for i in range(self.n):
                self.F[i] = fitness_fn(self.X[i,:])
                if self.F[i] < self.best_f:
                    self.best_f = self.F[i]
                    self.best_x = self.X[i,:].copy()

            # 按适应度排序
            idx = np.argsort(self.F)
            self.X = self.X[idx,:]
            self.F = self.F[idx]
            best = self.X[0,:].copy()
            worst = self.X[-1,:].copy()

            # = 发现者更新 =
            r2 = self.rng.uniform(0,1)
            for i in range(self.nP):
                if r2 < 0.8:
                    self.X[i,:] = self.X[i,:]*np.exp(-(i)/self.T) + self.rng.normal(0,1,size=self.dim)
                else:
                    self.X[i,:] = self.X[i,:] + self.rng.normal(0,1,size=self.dim)*(self.X[i,:]-best)

            # = 加入者更新 =
            for i in range(self.nP, self.n):
                self.X[i,:] = self.rng.normal(0,1,size=self.dim)*np.abs(self.X[i,:]-best)

            # = 警戒者更新 =
            aware_idx = self.rng.choice(self.n, size=self.nA, replace=False)
            for i in aware_idx:
                self.X[i,:] = best + self.rng.normal(0,1,size=self.dim)*np.abs(self.X[i,:]-worst)

        final_f = fitness_fn(self.best_x)
        return self.best_x, final_f

# ============ 5）适应度包装函数 ============

DIM = 4 * len(customers)

def decode_and_evaluate(x):
    n = len(customers)
    route_key = x[:n]
    truck_key = x[n:2*n]
    L_key     = x[2*n:3*n]
    R_key     = x[3*n:4*n]
    route, sorties = keys_to_plan(route_key, truck_key, L_key, R_key, rng=np.random.default_rng())
    total_cost, _, _, _ = check_and_cost(route, sorties)
    return total_cost

# ============ 6）运行SSA算法 ============

ssa = SSA(dim=DIM, n_pop=50, max_iter=400, p_producer=0.2, p_awareness=0.1, seed=0)
best_x, best_f = ssa.optimize(decode_and_evaluate)

n = len(customers)
rk = best_x[:n]; tk = best_x[n:2*n]; lk = best_x[2*n:3*n]; rk2 = best_x[3*n:4*n]
best_route, best_sorties = keys_to_plan(rk, tk, lk, rk2, rng=np.random.default_rng(1))
best_total, best_truck_cost, best_drone_cost, best_penalty = check_and_cost(best_route, best_sorties)

print("=== 最优解结果 ===")
print(f"总成本: {best_total:.3f}  (车辆: {best_truck_cost:.3f}, 无人机: {best_drone_cost:.3f}, 罚分: {best_penalty:.3f})")
print("车辆路线:", best_route)
print("无人机任务 (起飞点索引, 客户编号, 会合点索引):", best_sorties)

# ============ 7）结果可视化 ============

def plot_plan(route, sorties, fname="best_plan.png"):
    X = points[:, 0]; Y = points[:, 1]
    plt.figure(figsize=(7,6))

    # 绘制车辆路径（蓝线）
    for a,b in zip(route[:-1], route[1:]):
        plt.plot([X[a],X[b]], [Y[a],Y[b]], 'b-', lw=2, alpha=0.8)
    plt.scatter(X[route], Y[route], c='b', s=20, label='车辆路径节点')
    plt.scatter(X[0], Y[0], c='gold', s=120, marker='*', label='Depot')

    # 绘制无人机任务（红虚线）
    for (li, j, rk) in sorties:
        i_node = route[li]
        k_node = route[rk]
        plt.plot([X[i_node], X[j]], [Y[i_node], Y[j]], 'r--', lw=1.5, alpha=0.9)
        plt.plot([X[j], X[k_node]], [Y[j], Y[k_node]], 'r--', lw=1.5, alpha=0.9)
        plt.scatter([X[j]],[Y[j]], c='r', s=40)

    plt.title("1 辆车 + 3 架无人机 协同配送路径（SSA优化）")
    plt.legend()
    plt.axis('equal'); plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()

plot_plan(best_route, best_sorties, "best_plan.png")
print("✅ 已保存图片：best_plan.png")
