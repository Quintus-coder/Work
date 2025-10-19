import numpy as np
import matplotlib.pyplot as plt

# ========== 中文字体配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------- 参数设置 ----------------
NUM_DRONES = 3  # 无人机数量
DRONE_SPEED = 40.0  # 无人机速度
TRUCK_SPEED = 30.0  # 卡车速度
DRONE_COST_PER_UNIT_DISTANCE = 0.4
TRUCK_COST_PER_UNIT_DISTANCE = 0.2

# 仓库位置 (纬度, 经度)
truck_position = np.array([22.584514, 113.131083])

# 33 个任务点 (纬度, 经度)
tasks = np.array([
    [22.609350, 113.110278], [22.597057, 113.122096], [22.608260, 113.084365],
    [22.590445, 113.155086], [22.575208, 113.096191], [22.588793, 113.095735],
    [22.612061, 113.086892], [22.589965, 113.128916], [22.610474, 113.078767],
    [22.594341, 113.095619], [22.607121, 113.056570], [22.622576, 113.094734],
    [22.586587, 113.089055], [22.580411, 113.125352], [22.583903, 113.092061],
    [22.603238, 113.065627], [22.583785, 113.129778], [22.567947, 113.072823],
    [22.597246, 113.128330], [22.611808, 113.120081], [22.602605, 113.123606],
    [22.622505, 113.089523], [22.581607, 113.103356], [22.599589, 113.078248],
    [22.567604, 113.071638], [22.586437, 113.068015], [22.571884, 113.082405],
    [22.582698, 113.125646], [22.616603, 113.080720], [22.566321, 113.066926],
    [22.590046, 113.166816], [22.634591, 113.112116], [22.607044, 113.056159],
])

# ---------------- 数据准备 ----------------
points = np.vstack([truck_position, tasks])
n_all = len(points)
customers = list(range(0, n_all))

def euclidean_distance(p1, p2):                         # TODO:距离的计算
    return float(np.linalg.norm(p1 - p2))

def build_distance_matrix():
    D = np.zeros((n_all, n_all), dtype=float)
    for i in range(n_all):
        for j in range(n_all):
            if i != j:
                D[i, j] = euclidean_distance(points[i], points[j])
    return D


DIST_MATRIX = build_distance_matrix()


# ---------------- 解码辅助函数 ----------------
def sigmoid(x):
    x = np.clip(x, -50, 50)  # 限制输入范围，防止 exp 溢出
    return 1.0 / (1.0 + np.exp(-x))


def keys_to_plan(route_key, truck_key, L_key, R_key, rng):
    order = np.argsort(route_key)
    truck_mask = sigmoid(truck_key) > 0.5
    if truck_mask.sum() == 0:
        truck_mask[rng.integers(0, len(customers))] = True

    truck_nodes = [customers[i] for i in order if truck_mask[i]]
    route = [0] + truck_nodes + [0]

    drone_nodes = [customers[i] for i in order if not truck_mask[i]]

    sorties = []
    m = len(route)
    if m < 2:
        route = [0, 0]
        m = 2
    for j in drone_nodes:
        li = int(np.clip(int((m - 1) * abs(np.tanh(L_key[j - 1]))), 0, m - 2))
        rk_min = li + 1
        rk = rk_min + int(
            np.clip(
                int((m - 1 - rk_min) * abs(np.tanh(R_key[j - 1]))),
                0,
                max(0, (m - 1 - rk_min)),
            )
        )
        sorties.append((li, j, rk))

    return route, sorties


# ---------------- 代价计算 ----------------
def route_length(route):
    return sum(DIST_MATRIX[a, b] for a, b in zip(route[:-1], route[1:]))


def arrival_times(route):
    t = [0.0]
    for a, b in zip(route[:-1], route[1:]):
        t.append(t[-1] + float(DIST_MATRIX[a, b]) / TRUCK_SPEED)
    return np.array(t)


def check_and_cost(route, sorties):
    BIG = 1e6
    penalty = 0.0

    truck_cost = TRUCK_COST_PER_UNIT_DISTANCE * route_length(route)
    t_arr = arrival_times(route)

    intervals = []
    drone_cost = 0.0
    truck_served = set(route[1:-1])
    drone_served = set()

    for (li, j, rk) in sorties:
        if not (0 <= li < len(route) - 1 and li < rk < len(route)):
            penalty += BIG
            continue
        i_node, k_node = route[li], route[rk]
        flight = DIST_MATRIX[i_node, j] + DIST_MATRIX[j, k_node]
        t_need = flight / DRONE_SPEED
        t_have = t_arr[rk] - t_arr[li]
        if t_have + 1e-9 < t_need:
            penalty += BIG * (t_need - t_have)
        if j in truck_served:
            penalty += BIG
        intervals.append((t_arr[li], t_arr[rk]))
        drone_cost += DRONE_COST_PER_UNIT_DISTANCE * flight
        drone_served.add(j)

    all_served = truck_served.union(drone_served)
    missing = set(customers) - all_served
    if missing:
        penalty += BIG * len(missing)

    events = []
    for s, e in intervals:
        events.append((s, +1))
        events.append((e, -1))
    events.sort()
    active = 0
    for _, ev in events:
        active += ev
        if active > NUM_DRONES:
            penalty += BIG * (active - NUM_DRONES)

    total_cost = truck_cost + drone_cost + penalty
    return total_cost, truck_cost, drone_cost, penalty


# ---------------- 麻雀搜索算法 ----------------
class SSA:
    def __init__(self, dim, n_pop=40, max_iter=300, p_producer=0.2, p_awareness=0.1, seed=42):
        self.dim = dim
        self.n = n_pop
        self.T = max_iter
        self.nP = max(1, int(p_producer * n_pop))
        self.nA = max(1, int(p_awareness * n_pop))
        self.rng = np.random.default_rng(seed)
        self.X = self.rng.normal(0, 1, size=(self.n, self.dim))
        self.F = np.full(self.n, np.inf)
        self.best_x = None
        self.best_f = np.inf

    def optimize(self, fitness_fn):
        for t in range(self.T):
            for i in range(self.n):
                self.F[i] = fitness_fn(self.X[i, :])
                if self.F[i] < self.best_f:
                    self.best_f = self.F[i]
                    self.best_x = self.X[i, :].copy()

            idx = np.argsort(self.F)
            self.X = self.X[idx, :]
            self.F = self.F[idx]
            best = self.X[0, :].copy()
            worst = self.X[-1, :].copy()

            r2 = self.rng.uniform(0, 1)
            for i in range(self.nP):
                if r2 < 0.8:
                    self.X[i, :] = self.X[i, :] * np.exp(-(i) / self.T) + self.rng.normal(0, 1, size=self.dim)
                else:
                    self.X[i, :] = self.X[i, :] + self.rng.normal(0, 1, size=self.dim) * (self.X[i, :] - best)

            for i in range(self.nP, self.n):
                self.X[i, :] = self.rng.normal(0, 1, size=self.dim) * np.abs(self.X[i, :] - best)

            aware_idx = self.rng.choice(self.n, size=self.nA, replace=False)
            for i in aware_idx:
                self.X[i, :] = best + self.rng.normal(0, 1, size=self.dim) * np.abs(self.X[i, :] - worst)

        final_f = fitness_fn(self.best_x)
        return self.best_x, final_f


# ---------------- 适应度函数 ----------------
DIM = 4 * len(customers)


def decode_and_evaluate(x):
    n = len(customers)
    route_key = x[:n]
    truck_key = x[n:2 * n]
    L_key = x[2 * n:3 * n]
    R_key = x[3 * n:4 * n]
    route, sorties = keys_to_plan(route_key, truck_key, L_key, R_key, rng=np.random.default_rng())
    total_cost, _, _, _ = check_and_cost(route, sorties)
    return total_cost


# ---------------- 可视化 ----------------
def plot_plan(route, sorties, fname="ssa_plan.png"):
    X = points[:, 0]
    Y = points[:, 1]
    plt.figure(figsize=(9, 7))

    for a, b in zip(route[:-1], route[1:]):
        plt.plot([X[a], X[b]], [Y[a], Y[b]], color='#d62728', lw=2, alpha=0.7)
    plt.scatter(X[route], Y[route], c='#d62728', s=40, label='卡车路径节点')
    plt.scatter(X[0], Y[0], c='gold', s=120, marker='*', label='仓库')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    styles = ['-', '--', ':']
    for idx, (li, j, rk) in enumerate(sorties):
        color = colors[idx % len(colors)]
        style = styles[idx % len(styles)]
        i_node = route[li]
        k_node = route[rk]
        plt.plot([X[i_node], X[j]], [Y[i_node], Y[j]], linestyle=style, color=color, lw=2, alpha=0.9)
        plt.plot([X[j], X[k_node]], [Y[j], Y[k_node]], linestyle=style, color=color, lw=2, alpha=0.9)
        plt.scatter([X[j]], [Y[j]], c=color, s=50)

    for idx, node in enumerate(route):
        plt.text(X[node], Y[node], f"T{idx}", fontsize=8, color='black',
                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle="circle,pad=0.2"),
                 ha='center', va='center')

    plt.title("麻雀搜索算法 - 车辆与无人机协同配送路径")
    plt.xlabel("纬度")
    plt.ylabel("经度")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


# ---------------- 主程序 ----------------
if __name__ == "__main__":
    ssa = SSA(dim=DIM, n_pop=60, max_iter=400, p_producer=0.2, p_awareness=0.1, seed=0)
    best_x, best_f = ssa.optimize(decode_and_evaluate)

    n = len(customers)
    rk = best_x[:n]
    tk = best_x[n:2 * n]
    lk = best_x[2 * n:3 * n]
    rk2 = best_x[3 * n:4 * n]

    best_route, best_sorties = keys_to_plan(rk, tk, lk, rk2, rng=np.random.default_rng(1))
    best_total, best_truck_cost, best_drone_cost, best_penalty = check_and_cost(best_route, best_sorties)

    print("=== 麻雀搜索算法优化结果 ===")
    print(f"总成本: {best_total:.4f} (车辆: {best_truck_cost:.4f}, 无人机: {best_drone_cost:.4f}, 罚分: {best_penalty:.4f})")
    print("卡车路线:", best_route)
    print("无人机任务 (起飞点索引, 客户编号, 会合点索引):", best_sorties)

    plot_plan(best_route, best_sorties, "ssa_plan.png")
    print("✅ 已保存路径图: ssa_plan.png")