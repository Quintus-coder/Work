import numpy as np
import matplotlib.pyplot as plt

# ========== 中文乱码修复配置 ==========
# 设置中文字体（例如 SimHei 黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像时负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# ====================================

# ---------------- 参数设置 ----------------
NUM_DRONES = 3  # 三架无人机
DRONE_SPEED = 40.0  # 无人机速度
TRUCK_SPEED = 30.0  # 卡车速度
COST_PER_UNIT_DISTANCE = 0.4
TRUCK_COST_PER_UNIT_DISTANCE = 0.2
TARGET_DRONE_RATIO = 0.3  # 无人机期望执行比例
ALPHA = 0.00001  # 动态平衡因子

# 麻雀算法参数
POPULATION_SIZE = 50
MAX_ITERATIONS = 200
PRODUCER_RATIO = 0.2
AWARE_RATIO = 0.1
SAFETY_THRESHOLD = 0.6
LOWER_BOUND = 0.0
UPPER_BOUND = 1.0

# 仓库位置
truck_position = np.array([22.584514, 113.131083])  # (纬度, 经度)

# 33个任务点
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
NUM_TASKS = len(tasks)

# 无人机颜色列表
DRONE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝、橙、绿
TRUCK_COLOR = '#d62728'  # 红色
LAUNCH_MARKER_COLOR = 'darkred'
RENDEZVOUS_MARKER_COLOR = 'black'

# ***路径偏移参数 (增大偏移量)***
OFFSET_MAGNITUDE = 0.0006  # 经纬度偏移量，用于路径分离

# ***绘图线宽和透明度调整***
TRUCK_LINE_WIDTH = 1.0  # 卡车路径线宽减小
TRUCK_ALPHA = 0.4  # 卡车路径透明度提高
DRONE_LINE_WIDTH = 2.0  # 无人机路径线宽增大
DRONE_ALPHA = 1.0  # 无人机路径不透明


# ---------------- 基础函数 ----------------
def distance(p1, p2):   return float(np.linalg.norm(p1 - p2))


def cost_drone(truck_pos, task):
    """无人机往返成本和卡车新位置"""
    to_task_distance = distance(truck_pos, task)
    to_task_time = to_task_distance / DRONE_SPEED
    truck_move_distance = TRUCK_SPEED * to_task_time
    direction = task - truck_pos
    if np.linalg.norm(direction) > 0:
        direction /= np.linalg.norm(direction)
    new_truck_pos = truck_pos + direction * truck_move_distance
    return_distance = distance(task, new_truck_pos)
    total_distance = to_task_distance + return_distance
    total_cost = total_distance * COST_PER_UNIT_DISTANCE
    return total_cost, new_truck_pos, total_distance / DRONE_SPEED


def cost_truck(truck_pos, task):
    """卡车前往任务点成本和卡车新位置"""
    dist = distance(truck_pos, task)
    return dist * TRUCK_COST_PER_UNIT_DISTANCE, task.copy(), dist / TRUCK_SPEED


# ---------------- 节点类 ----------------
class Node:
    def __init__(self, visited_mask, truck_pos, cost, path, drone_count):
        self.visited_mask = visited_mask
        self.truck_pos = truck_pos
        self.cost = cost
        self.path = path
        self.drone_count = drone_count

    def __lt__(self, other):
        return self.cost < other.cost


# ---------------- 麻雀算法相关函数 ----------------
def decode_solution(position):
    """将连续位置解码为任务顺序和执行模式"""
    order_scores = position[:NUM_TASKS]
    mode_scores = position[NUM_TASKS: 2 * NUM_TASKS]
    order = list(np.argsort(order_scores))
    modes = ['drone' if mode_scores[i] > 0.5 else 'truck' for i in range(NUM_TASKS)]
    return order, modes


def simulate_sequence(order, modes):
    """根据顺序与模式模拟一次配送过程，返回最终节点与适应度信息"""
    visited_mask = 0
    truck_pos = truck_position.copy()
    total_cost = 0.0
    path = []
    drone_count = 0

    for task_idx in order:
        task = tasks[task_idx]
        if modes[task_idx] == 'drone':
            step_cost, new_pos, _ = cost_drone(truck_pos, task)
            drone_count += 1
        else:
            step_cost, new_pos, _ = cost_truck(truck_pos, task)

        total_cost += step_cost
        path.append((modes[task_idx], task_idx, total_cost))
        truck_pos = new_pos
        visited_mask |= (1 << task_idx)

    back_cost, truck_pos, _ = cost_truck(truck_pos, truck_position)
    total_cost += back_cost
    path.append(("truck_back", -1, total_cost))

    drone_ratio = drone_count / max(1, NUM_TASKS)
    balance_penalty = abs(drone_ratio - TARGET_DRONE_RATIO) * ALPHA * total_cost
    fitness = total_cost + balance_penalty

    node = Node(visited_mask, truck_pos, total_cost, path, drone_count)
    return fitness, node, total_cost


def evaluate_position(position):
    """评估给定位置的适应度"""
    order, modes = decode_solution(position)
    fitness, node, total_cost = simulate_sequence(order, modes)
    return fitness, node, total_cost


def sparrow_search():
    dim = NUM_TASKS * 2
    population = np.random.uniform(LOWER_BOUND, UPPER_BOUND, size=(POPULATION_SIZE, dim))
    fitness_list = np.zeros(POPULATION_SIZE)

    best_solution = None
    best_cost = float("inf")
    best_fitness = float("inf")

    for i in range(POPULATION_SIZE):
        fitness, node, total_cost = evaluate_position(population[i])
        fitness_list[i] = fitness
        if fitness < best_fitness:
            best_fitness = fitness
            best_cost = total_cost
            best_solution = node

    for iter_idx in range(MAX_ITERATIONS):
        sort_idx = np.argsort(fitness_list)
        population = population[sort_idx]
        fitness_list = fitness_list[sort_idx]

        producer_count = max(1, int(POPULATION_SIZE * PRODUCER_RATIO))
        aware_count = max(1, int(POPULATION_SIZE * AWARE_RATIO))

        global_best = population[0].copy()
        global_worst = population[-1].copy()

        for i in range(producer_count):
            r2 = np.random.rand()
            if r2 < SAFETY_THRESHOLD:
                population[i] = population[i] * np.exp(-i / (np.random.rand() * MAX_ITERATIONS + 1e-8))
            else:
                population[i] = population[i] + np.random.normal(size=dim)

        for i in range(producer_count, POPULATION_SIZE):
            if i > POPULATION_SIZE / 2:
                population[i] = np.random.normal(size=dim) * np.exp((global_worst - population[i]) / (i ** 2 + 1e-8))
            else:
                population[i] = global_best + np.abs(population[i] - global_best) * np.random.uniform(-1, 1, size=dim)

        aware_indices = np.random.choice(POPULATION_SIZE, aware_count, replace=False)
        for idx in aware_indices:
            if fitness_list[idx] > best_fitness:
                population[idx] = global_best + np.random.normal(size=dim) * np.abs(population[idx] - global_best)
            else:
                population[idx] = population[idx] + np.random.uniform(-1, 1, size=dim) * np.abs(population[idx] - global_worst)

        population = np.clip(population, LOWER_BOUND, UPPER_BOUND)

        for i in range(POPULATION_SIZE):
            fitness, node, total_cost = evaluate_position(population[i])
            fitness_list[i] = fitness
            if fitness < best_fitness:
                best_fitness = fitness
                best_cost = total_cost
                best_solution = node

    return best_solution, best_cost


# ---------------- 路径偏移辅助函数 ----------------
def get_offset_vector(p1, p2, magnitude):
    """计算垂直于 p1->p2 方向的单位向量，并乘以偏移量"""
    direction = p2 - p1
    if np.linalg.norm(direction) == 0:
        return np.array([0, 0])

    direction = direction / np.linalg.norm(direction)
    # 垂直向量 (交换 x, y 并取反其中一个)
    perpendicular = np.array([-direction[1], direction[0]])
    return perpendicular * magnitude


# ---------------- 主函数 ----------------
if __name__ == "__main__":
    best_solution, best_cost = sparrow_search()

    if not best_solution:
        print("❌ 未找到可行方案")
    else:
        print("\n✅ 麻雀算法 结果：")
        print(f"总成本: {best_cost:.4f} 元")

        # ---------- Matplotlib 可视化 (横纵坐标对调) ----------
        plt.figure(figsize=(9, 7))

        # 任务点: X轴(纬度 0), Y轴(经度 1)
        plt.scatter(tasks[:, 0], tasks[:, 1], c='blue', label='任务点', zorder=5)
        # 仓库点: X轴(纬度 0), Y轴(经度 1)
        plt.scatter(truck_position[0], truck_position[1], c='red', marker='s', s=100, label='仓库', zorder=7)

        # 准备图例句柄 (不变)
        legend_handles = [
            plt.Line2D([0], [0], color=TRUCK_COLOR, lw=TRUCK_LINE_WIDTH * 2, alpha=TRUCK_ALPHA, label='卡车连续路径'),
            plt.Line2D([0], [0], color=DRONE_COLORS[0], lw=DRONE_LINE_WIDTH, linestyle='-', label=f'无人机路径 (1)'),
            plt.Line2D([0], [0], color=DRONE_COLORS[1], lw=DRONE_LINE_WIDTH, linestyle='--', label=f'无人机路径 (2)'),
            plt.Line2D([0], [0], color=DRONE_COLORS[2], lw=DRONE_LINE_WIDTH, linestyle=':', label=f'无人机路径 (3)'),
        ]
        proxy_launch = plt.Line2D([0], [0], marker='x', color='w', markerfacecolor=LAUNCH_MARKER_COLOR,
                                  markeredgecolor=LAUNCH_MARKER_COLOR, markersize=8, label='无人机发射点 (A)')
        proxy_rendezvous = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=RENDEZVOUS_MARKER_COLOR,
                                      markeredgecolor=RENDEZVOUS_MARKER_COLOR, markersize=6, label='无人机回收点 (C)')

        legend_handles.extend([proxy_launch, proxy_rendezvous])

        current_pos = truck_position.copy()
        drone_task_counter = 0

        # --- 1. 收集卡车所有经过的点 (A点, C点, 卡车任务终点) ---
        truck_path_coords = [current_pos.copy()]
        drone_paths_data = []

        temp_current_pos = truck_position.copy()

        for step, (mode, task_id, _) in enumerate(best_solution.path, 1):

            if task_id == -1:
                task_coords = truck_position
            else:
                task_coords = tasks[task_id]

            A_pos = temp_current_pos.copy()

            # ---------------- 计算新位置 ----------------
            if mode == 'drone':
                _, new_pos, _ = cost_drone(temp_current_pos, task_coords)
                C_pos = new_pos.copy()

                # --- 计算偏移量 ---
                offset_vec = get_offset_vector(A_pos, C_pos, OFFSET_MAGNITUDE)

                # 无人机路径偏移
                A_offset = A_pos + offset_vec
                C_offset = C_pos + offset_vec
                B_offset = task_coords.copy()

                # 存储无人机路径数据
                drone_paths_data.append({
                    'A': A_pos, 'B': task_coords, 'C': C_pos,
                    'A_plot': A_offset, 'C_plot': C_offset, 'B_plot': B_offset,
                    'color': DRONE_COLORS[drone_task_counter % NUM_DRONES],
                    'style': ['-', '--', ':'][drone_task_counter % NUM_DRONES],
                    'task_id': task_id,
                    'step': step
                })

                drone_task_counter += 1

            elif mode == 'truck' or mode == 'truck_back':
                _, new_pos, _ = cost_truck(temp_current_pos, task_coords)
                C_pos = new_pos.copy()

            # --- 更新卡车路径点 ---
            truck_path_coords.append(C_pos.copy())
            temp_current_pos = C_pos

        # --- 2. 绘制连续的卡车路径 ---
        # 交换索引: truck_path_coords[: , 0] 是纬度 (新的X轴)
        #          truck_path_coords[: , 1] 是经度 (新的Y轴)
        truck_lat_x = [p[0] for p in truck_path_coords]
        truck_lon_y = [p[1] for p in truck_path_coords]

        plt.plot(truck_lat_x, truck_lon_y, color=TRUCK_COLOR, linewidth=TRUCK_LINE_WIDTH,
                 linestyle='-', alpha=TRUCK_ALPHA, zorder=2, label='_nolegend_')

        # --- 3. 绘制无人机路径、A/C点标记和任务编号 ---
        for data in drone_paths_data:
            # 绘制 A->B 路径: [纬度_A, 纬度_B], [经度_A, 经度_B]
            plt.plot([data['A_plot'][0], data['B_plot'][0]],
                     [data['A_plot'][1], data['B_plot'][1]],
                     color=data['color'], linewidth=DRONE_LINE_WIDTH, linestyle=data['style'], alpha=DRONE_ALPHA,
                     zorder=3)

            # 绘制 B->C 路径: [纬度_B, 纬度_C], [经度_B, 经度_C]
            plt.plot([data['B_plot'][0], data['C_plot'][0]],
                     [data['B_plot'][1], data['C_plot'][1]],
                     color=data['color'], linewidth=DRONE_LINE_WIDTH, linestyle=data['style'], alpha=DRONE_ALPHA,
                     zorder=3)

            # 标记 A 点: X轴(纬度), Y轴(经度)
            plt.scatter(data['A'][0], data['A'][1], marker='x', c=LAUNCH_MARKER_COLOR, s=50, zorder=6)

            # 标记 C 点: X轴(纬度), Y轴(经度)
            plt.scatter(data['C'][0], data['C'][1], marker='o', c=RENDEZVOUS_MARKER_COLOR, s=30, zorder=6)

            # 任务编号 (在任务点 B 上): X轴(纬度), Y轴(经度)
            plt.text(data['B'][0], data['B'][1],
                     f"{data['step']}", fontsize=8, color='black',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="circle,pad=0.2"),
                     ha='center', va='center', zorder=6)

        # 标注卡车独立任务点的编号
        current_pos = truck_position.copy()
        for step, (mode, task_id, _) in enumerate(best_solution.path, 1):
            if task_id != -1 and mode == 'truck':
                _, new_pos, _ = cost_truck(current_pos, tasks[task_id])

                # X轴(纬度), Y轴(经度)
                plt.text(float(tasks[task_id][0]), float(tasks[task_id][1]),
                         f"{step}", fontsize=8, color='black',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle="circle,pad=0.2"),
                         ha='center', va='center', zorder=6)
                current_pos = new_pos
            elif mode == 'drone':
                _, new_pos, _ = cost_drone(current_pos, tasks[task_id])
                current_pos = new_pos
            elif mode == 'truck_back':
                _, new_pos, _ = cost_truck(current_pos, truck_position)
                current_pos = new_pos

        plt.title(f"麻雀算法 路径 (优化可视化 - X轴:纬度, Y轴:经度) 总成本 = {best_cost:.2f} 元")
        plt.xlabel("纬度")  # 交换标签
        plt.ylabel("经度")  # 交换标签

        all_handles = [
                          plt.scatter([], [], c='blue', label='任务点', zorder=5),
                          plt.scatter([], [], c='red', marker='s', s=100, label='仓库', zorder=7)
                      ] + legend_handles

        plt.legend(handles=all_handles, loc='best')
        plt.grid(True)
        plt.show()


        # ---------------- Folium 地图可视化 (生成html文件，地图坐标不变) ----------------
        # Folium 保持 (纬度, 经度) 的标准地理坐标
        def plot_sparrow_search_on_map(best_solution):
            # ... (Folium 部分保持不变，因为 Folium 必须使用 [纬度, 经度] 的标准格式)
            print("\n[交互式地图 Folium 部分保持标准地理坐标 (Y:纬度, X:经度)]")