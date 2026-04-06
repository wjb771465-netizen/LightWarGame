# ai/envs/observation.py
import numpy as np
from game import create_fog_view_for_player

# 假设 Region 和 GameState 的类型定义可以从 game 模块导入（用于类型提示）
# from game import Region, GameState

# --- 1. 从 game/loaddata.py 移入的静态图结构定义 ---

# 1-indexed 邻接表
_ADJACENT_LISTS = [
    None,
    [2, 3],  # 北京 (1)
    [1, 3],  # 天津 (2)
    [1, 2, 4, 5, 15, 16, 6],  # 河北 (3)
    [3, 5, 16, 27],  # 山西 (4)
    [3, 4, 6, 7, 8, 27, 28, 30, 31],  # 内蒙古 (5)
    [3, 5, 7],  # 辽宁 (6)
    [5, 6, 8],  # 吉林 (7)
    [7, 5],  # 黑龙江 (8)
    [10, 11],  # 上海 (9)
    [9, 11, 12, 15],  # 江苏 (10)
    [9, 10, 12, 13, 14],  # 浙江 (11)
    [10, 11, 14, 16, 15],  # 安徽 (12)
    [11, 14, 19],  # 福建 (13)
    [14, 15, 12, 16, 17, 18, 19],  # 江西 (14)
    [3, 10, 12, 16],  # 山东 (15)
    [3, 4, 12, 14, 15, 17, 27],  # 河南 (16)
    [16, 12, 14, 18, 22, 27],  # 湖北 (17)
    [14, 17, 19, 20, 22, 23, 24],  # 湖南 (18)
    [13, 14, 18, 20, 21],  # 广东 (19)
    [18, 19, 24, 25],  # 广西 (20)
    [19],  # 海南 (21)
    [17, 18, 23, 24, 27],  # 重庆 (22)
    [22, 18, 24, 25, 26, 27, 28, 29],  # 四川 (23)
    [22, 18, 23, 20, 25],  # 贵州 (24)
    [20, 24, 23, 26],  # 云南 (25)
    [25, 23, 29, 31],  # 西藏 (26)
    [4, 16, 17, 22, 23, 28, 30, 5],  # 陕西 (27)
    [27, 23, 29, 30, 31, 5],  # 甘肃 (28)
    [28, 23, 26, 31],  # 青海 (29)
    [27, 28, 5, 31],  # 宁夏 (30)
    [5, 28, 29, 30, 26]  # 新疆 (31)
]


# --- 2. 图结构辅助函数 ---

def get_adjacency_matrix(num_regions=31) -> np.ndarray:
    """
    生成 (N, N) 的邻接矩阵. N = 31.
    """
    adj_matrix = np.zeros((num_regions, num_regions), dtype=np.float32)
    for i in range(1, num_regions + 1):
        idx_i = i - 1  # 0-indexed
        for neighbor in _ADJACENT_LISTS[i]:
            idx_j = neighbor - 1  # 0-indexed
            adj_matrix[idx_i, idx_j] = 1
            adj_matrix[idx_j, idx_i] = 1  # 确保对称（无向图）
    return adj_matrix


def get_edge_index(num_regions=31) -> np.ndarray:
    """
    生成 GNN (PyTorch Geometric) 兼容的 edge_index.
    形状为 [2, E], E 是边的总数 (无向图, 包含双向).
    """
    adj_matrix = get_adjacency_matrix(num_regions)
    # 找到所有非零元素的索引 (即, 边的位置)
    rows, cols = np.where(adj_matrix > 0)
    # 堆叠成 [2, E] 格式
    edge_index = np.stack([rows, cols], axis=0).astype(np.int64)
    return edge_index


# --- 3. 修改后的 Observation Builder ---

class FogToGraphFeatures:
    """
    将战争迷雾 (fog_view) 转换为图的节点特征矩阵 (Node Feature Matrix).
    输出形状为 (N, F), 即 (31, 4).
    """

    def __init__(self):
        self.num_regions = 31
        self.channels = 4  # F: 4 个特征 (is_mine, is_enemy, norm_troops, norm_growth)
        self.max_growth = 8
        self.output_shape = (self.num_regions, self.channels)

    def _get_max_troops(self, regions) -> int:
        max_t = 1
        for r in regions[1:]:
            if r.owner > 0 and r.troops > 0:
                max_t = max(max_t, r.troops)
        return max_t

    def _normalize_troops(self, troops: int, max_t: int) -> float:
        return min(troops / max_t, 1.0) if max_t > 0 and troops > 0 else 0.0

    def _normalize_growth(self, growth: int) -> float:
        return growth / self.max_growth

    def __call__(self, game_state, player_id):
        # type: (GameState, int) -> np.ndarray
        """
        生成 (31, 4) 的节点特征矩阵.
        """
        fog_view = create_fog_view_for_player(game_state, player_id)
        regions = fog_view["regions"]
        max_troops = self._get_max_troops(regions)

        # 节点特征矩阵, 形状 (N, F) = (31, 4)
        node_features = np.zeros(self.output_shape, dtype=np.float32)

        for rid in range(1, 32):
            idx = rid - 1  # 0-indexed
            r = regions[rid]

            if r.owner == 0:  # 0: 无主隐藏 (Fog of War)
                continue

            if r.owner == player_id:
                node_features[idx, 0] = 1.0  # 特征 0: is_mine
                node_features[idx, 2] = self._normalize_troops(r.troops, max_troops)  # 特征 2: norm_troops
                node_features[idx, 3] = self._normalize_growth(r.base_growth)  # 特征 3: norm_growth
            else:
                # 敌人领地 (owner > 0 and owner != player_id)
                node_features[idx, 1] = 1.0  # 特征 1: is_enemy
                # 备注: 正常的迷雾规则下, 你不应该能看到敌人的准确兵力和增长
                # 如果你的 fog_view 允许看到, 你也可以在这里填充 [idx, 2] 和 [idx, 3]

        return node_features