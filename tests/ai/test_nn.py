"""
ai/algos/gnn.py 和 ai/algos/extractors.py 的单元测试。

纯张量测试，不依赖 SB3、游戏地图、YAML 配置。
"""

import unittest

import numpy as np
import torch
from gymnasium import spaces

from ai.algos.gnn import GNNBackbone, adj_to_edge_index
from ai.algos.extractors import GNNExtractor


class TestGNNBackbone(unittest.TestCase):

    def setUp(self):
        self.num_nodes = 4
        self.in_channels = 7
        # 不对称图: 0-1-2, 0-3
        #   节点 0 度 2（连 1,3），节点 2 度 1（只连 1），拓扑角色不同
        self.edge_index = torch.tensor(
            [[0, 1, 1, 2, 0, 3],
             [1, 0, 2, 1, 3, 0]], dtype=torch.long)
        self.model = GNNBackbone(
            num_nodes=self.num_nodes,
            in_channels=self.in_channels,
            hidden_channels=32,
            out_dim=64,
        )

    def test_forward_shape(self):
        """单 batch 和多 batch 输出 shape 正确。"""
        for B in (1, 4):
            x = torch.randn(B, self.num_nodes, self.in_channels)
            g = torch.randn(B, 2)
            out = self.model(x, self.edge_index, g)
            self.assertEqual(out.shape, (B, 64))

    def test_deterministic_in_eval(self):
        """eval 模式下同输入同输出。"""
        self.model.eval()
        x = torch.randn(2, self.num_nodes, self.in_channels)
        g = torch.randn(2, 2)
        with torch.no_grad():
            out1 = self.model(x, self.edge_index.clone(), g.clone())
            out2 = self.model(x, self.edge_index.clone(), g.clone())
        self.assertTrue(torch.allclose(out1, out2, atol=1e-6))

    def test_message_passing_has_effect(self):
        """交换拓扑角色不同的两个节点全部特征 → 输出应不同。"""
        self.model.eval()
        x = torch.randn(1, self.num_nodes, self.in_channels)

        # node 0（度 2）←→ node 2（度 1）整行互换
        x_swapped = x.clone()
        x_swapped[0, 0, :], x_swapped[0, 2, :] = x[0, 2, :].clone(), x[0, 0, :].clone()

        g = torch.randn(1, 2)
        with torch.no_grad():
            out = self.model(x, self.edge_index, g.clone())
            out_swapped = self.model(x_swapped, self.edge_index, g.clone())
        self.assertFalse(torch.allclose(out, out_swapped, atol=1e-4))

    def test_batch_edge_index_shape(self):
        """_batch_edge_index 输出 shape 为 (2, B*E)。"""
        E = self.edge_index.shape[1]
        batched = GNNBackbone._batch_edge_index(self.edge_index, batch_size=4, num_nodes=5)
        self.assertEqual(batched.shape, (2, 4 * E))

    def test_batch_edge_index_values(self):
        """batch=2 时第一样本 offset=0，第二样本 offset=N。"""
        N = 5
        ei = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
        batched = GNNBackbone._batch_edge_index(ei, batch_size=2, num_nodes=N)
        expected = torch.tensor([[0, 2, 5, 7],
                                  [2, 0, 7, 5]], dtype=torch.long)
        self.assertTrue(torch.equal(batched, expected))


class TestAdjToEdgeIndex(unittest.TestCase):
    """adj_to_edge_index：邻接矩阵 → PyG edge_index 格式转换。"""

    def test_simple_chain(self):
        adj = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ], dtype=np.float32)
        ei = adj_to_edge_index(adj)
        self.assertEqual(ei.shape[1], 4)  # 2 条无向边 → 4 条有向边
        # 包含双向边
        edges = {(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])}
        self.assertIn((0, 1), edges)
        self.assertIn((1, 0), edges)
        self.assertIn((1, 2), edges)
        self.assertIn((2, 1), edges)

    def test_disconnected_graph(self):
        adj = np.zeros((3, 3), dtype=np.float32)
        ei = adj_to_edge_index(adj)
        self.assertEqual(ei.shape[1], 0)  # 无边

    def test_dtype_is_long(self):
        adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
        ei = adj_to_edge_index(adj)
        self.assertEqual(ei.dtype, torch.long)


class TestGNNExtractor(unittest.TestCase):
    """GNNExtractor：验证 flat obs → node_feats + global_feats 拆分正确。"""

    def setUp(self):
        self.num_regions = 3
        self.feat_dim = 5
        self.global_dim = 2
        self.obs_dim = self.num_regions * self.feat_dim + self.global_dim
        self.obs_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
        self.edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    def test_forward_shape(self):
        """输出 shape 为 (B, features_dim)。"""
        ext = GNNExtractor(self.obs_space, features_dim=64,
                           num_regions=self.num_regions, feat_dim=self.feat_dim,
                           global_dim=self.global_dim, edge_index=self.edge_index,
                           hidden_channels=32)
        obs = torch.randn(4, self.obs_dim)
        out = ext(obs)
        self.assertEqual(out.shape, (4, 64))

    def test_node_feats_reshape_correct(self):
        """验证前 15 维被 reshape 为 (B, 3, 5)，后 2 维为 global。"""
        ext = GNNExtractor(self.obs_space, features_dim=64,
                           num_regions=self.num_regions, feat_dim=self.feat_dim,
                           global_dim=self.global_dim, edge_index=self.edge_index,
                           hidden_channels=32)
        obs = torch.randn(2, self.obs_dim)
        # 把 node 部分填成可预测值，验证 reshape 不做乱序
        obs[0, :-2] = torch.arange(self.num_regions * self.feat_dim, dtype=torch.float32)
        obs[0, -2:] = torch.tensor([99.0, 100.0])
        ext.eval()
        with torch.no_grad():
            out = ext(obs)
        self.assertEqual(out.shape, (2, 64))


if __name__ == "__main__":
    unittest.main()
