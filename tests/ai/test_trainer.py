"""
ai/train/sb3_trainer.py 的单元测试。

验证：MaskablePPO 能跑通短训练、args 字段可读、checkpoint 能写入磁盘。
"""
import logging
import os
import random
import tempfile
import unittest

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

from ai.train.args import get_config
from ai.envs.env import LwgEnv
from ai.envs.utils import parse_config

SCENARIO = "duel/vsbaseline"
SHORT_STEPS = 1024


class TestMaskablePPOIntegration(unittest.TestCase):

    def test_short_train_does_not_crash(self):
        """MaskablePPO + LwgEnv 跑 SHORT_STEPS 步不崩溃。"""
        env = make_vec_env(lambda: LwgEnv(SCENARIO), n_envs=1)
        model = MaskablePPO(
            "MlpPolicy", env,
            n_steps=SHORT_STEPS,
            batch_size=64,
            n_epochs=1,
            verbose=0,
        )
        model.learn(total_timesteps=SHORT_STEPS)
        self.assertIsNotNone(model.policy)

    def test_args_scenario_parsed(self):
        """get_config 解析 --scenario 后，save_dir 默认含 ai/train/results/<scenario>/run_ 前缀。"""
        from ai.train.utils import resolve_save_dir
        parser = get_config()
        args = parser.parse_args(["--scenario", SCENARIO])
        self.assertEqual(args.scenario, SCENARIO)
        self.assertIsNone(args.save_dir)
        resolved = resolve_save_dir(args.scenario, args.save_dir)
        expected_prefix = os.path.join("ai", "train", "results", SCENARIO, "run_")
        self.assertTrue(resolved.startswith(expected_prefix), resolved)

    def test_env_config_loaded(self):
        """parse_config 能读取 vsbaseline.yaml，training.opponent 字段存在。"""
        cfg = parse_config(SCENARIO)
        self.assertEqual(cfg.training.opponent, "random")
        self.assertEqual(cfg.game.num_players, 2)
        self.assertEqual(cfg.game.max_players, 6)

    def test_checkpoint_saved(self):
        """训练 SHORT_STEPS 步后，save_dir 内有 .zip checkpoint 文件。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            from stable_baselines3.common.callbacks import CheckpointCallback

            env = make_vec_env(lambda: LwgEnv(SCENARIO), n_envs=1)
            model = MaskablePPO(
                "MlpPolicy", env,
                n_steps=SHORT_STEPS,
                batch_size=64,
                n_epochs=1,
                verbose=0,
            )
            cb = CheckpointCallback(
                save_freq=SHORT_STEPS,
                save_path=tmpdir,
                name_prefix="lwg_ppo",
            )
            model.learn(total_timesteps=SHORT_STEPS, callback=cb)

            zips = [f for f in os.listdir(tmpdir) if f.endswith(".zip")]
            self.assertTrue(len(zips) > 0, f"没有找到 .zip 文件，tmpdir 内容：{os.listdir(tmpdir)}")


class TestSelfPlayArgs(unittest.TestCase):
    """验证新增的采样策略 CLI 参数解析。"""

    def test_default_strategy_is_latest(self):
        parser = get_config()
        args = parser.parse_args(["--scenario", "duel/selfplay", "--self-play"])
        self.assertEqual(args.pool_sampling_strategy, "latest")

    def test_strategy_choices(self):
        parser = get_config()
        for strategy in ("latest", "uniform", "progress", "elo"):
            args = parser.parse_args([
                "--scenario", "duel/selfplay", "--self-play",
                "--pool-sampling-strategy", strategy,
            ])
            self.assertEqual(args.pool_sampling_strategy, strategy)

    def test_sampling_lam_default(self):
        parser = get_config()
        args = parser.parse_args(["--scenario", "duel/selfplay", "--self-play"])
        self.assertEqual(args.sampling_lam, 1.0)

    def test_sampling_scale_default(self):
        parser = get_config()
        args = parser.parse_args(["--scenario", "duel/selfplay", "--self-play"])
        self.assertEqual(args.sampling_scale, 100.0)

    def test_progress_D_default(self):
        parser = get_config()
        args = parser.parse_args(["--scenario", "duel/selfplay", "--self-play"])
        self.assertIsNone(args.progress_D)

    def test_pool_size_default_is_20(self):
        parser = get_config()
        args = parser.parse_args(["--scenario", "duel/selfplay", "--self-play"])
        self.assertEqual(args.self_play_pool_size, 20)


class TestFixedOpponentSpecs(unittest.TestCase):
    """_fixed_opponent_specs 边界行为测试。"""

    def setUp(self):
        from argparse import Namespace
        self.args = Namespace(
            scenario="duel/vsbaseline",
            seed=42,
            n_envs=1,
            win_rate_window=100,
            eval_opponent="random,rule,fsm",
        )
        self.args.save_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.args.save_dir, ignore_errors=True)

    def _make_trainer(self, eval_opponent: str):
        from unittest.mock import patch
        from ai.train.sb3_trainer import Sb3Trainer
        self.args.eval_opponent = eval_opponent
        # Sb3Trainer.__init__ 会调用 create_envs()/create_agent()，
        # 但 _fixed_opponent_specs 测试不需要它们 → mock 掉
        with patch.object(Sb3Trainer, "create_envs", return_value=(None, None)), \
             patch.object(Sb3Trainer, "create_agent", return_value=None):
            trainer = Sb3Trainer(self.args)
            trainer.env = None  # create_envs mock 返回了 (None, None)
            trainer.eval_env = None
            trainer.agent = None
        return trainer

    def test_types_equal_envs_one_to_one(self):
        """类型数 == 进程数：一一对应。"""
        trainer = self._make_trainer("random,rule")
        with self.assertNoLogs(level=logging.WARNING):
            specs = trainer._fixed_opponent_specs(2, opponent_id=2)
        self.assertEqual(len(specs), 2)
        self.assertEqual([s["type"] for s in specs], ["random", "rule"])

    def test_types_more_than_envs_random_sample(self):
        """类型数 > 进程数：random.sample 随机选，warning 列出入选类型。"""
        trainer = self._make_trainer("random,rule,fsm")
        random.seed(99)
        with self.assertLogs(level=logging.WARNING) as ctx:
            specs = trainer._fixed_opponent_specs(2, opponent_id=2)
        self.assertEqual(len(specs), 2)
        self.assertIn("随机选 2 种", ctx.output[0])
        # 用同一 seed 验证确定性
        random.seed(99)
        specs2 = trainer._fixed_opponent_specs(2, opponent_id=2)
        self.assertEqual([s["type"] for s in specs], [s["type"] for s in specs2])
        # 选中的都属于原集合
        for s in specs:
            self.assertIn(s["type"], ("random", "rule", "fsm"))

    def test_types_fewer_than_envs_cycle_fill(self):
        """类型数 < 进程数：循环填充至满，warning 输出。"""
        trainer = self._make_trainer("random,rule")
        with self.assertLogs(level=logging.WARNING) as ctx:
            specs = trainer._fixed_opponent_specs(5, opponent_id=2)
        self.assertEqual(len(specs), 5)
        self.assertIn("循环填充至 5", ctx.output[0])
        # 前 4 个是两轮 cycle
        self.assertEqual([s["type"] for s in specs[:4]],
                         ["random", "rule", "random", "rule"])


class TestGNNConfigConflict(unittest.TestCase):
    """--use-gnn + YAML use_adjacency: true → 报错退出。"""

    def test_conflict_raises_value_error(self):
        parser = get_config()
        # vsbaseline.yaml 有 use_adjacency: true
        args = parser.parse_args([
            "--scenario", "duel/vsbaseline", "--use-gnn",
            "--total-timesteps", "512",
        ])
        from ai.train.sb3_trainer import Sb3Trainer
        with self.assertRaises(ValueError) as ctx:
            Sb3Trainer(args)
        self.assertIn("use_adjacency", str(ctx.exception))
