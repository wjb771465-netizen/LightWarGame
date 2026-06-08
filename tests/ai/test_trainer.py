"""
ai/train/sb3_trainer.py 的单元测试。

验证：MaskablePPO 能跑通短训练、args 字段可读、checkpoint 能写入磁盘。
"""
import os
import tempfile
import unittest

from sb3_contrib import MaskablePPO
from stable_baselines3.common.env_util import make_vec_env

from ai.train.args import get_config
from ai.envs.env import LwgEnv
from ai.envs.utils import parse_config

SCENARIO = "1v1/vsbaseline"
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
        from ai.train.sb3_trainer import _resolve_save_dir
        parser = get_config()
        args = parser.parse_args(["--scenario", SCENARIO])
        self.assertEqual(args.scenario, SCENARIO)
        self.assertIsNone(args.save_dir)
        resolved = _resolve_save_dir(args)
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
        args = parser.parse_args(["--scenario", "1v1/selfplay", "--self-play"])
        self.assertEqual(args.pool_sampling_strategy, "latest")

    def test_strategy_choices(self):
        parser = get_config()
        for strategy in ("latest", "uniform", "progress", "elo"):
            args = parser.parse_args([
                "--scenario", "1v1/selfplay", "--self-play",
                "--pool-sampling-strategy", strategy,
            ])
            self.assertEqual(args.pool_sampling_strategy, strategy)

    def test_sampling_lam_default(self):
        parser = get_config()
        args = parser.parse_args(["--scenario", "1v1/selfplay", "--self-play"])
        self.assertEqual(args.sampling_lam, 1.0)

    def test_sampling_scale_default(self):
        parser = get_config()
        args = parser.parse_args(["--scenario", "1v1/selfplay", "--self-play"])
        self.assertEqual(args.sampling_scale, 100.0)

    def test_progress_D_default(self):
        parser = get_config()
        args = parser.parse_args(["--scenario", "1v1/selfplay", "--self-play"])
        self.assertIsNone(args.progress_D)

    def test_pool_size_default_is_20(self):
        parser = get_config()
        args = parser.parse_args(["--scenario", "1v1/selfplay", "--self-play"])
        self.assertEqual(args.self_play_pool_size, 20)
