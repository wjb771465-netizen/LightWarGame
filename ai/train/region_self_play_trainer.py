from __future__ import annotations

import os
import random

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from ai.algos.policy import SB3Policy
from ai.algos.region_pool import RegionPool
from ai.envs.env import LwgEnv
from ai.envs.opponents import PolicyOpponent
from ai.train.metrics import WinRateCallback
from ai.train.self_play_trainer import SelfPlayTrainer


class RegionSelfPlayTrainer(SelfPlayTrainer):
    """每个地区维护独立模型，随机轮转训练，对手从地区策略池中随机抽取。"""

    def __init__(self, args) -> None:
        super().__init__(args)
        raw = getattr(args, "region_self_play_regions", None)
        self.regions: list[int] = (
            list(range(1, 32)) if raw is None
            else [int(x.strip()) for x in raw.split(",")]
        )
        self.pool = RegionPool(history=args.region_pool_history)

    def train(self) -> None:
        self._init_logging()

        scenario = self.args.scenario
        envs: dict[int, VecMonitor] = {}
        agents: dict[int, SB3Policy] = {}
        win_cbs: dict[int, WinRateCallback] = {}

        for R in self.regions:
            os.makedirs(self._region_dir(R), exist_ok=True)
            # TODO: 支持 --n-envs 传参；跨地区并行训练亦未实现
            env = VecMonitor(
                make_vec_env(lambda: LwgEnv(scenario), n_envs=8),
                info_keywords=("win", "turn"),
            )
            envs[R] = env
            agents[R] = SB3Policy(
                env=env, args=self.args,
                tb_log_dir=os.path.join(self._region_dir(R), "tb"),
            )
            win_cbs[R] = WinRateCallback(window=self.args.win_rate_window)

        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq

        while True:
            active = [R for R in self.regions if agents[R].num_timesteps < total]
            if not active:
                break
            # 随机轮转，避免固定顺序导致某些地区长期得不到训练
            R = random.choice(active)
            env = envs[R]
            agent = agents[R]

            result = self.pool.sample_opponent(exclude_region=R)
            if result is None:
                # 冷启动：池子还没有任何 checkpoint，用规则对手占位
                opp_region = next(r for r in self.regions if r != R)
                opp = self._make_warmup_opponent("rule", player_id=2)
            else:
                opp_region, entry = result
                opp = PolicyOpponent(
                    player_id=2,
                    policy=SB3Policy(path=entry.path),
                    obs_encoder=env.get_attr("obs_encoder")[0],
                    act_encoder=env.get_attr("act_encoder")[0],
                )

            env.env_method("set_opponent", opp)
            env.env_method("set_capitals", R, opp_region)

            steps = min(chunk, total - agent.num_timesteps)
            agent.learn(steps, callback=[win_cbs[R]])
            step = agent.num_timesteps

            ckpt = os.path.join(self._region_dir(R), f"ckpt_{step}")
            agent.save(ckpt)
            self.pool.add(R, ckpt + ".zip", step)
            self.pool.save(os.path.join(self.save_dir, "pool_state.json"))

            t = win_cbs[R]._tracker
            self.log_metrics({
                f"region_{R}/win_rate_global": t.win_rate_global,
                f"region_{R}/win_rate_window": t.win_rate_window,
            }, step)

        for R in self.regions:
            agents[R].save(os.path.join(self._region_dir(R), "final"))
        print(f"[RegionSelfPlay] 训练完成，模型保存至 {self.save_dir}")

    def _region_dir(self, R: int) -> str:
        return os.path.join(self.save_dir, f"region_{R}")
