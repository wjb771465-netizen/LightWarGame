from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ai.algos.policy import SB3Policy
from ai.algos.region_pool import RegionPool
from ai.envs.env import LwgEnv
from ai.envs.opponents import PolicyOpponent
from ai.train.metrics import WinRateCallback
from ai.train.self_play_trainer import SelfPlayTrainer


class RegionSelfPlayTrainer(SelfPlayTrainer):
    """每个地区维护独立模型，支持多线程并行训练，对手从地区策略池中随机抽取。"""

    def __init__(self, args) -> None:
        super().__init__(args)
        raw = getattr(args, "region_self_play_regions", None)
        self.regions: list[int] = (
            list(range(1, 32)) if raw is None
            else [int(x.strip()) for x in raw.split(",")]
        )
        self.pool = RegionPool(history=args.self_play_pool_size)
        self._log_lock = threading.Lock()
        torch.set_num_threads(args.n_training_threads)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def train(self) -> None:
        self._init_logging()

        scenario = self.args.scenario
        envs: dict[int, VecMonitor] = {}
        agents: dict[int, SB3Policy] = {}
        win_cbs: dict[int, WinRateCallback] = {}

        for R in self.regions:
            os.makedirs(self._region_dir(R), exist_ok=True)
            env = VecMonitor(
                make_vec_env(lambda: LwgEnv(scenario), n_envs=self.args.n_envs, vec_env_cls=SubprocVecEnv, monitor_kwargs=None),
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

        # Chunk-interleaved with pre-snapshot: each round samples all
        # opponents BEFORE any region trains, so every region sees the
        # same pool state (previous round's checkpoints).  Then all
        # regions train one chunk and save.  This guarantees fair
        # cross-region self-play after the first cold-start round.
        while True:
            active = [R for R in self.regions if agents[R].num_timesteps < total]
            if not active:
                break

            # Phase 1: snapshot opponents — all regions sample from the
            #          SAME pool state (previous round's checkpoints).
            #          Store lightweight entries only; model loading is
            #          deferred to Phase 2 to parallelize I/O across workers.
            round_opponents: dict[int, tuple] = {}
            for R in active:
                result = self.pool.sample_opponent(
                    exclude_region=R,
                    strategy=self.args.pool_sampling_strategy,
                    lam=self.args.sampling_lam,
                    s=self.args.sampling_scale,
                    progress_D=self.args.progress_D,
                )
                if result is None:
                    opp_region = next(r for r in self.regions if r != R)
                    round_opponents[R] = (None, opp_region,
                                          f"rule(region={opp_region})")
                else:
                    opp_region, entry = result
                    round_opponents[R] = (entry, opp_region,
                                          f"policy(region={opp_region}, step={entry.step})")

            # Phase 2: train all regions with their pre-sampled opponents,
            #          parallelised via ThreadPoolExecutor.
            def _train_chunk(R: int) -> None:
                agent = agents[R]
                env = envs[R]
                win_cb = win_cbs[R]
                entry_or_none, opp_region, opp_label = round_opponents[R]
                steps = min(chunk, total - agent.num_timesteps)

                # Build opponent inside the worker so that model-load I/O
                # is parallelised across regions.
                if entry_or_none is None:
                    opp = self._make_warmup_opponent("rule", player_id=2)
                else:
                    opp = PolicyOpponent(
                        player_id=2,
                        policy=SB3Policy(path=entry_or_none.path),
                        obs_encoder=env.get_attr("obs_encoder")[0],
                        act_encoder=env.get_attr("act_encoder")[0],
                    )

                env.env_method("set_opponent", opp)
                env.env_method("set_capitals", R, opp_region)
                print(f"[RegionSP R={R}] round, agent_cap={R}, opp_cap={opp_region}, "
                      f"opponent={opp_label}")

                agent.learn(steps, callback=[win_cb])
                step = agent.num_timesteps

                ckpt = os.path.join(self._region_dir(R), f"ckpt_{step}")
                agent.save(ckpt)
                self.pool.add(R, ckpt + ".zip", step)

                t = win_cb._tracker
                self.log_metrics({
                    f"region_{R}/win_rate_global": t.win_rate_global,
                    f"region_{R}/win_rate_window": t.win_rate_window,
                }, step)

            max_workers = max(1, min(self.args.parallel_regions, len(active)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                fut_to_R = {executor.submit(_train_chunk, R): R for R in active}
                for f in as_completed(fut_to_R):
                    f.result()  # propagate first exception from any worker

            # Persist pool state after each full round
            self.pool.save(os.path.join(self.save_dir, "pool_state.json"))

        for R in self.regions:
            agents[R].save(os.path.join(self._region_dir(R), "final"))
        print(f"[RegionSelfPlay] 训练完成，模型保存至 {self.save_dir}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def log_metrics(self, metrics: dict, step: int) -> None:
        """Thread-safe metrics logging."""
        if self.args.wandb:
            import wandb
            with self._log_lock:
                wandb.log({k: v for k, v in metrics.items() if v is not None}, step=step)
        else:
            import logging
            logging.info("step=%d %s", step, metrics)

    def _region_dir(self, R: int) -> str:
        return os.path.join(self.save_dir, f"region_{R}")
