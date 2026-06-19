from __future__ import annotations

import logging
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch

from ai.algos.policy import SB3Policy
from ai.algos.region_pool import RegionPool
from ai.train.metrics import WinRateCallback
from ai.train.self_play_trainer import SelfPlayTrainer
from ai.train.utils import (
    checkpoint_path,
    extract_ckpt_step,
    final_model_path,
    format_eval_specs,
    resolve_save_dir,
    set_seeds,
)


class RegionSelfPlayTrainer(SelfPlayTrainer):
    """每个地区维护独立模型，支持多线程并行训练，对手从地区策略池中随机抽取。"""

    def __init__(self, args) -> None:
        self.args = args
        self.save_dir = resolve_save_dir(args.scenario, args.save_dir)
        set_seeds(args.seed)

        raw = getattr(args, "region_self_play_regions", None)
        self._regions: list[int] = (
            list(range(1, 32)) if raw is None
            else [int(x.strip()) for x in raw.split(",")]
        )
        self.pool = RegionPool(history=args.self_play_pool_size)
        self._log_lock = threading.Lock()
        torch.set_num_threads(args.n_training_threads)

        self.save_dirs: dict[int, str] = {}
        self.envs: dict[int, object] = {}
        self.eval_envs: dict[int, object] = {}
        self.agents: dict[int, SB3Policy] = {}
        self._win_cbs: dict[int, WinRateCallback] = {}
        self._agent_elos: dict[int, float] = {}

        for R in self._regions:
            self.save_dirs[R] = os.path.join(self.save_dir, f"region_{R}")
            os.makedirs(self.save_dirs[R], exist_ok=True)
            self.envs[R], self.eval_envs[R] = self.create_envs()
            self.agents[R] = self.create_agent(self.envs[R], tb_log_dir=self.save_dirs[R])
            self._win_cbs[R] = WinRateCallback(window=self.args.win_rate_window)
            self._agent_elos[R] = 1200.0

    def _opponent_id(self) -> int:
        # 所有 env 共享同一 YAML 配置，agent_id 相同，读一次缓存即可。
        # 这里用第一个 region 的 env 只在 __init__ 阶段的单线程安全窗口内读取。
        if not hasattr(self, "_cached_opponent_id"):
            env = self.envs[self._regions[0]]
            agent_id: int = env.get_attr("agent_id")[0]
            num_players: int = env.get_attr("config")[0].game.num_players
            self._cached_opponent_id = next(p for p in range(1, num_players + 1) if p != agent_id)
        return self._cached_opponent_id


    def _sample_opponent_specs(self, pool, n_total: int, opponent_id: int,
                               exclude_region: int) -> list[dict]:
        """从 RegionPool 中有放回采样，spec 额外携带 opp_region。"""
        strategy = self.args.pool_sampling_strategy
        lam = self.args.sampling_lam
        scale = self.args.sampling_scale

        specs = []
        for _ in range(n_total):
            result = pool.sample_opponent(
                exclude_region=exclude_region, strategy=strategy,
                lam=lam, s=scale, progress_D=self.args.progress_D,
            )
            if result is not None:
                rid, entry = result
                specs.append({
                    "type": "policy", "player_id": opponent_id,
                    "path": entry.path.replace(".zip", ""),
                    "opp_region": rid,
                })
            else:
                specs.append({"type": self.args.self_play_initial_opponent,
                               "player_id": opponent_id})
        return specs


    def train(self) -> None:
        self._init_logging()

        n_opponents = self.args.n_opponents or self.args.n_envs
        n_envs = self.args.n_envs
        if n_envs % n_opponents != 0:
            raise ValueError(
                f"--n-envs ({n_envs}) 须为 --n-opponents ({n_opponents}) 的整数倍"
            )
        per_opponent = n_envs // n_opponents

        opponent_id = self._opponent_id()
        total = self.args.total_timesteps
        chunk = self.args.checkpoint_freq

        while True:
            active = [R for R in self._regions if self.agents[R].num_timesteps < total]
            if not active:
                break

            # Phase 1: snapshot opponents from SAME pool state
            round_specs: dict[int, list[dict]] = {}
            for R in active:
                round_specs[R] = self._sample_opponent_specs(
                    self.pool, n_opponents, opponent_id=opponent_id, exclude_region=R)

            # Phase 2: train all regions in parallel
            def _train_chunk(R: int) -> None:
                agent = self.agents[R]
                env = self.envs[R]
                win_cb = self._win_cbs[R]
                specs = round_specs[R]
                steps = min(chunk, total - agent.num_timesteps)

                opp_info = []
                for i, spec in enumerate(specs):
                    indices = list(range(i * per_opponent, (i + 1) * per_opponent))
                    if not indices:
                        continue
                    env.env_method("set_opponent", spec, indices=indices)
                    opp_region = spec.get("opp_region")
                    if opp_region is None:
                        opp_region = random.choice([r for r in self._regions if r != R])
                    env.env_method("set_capitals", R, opp_region, indices=indices)
                    if spec["type"] == "policy":
                        opp_info.append((opp_region, int(extract_ckpt_step(spec["path"]))))
                    else:
                        opp_info.append((opp_region, spec["type"]))
                info_str = ", ".join(f"R{r}->s{step}" if isinstance(step, int) else f"R{r}->{step}"
                                     for r, step in opp_info[:8])
                if len(opp_info) > 8:
                    info_str += ", ..."
                logging.info("[RegionSP R=%d] agent=%d, opps=[%s], envs=%d, per=%d",
                             R, R, info_str, n_envs, per_opponent)

                agent.learn(steps, callback=[win_cb])
                agent._model._custom_logger = True
                step = agent.num_timesteps

                ckpt = checkpoint_path(self.save_dirs[R], step)
                agent.save(ckpt)

                ckpt_zip = ckpt + ".zip"
                if self.args.use_eval:
                    results = self.eval(ckpt, step, region=R)
                    prev_elo = self._agent_elos[R]
                    evicted, self._agent_elos[R], accepted = self.pool.add(
                        R, ckpt_zip, step, elo=self._agent_elos[R], outcomes=results)
                    if accepted:
                        logging.info("[RegionSP R=%d] step=%d, ELO %.1f -> %.1f, 入池",
                                     R, step, prev_elo, self._agent_elos[R])
                    else:
                        logging.info("[RegionSP R=%d] step=%d, ELO %.1f -> %.1f, 跳过入池",
                                     R, step, prev_elo, self._agent_elos[R])
                else:
                    self.pool.add(R, ckpt_zip, step)

                if self.args.use_eval:
                    self.log_eval_metrics({"elo": self._agent_elos[R]}, step, region=R)

            max_workers = max(1, min(self.args.parallel_regions, len(active)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                fut_to_R = {executor.submit(_train_chunk, R): R for R in active}
                for f in as_completed(fut_to_R):
                    f.result()

            self.pool.save(os.path.join(self.save_dir, "pool_state.json"))

        for R in self._regions:
            final = final_model_path(self.save_dirs[R])
            self.agents[R].save(final)
        logging.info("[RegionSelfPlay] 训练完成，模型保存至 %s", self.save_dir)

        for R in self._regions:
            opp = random.choice([r for r in self._regions if r != R])
            self.render(final_model_path(self.save_dirs[R]),
                        save_dir=self.save_dirs[R], agent_capital=R, opponent_capital=opp)


    def eval(self, ckpt: str, step: int, region: int | None = None) -> list:
        R = region
        specs = self.choose_eval_opponents(region=R)
        if not specs:
            return []

        from ai.train.eval import evaluate, aggregate_win_rate, aggregate_avg_turns

        summary = format_eval_specs(specs)
        logging.info("[Eval R=%d] step=%d n=%d eps=%d [%s]",
                     R, step, len(specs), self.args.eval_episodes, summary)

        eval_env = self.eval_envs[R]
        for i in range(min(len(specs), eval_env.num_envs)):
            spec = specs[i]
            eval_env.env_method("set_opponent", spec, indices=[i])
            opp_region = spec.get("opp_region")
            if opp_region is None:
                opp_region = random.choice([r for r in self._regions if r != R])
            eval_env.env_method("set_capitals", R, opp_region, indices=[i])

        results = evaluate(ckpt, eval_env, self.args.eval_episodes, specs)

        by_type: dict[str, list] = {}
        for r in results:
            t = r.opponent_spec["type"]
            if t == "policy":
                continue
            by_type.setdefault(t, []).append(r)
        for opp_type, group in by_type.items():
            self.log_eval_metrics({
                f"vs_{opp_type}/win_rate": aggregate_win_rate(group),
                f"vs_{opp_type}/avg_turns": aggregate_avg_turns(group),
            }, step, region=R)

        return results

    def choose_eval_opponents(self, include_fixed: bool = True, region: int | None = None) -> list[dict]:
        """从 RegionPool 采样 eval 对手 + 可选固定对手。"""
        eval_n_envs = self.args.eval_n_envs or self.args.n_envs
        eval_n_opponents = self.args.eval_n_opponents or self.args.n_opponents or self.args.n_envs
        opponent_id = self._opponent_id()
        R = region if region is not None else self._regions[0]

        if not self.pool.available_regions():
            pool_specs = eval_n_envs * [
                {"type": self.args.self_play_initial_opponent, "player_id": opponent_id},
            ]
        else:
            per_opponent = max(1, eval_n_envs // eval_n_opponents)
            specs = self._sample_opponent_specs(self.pool, eval_n_opponents, opponent_id,
                                                exclude_region=R)
            pool_specs = []
            for spec in specs:
                pool_specs.extend([spec] * per_opponent)
            pool_specs = pool_specs[:eval_n_envs]

        if include_fixed and self.args.eval_opponent:
            half = max(1, eval_n_envs // 2)
            pool_specs = pool_specs[:half]
            fixed_specs = self._fixed_opponent_specs(eval_n_envs - half, opponent_id)
            pool_specs = pool_specs + fixed_specs

        for spec in pool_specs:
            if spec.get("opp_region") is None:
                spec["opp_region"] = random.choice([r for r in self._regions if r != R])
        return pool_specs


    def log_eval_metrics(self, metrics: dict, step: int, region: int | None = None) -> None:
        if self.args.wandb:
            import wandb
            with self._log_lock:
                prefix = f"region_{region}_eval/" if region is not None else "eval/"
                data = {"global_step": step}
                data.update({prefix + k: v for k, v in metrics.items() if v is not None})
                wandb.log(data)
        else:
            logging.info("step=%d region=%s %s", step, region, metrics)
